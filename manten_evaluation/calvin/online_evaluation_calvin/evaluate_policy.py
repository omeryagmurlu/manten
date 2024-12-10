# ruff: noqa

"""Modified from
https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/evaluate_policy.py
"""

import gc
import logging
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from manten_evaluation.calvin.online_evaluation_calvin.common_utils import (
    get_gripper_loc_bounds,
)
from manten.utils.utils_root import root
from manten_evaluation.calvin.manten_calvin_agent_proxy_client import create_model
from manten_evaluation.calvin.online_evaluation_calvin.evaluate_utils import (
    collect_results,
    count_success,
    get_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    prepare_proprio_states,
    prepare_visual_states,
    write_results,
)
from manten_evaluation.calvin.online_evaluation_calvin.multistep_sequences import (
    get_sequences,
)

logger = logging.getLogger(__name__)

EP_LEN = 60
EXECUTE_LEN = 20


def make_env(dataset_path, show_gui=True, split="validation", scene=None):
    val_folder = Path(dataset_path) / f"{split}"
    if scene is not None:
        env = get_env(val_folder, show_gui=show_gui, scene=scene)
    else:
        env = get_env(val_folder, show_gui=show_gui)

    return env


def evaluate_policy(
    model,
    env,
    conf_dir,
    eval_log_dir=None,
    save_video=False,
    sequence_indices=[],
    num_sequences=1000,
):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: an instance of CalvinBaseModel
        env: an instance of CALVIN_ENV
        conf_dir: Path to the directory containing the config files of CALVIN
        eval_log_dir: Path where to log evaluation results
        save_video: a boolean indicates whether to save the video
        sequence_indices: a list of integers indicates the indices of the
            instruction chains to evaluate

    Returns:
        results: a list of integers indicates the number of tasks completed
    """
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(num_sequences)

    results, tested_sequence_indices = collect_results(eval_log_dir)

    for seq_ind, (initial_state, eval_sequence) in enumerate(eval_sequences):
        if sequence_indices and seq_ind not in sequence_indices:
            continue
        if seq_ind in tested_sequence_indices:
            continue
        result, videos = evaluate_sequence(
            env,
            model,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            save_video,
        )
        write_results(eval_log_dir, seq_ind, result)
        results.append(result)
        str_results = (
            " ".join(
                [
                    f"{i + 1}/5 : {v * 100:.1f}% |"
                    for i, v in enumerate(count_success(results))
                ]
            )
            + "|"
        )
        print(str_results + "\n")

        if save_video:
            import moviepy.video.io.ImageSequenceClip

            clip = []

            for task_ind, (subtask, video) in enumerate(
                zip(eval_sequence, videos, strict=False)
            ):
                for img_ind, img in enumerate(video):
                    # cv2.putText(
                    #     img,
                    #     f"{task_ind}: {subtask}",
                    #     (10, 180),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.5,
                    #     (0, 0, 0),
                    #     1,
                    #     2,
                    # )
                    video[img_ind] = img
                clip.extend(video)
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(clip, fps=30)
            clip.write_videofile(f"calvin_seq{seq_ind}.mp4")

    return results


def evaluate_sequence(
    env, model, task_checker, initial_state, eval_sequence, val_annotations, save_video
):
    """
    Evaluates a sequence of language instructions.

    Args:
        env: an instance of CALVIN_ENV
        model: an instance of CalvinBaseModel
        task_checker: an indicator of whether the current task is completed
        initial_state: a tuple of `robot_obs` and `scene_obs`
            see: https://github.com/mees/calvin/blob/main/dataset/README.md#state-observation
        eval_sequence: a list indicates the instruction chain
        val_annotations: a dictionary of task instructions
        save_video: a boolean indicates whether to save the video

    Returns:
        success_counter: an integer indicates the number of tasks completed
        video_aggregator: a list of lists of images that shows the trajectory
            of the robot

    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter, video_aggregators = 0, []
    for subtask in eval_sequence:
        # get lang annotation for subtask
        lang_annotation = val_annotations[subtask][0]
        success, video = rollout(env, model, task_checker, subtask, lang_annotation)
        video_aggregators.append(video)

        if success:
            success_counter += 1
        else:
            return success_counter, video_aggregators
    return success_counter, video_aggregators


def rollout(env, model, task_oracle, subtask, lang_annotation):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).

    Args:
        env: an instance of CALVIN_ENV
        model: an instance of CalvinBaseModel
        task_oracle: an indicator of whether the current task is completed
        subtask: a string indicates the task name
        lang_annotation: a string indicates the instruction of the task

    Returns:
        Success/Fail: a boolean indicates whether the task is completed
        video: a list of images that shows the trajectory of the robot
    """
    video = []  # show video for debugging
    obs = env.get_obs()

    model.reset()
    start_info = env.get_info()

    print("------------------------------")
    print(f"task: {lang_annotation}")
    video.append(obs["rgb_obs"]["rgb_static"])

    pbar = tqdm(range(EP_LEN))
    for step in pbar:
        obs = prepare_visual_states(obs, env)
        obs = prepare_proprio_states(obs, env)
        lang_embeddings = model.encode_instruction(lang_annotation)
        with torch.cuda.amp.autocast():
            trajectory = model.step(obs, lang_embeddings)
        for act_ind in range(min(trajectory.shape[1], EXECUTE_LEN)):
            # calvin_env executes absolute action in the format of:
            # [[x, y, z], [euler_x, euler_y, euler_z], [open]]
            curr_action = [
                trajectory[0, act_ind, :3],
                trajectory[0, act_ind, 3:6],
                trajectory[0, act_ind, [6]],
            ]
            pbar.set_description(f"step: {step}")
            curr_proprio = obs["proprio"]
            obs, _, _, current_info = env.step(curr_action)
            obs["proprio"] = curr_proprio

            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(
                start_info, current_info, {subtask}
            )

            video.append(obs["rgb_obs"]["rgb_static"])

            if len(current_task_info) > 0:
                return True, video

    return False, video


def get_calvin_gripper_loc_bounds(file_path):
    with open(file_path) as stream:
        bounds = yaml.safe_load(stream)
        min_bound = bounds["act_min_bound"][:3]
        max_bound = bounds["act_max_bound"][:3]
        gripper_loc_bounds = np.stack([min_bound, max_bound])

    return gripper_loc_bounds


def main(cfg):
    # These location bounds are extracted from language-annotated episodes
    # if cfg.gripper_loc_bounds is None:
    #     gripper_loc_bounds = np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0
    # else:
    #     gripper_loc_bounds = get_gripper_loc_bounds(
    #         cfg.gripper_loc_bounds,
    #         task=cfg.tasks[0] if len(cfg.tasks) == 1 else None,
    #         buffer=cfg.gripper_loc_bounds_buffer,
    #     )

    # These location bounds are extracted from every episode in play trajectory
    if cfg.calvin_gripper_loc_bounds is not None:
        calvin_gripper_loc_bounds = get_calvin_gripper_loc_bounds(
            cfg.calvin_gripper_loc_bounds
        )
    else:
        calvin_gripper_loc_bounds = None

    # set random seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # evaluate a custom model
    model = create_model(
        **cfg.calvin_agent_proxy_client, calvin_gripper_loc_bounds=calvin_gripper_loc_bounds
    )

    if os.environ.get("LOCAL_RANK") is not None and os.environ.get("WORLD_SIZE") is not None:
        sequence_indices = [
            i
            for i in range(
                int(os.environ["LOCAL_RANK"]),
                cfg.num_sequences,
                int(os.environ["WORLD_SIZE"]),
            )
        ]
    else:
        logger.warning(
            "LOCAL_RANK or WORLD_SIZE not found in environment variables, using single CPU"
        )
        sequence_indices = list(range(cfg.num_sequences))  # disable multi-cpu for now

    env = make_env(cfg.calvin_dataset_path, show_gui=False)
    evaluate_policy(
        model,
        env,
        conf_dir=Path(cfg.calvin_model_path) / "conf",
        eval_log_dir=cfg.base_log_dir,
        sequence_indices=sequence_indices,
        save_video=cfg.save_video,
        num_sequences=cfg.num_sequences,
    )

    results, sequence_inds = collect_results(cfg.base_log_dir)
    str_results = (
        " ".join(
            [f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]
        )
        + "|"
    )
    print(f"Load {len(results)}/1000 episodes...")
    print(str_results + "\n")

    del env
    gc.collect()


@hydra.main(
    version_base=None,
    config_path=str(root / "manten_evaluation" / "calvin" / "configs"),
    config_name="evaluate_policy",
)
def run(cfg):
    main(cfg)


if __name__ == "__main__":
    # args = Arguments().parse_args()
    # args.local_rank = int(os.environ["LOCAL_RANK"])

    # # DDP initialization
    # torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend="nccl", init_method="env://")
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    run()
