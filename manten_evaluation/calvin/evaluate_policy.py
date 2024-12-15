# ruff: noqa: UP007

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from manten_evaluation.calvin.pcd_obs_utils import compute_pcd_as_part_of_obs
from manten_evaluation.calvin.rollout_video import RolloutVideo

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

import einops
import hydra
import pyhash
import torch.distributed as dist
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_env.envs.play_table_env import get_env
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
from tqdm.auto import tqdm

hasher = pyhash.fnv1_64()
logger = logging.getLogger(__name__)


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # env = Wrapper(env)
    return env


def get_video_tag(i):
    if dist.is_available() and dist.is_initialized():
        i = i * dist.get_world_size() + dist.get_rank()
    return f"_long_horizon/sequence_{i}"


def evaluate_policy(
    model,
    env,
    calvin_models_dir,
    epoch=42,
    eval_log_dir=None,
    debug=False,
    num_sequences=1000,
    num_videos=0,
    **kwargs,
):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_models_dir) / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    if num_videos > 0:
        rollout_video = RolloutVideo(
            logger=logger, empty_cache=False, log_to_file=True, save_dir=eval_log_dir
        )
    else:
        rollout_video = None

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(num_sequences)

    results = []

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for seq_idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        if seq_idx < num_videos:
            caption = " -> ".join(eval_sequence)
            rollout_video.new_video(tag=get_video_tag(seq_idx), caption=caption)
        result = evaluate_sequence(
            env=env,
            model=model,
            task_checker=task_oracle,
            initial_state=initial_state,
            eval_sequence=eval_sequence,
            val_annotations=val_annotations,
            debug=debug,
            rollout_video=rollout_video if seq_idx < num_videos else None,
            **kwargs,
        )
        results.append(result)
        if seq_idx < num_videos:
            rollout_video.write_to_tmp()
        if not debug:
            eval_sequences.set_description(
                " ".join(
                    [
                        f"{i + 1}/5 : {v * 100:.1f}% |"
                        for i, v in enumerate(count_success(results))
                    ]
                )
                + "|"
            )

    if num_videos > 0:
        rollout_video._log_videos_to_file(0)  # noqa: SLF001
    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(
    env,
    task_checker,
    initial_state,
    eval_sequence,
    debug,
    rollout_video: Optional[RolloutVideo] = None,
    **kwargs,
):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        if rollout_video is not None:
            rollout_video.new_subtask()
        success = rollout(
            env=env,
            task_oracle=task_checker,
            subtask=subtask,
            debug=debug,
            rollout_video=rollout_video,
            **kwargs,
        )
        if rollout_video is not None:
            rollout_video.draw_outcome(success)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(  # noqa: C901
    env,
    model,
    task_oracle,
    subtask,
    val_annotations,
    debug,
    ep_len=360,
    rollout_video: Optional[RolloutVideo] = None,
    pcd_as_part_of_obs=False,
):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    if pcd_as_part_of_obs:
        obs = compute_pcd_as_part_of_obs(obs, env)
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    for _step in range(ep_len):
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        if pcd_as_part_of_obs:
            obs = compute_pcd_as_part_of_obs(obs, env)
        if debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if rollout_video is not None:
            rollout_video.update(
                einops.rearrange(
                    torch.from_numpy(obs["rgb_obs"]["rgb_static"]), "h w c -> 1 1 c h w"
                )
            )

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {subtask}
        )
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            if rollout_video is not None:
                rollout_video.add_language_instruction(lang_annotation)
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
    if rollout_video is not None:
        rollout_video.add_language_instruction(lang_annotation)
    return False


@hydra.main(config_path="configs", config_name="evaluate_policy")
def main(cfg: DictConfig):
    seed_everything(0, workers=True)

    model = hydra.utils.instantiate(cfg.agent_proxy)

    env = make_env(cfg.dataset_path)
    evaluate_policy(
        model,
        env,
        calvin_models_dir=cfg.calvin_models_dir,
        debug=cfg.debug,
        eval_log_dir=cfg.eval_log_dir,
        num_sequences=cfg.num_sequences,
        ep_len=cfg.ep_len,
        num_videos=cfg.num_videos,
        pcd_as_part_of_obs=cfg.pcd_as_part_of_obs,
    )


if __name__ == "__main__":
    main()
