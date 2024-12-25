from collections import defaultdict

import gymnasium as gym
import mani_skill.envs  # noqa: F401 # loads up the envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import CPUGymWrapper, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from tqdm import tqdm


def make_eval_envs(  # noqa: C901
    env_id,
    *,
    num_envs: int,
    sim_backend: str,
    env_kwargs: dict,
    video_dir: str | None = None,
    save_video: bool = False,
    wrappers: list[gym.Wrapper] | None = None,
):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
    """
    if wrappers is None:
        wrappers = []

    if not save_video:
        video_dir = None

    if sim_backend == "cpu":

        def cpu_make_env(env_id, seed, video_dir=None, env_kwargs=None):
            if env_kwargs is None:
                env_kwargs = {}

            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if video_dir:
                    env = RecordEpisode(
                        env,
                        output_dir=video_dir,
                        save_trajectory=False,
                        info_on_video=True,
                        source_type="diffusion_policy",
                        source_desc="diffusion_policy evaluation rollout",
                    )
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk

        vector_cls = (
            gym.vector.SyncVectorEnv
            if num_envs == 1
            else lambda x: gym.vector.AsyncVectorEnv(x, context="forkserver")
        )
        env = vector_cls(
            [
                cpu_make_env(env_id, seed, video_dir if seed == 0 else None, env_kwargs)
                for seed in range(num_envs)
            ]
        )
    else:
        env = gym.make(
            env_id,
            num_envs=num_envs,
            sim_backend=sim_backend,
            reconfiguration_freq=1,
            **env_kwargs,
        )
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        if video_dir:
            env = RecordEpisode(
                env,
                output_dir=video_dir,
                save_trajectory=False,
                save_video=True,
                source_type="diffusion_policy",
                source_desc="diffusion_policy evaluation rollout",
                max_steps_per_video=max_episode_steps,
            )
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env


def evaluate_via_agent(agent, envs, num_eval_episodes, sim_backend, progress_bar=False):  # noqa: C901
    eval_metrics = defaultdict(list)
    obs, info = envs.reset()
    eps_count = 0
    if progress_bar:
        pbar = tqdm(total=num_eval_episodes)
        pbar.set_description("maniskill evaluation")
    # i = 0
    while eps_count < num_eval_episodes:
        # i += 1
        # obs[:, 0] = i
        action_seq = agent.step(obs)
        if sim_backend == "cpu":
            action_seq = action_seq.cpu().numpy()

        for i in range(action_seq.shape[1]):
            obs, rew, terminated, truncated, info = envs.step(action_seq[:, i])
            if truncated.any():
                break

        if truncated.any():
            assert (
                truncated.all() == truncated.any()
            ), "all episodes should truncate at the same time for fair evaluation with other algorithms"
            if isinstance(info["final_info"], dict):
                for k, v in info["final_info"]["episode"].items():
                    eval_metrics[k].append(v.float().cpu().numpy())
            else:
                for final_info in info["final_info"]:
                    for k, v in final_info["episode"].items():
                        eval_metrics[k].append(v)
            eps_count += envs.num_envs
            if progress_bar:
                pbar.update(envs.num_envs)

    return eval_metrics
