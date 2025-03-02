from collections import defaultdict

import numpy as np
from tqdm import tqdm


def make_eval_envs(  # noqa: C901
    env_id,
    *,
    num_envs: int,
    sim_backend: str,
    env_kwargs: dict,
    video_dir: str | None = None,
    save_video: bool = False,
    wrappers: list | None = None,
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
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 # register envs
    from mani_skill.utils import gym_utils
    from mani_skill.utils.wrappers import CPUGymWrapper, RecordEpisode
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    if wrappers is None:
        wrappers = []

    if not save_video:
        video_dir = None

    if sim_backend == "cpu":

        def cpu_make_env(env_id, seed, video_dir=None, env_kwargs=None):  # noqa: ARG001
            if env_kwargs is None:
                env_kwargs = {}

            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                # env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if video_dir:
                    env = RecordEpisode(
                        env,
                        output_dir=video_dir,
                        save_trajectory=False,
                        info_on_video=True,
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
                max_steps_per_video=max_episode_steps,
            )
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env


def evaluate_via_agent(  # noqa: C901
    agent, envs, num_eval_episodes, sim_backend, progress_bar=False, aex=None
):
    from manten_evaluation.maniskill2.lib.utils_wrappers import ActionExecutor

    if aex is None:
        aex = ActionExecutor()

    eval_metrics = defaultdict(list)
    obs, info = envs.reset()
    eps_count = 0
    if progress_bar:
        pbar = tqdm(total=num_eval_episodes)
        pbar.set_description("maniskill evaluation")
    while eps_count < num_eval_episodes:
        action_seq = agent.step(obs)
        if sim_backend == "cpu":
            action_seq = action_seq.cpu().numpy()

        obs, rew, terminated, truncated, info = aex.execute_action_in_env(action_seq, envs)

        if truncated.any():
            assert truncated.all() == truncated.any(), (
                "all episodes should truncate at the same time for fair evaluation with other algorithms"
            )
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


class ManiskillEvaluation:
    def __init__(
        self,
        *,
        output_dir,
        env_id,
        num_eval_episodes,
        sim_backend,
        num_envs,
        env_kwargs,
        agent_wrapper=None,
        action_executor=None,
        save_video=False,
        wrappers=None,
        progress_bar=True,
        name_extension="",
    ):
        if wrappers is None:
            wrappers = []

        self.agent_wrapper = agent_wrapper
        self.output_dir = output_dir
        self.num_eval_episodes = num_eval_episodes
        self.sim_backend = sim_backend
        self.num_envs = num_envs
        self.env_kwargs = env_kwargs
        self.action_executor = action_executor
        self.save_video = save_video
        self.wrappers = wrappers
        self.env_id = env_id
        self.progress_bar = progress_bar
        self.name_extension = name_extension

    def evaluate(self, agent):
        if self.agent_wrapper is not None:
            agent = self.agent_wrapper(agent=agent)

        # load lazy imports
        wrappers = [wrapper() for wrapper in self.wrappers]
        aex = self.action_executor() if self.action_executor is not None else None

        envs = make_eval_envs(
            env_id=self.env_id,
            num_envs=self.num_envs,
            sim_backend=self.sim_backend,
            env_kwargs=self.env_kwargs,
            wrappers=wrappers,
            save_video=self.save_video,
            video_dir=self.output_dir,
        )

        outputs = evaluate_via_agent(
            agent=agent,
            envs=envs,
            num_eval_episodes=self.num_eval_episodes,
            sim_backend=self.sim_backend,
            progress_bar=self.progress_bar,
            aex=aex,
        )

        envs.close()
        # TODO: this doesn't work properly for CPU envs for now,  # This pretty low priority
        # it crashes ~10 mins after the evaluation is done with some
        # weird vulkan error, so just run online eval only once at the end
        # for now

        infos = {k: np.array(v) for k, v in outputs.items()}
        rich_media = {}
        if self.save_video:
            rich_media["videos"] = {
                f"video_{num}": f"{self.output_dir}/{num}.mp4"
                for num in range(self.num_eval_episodes // self.num_envs)
            }
        return infos, rich_media

    @property
    def eval_name(self):
        return f"eval-maniskill-{self.env_id}-{self.name_extension}"


if __name__ == "__main__":
    import gymnasium as gym

    env_kwargs = {
        "control_mode": "pd_ee_delta_pose",
        "reward_mode": "sparse",
        "obs_mode": "rgb",
        "render_mode": "rgb_array",
        # max_episode_steps="300",
        "max_episode_steps": 100,
    }

    env = gym.make("PickCube-v1", reconfiguration_freq=1, **env_kwargs)

    print(env.observation_space)
