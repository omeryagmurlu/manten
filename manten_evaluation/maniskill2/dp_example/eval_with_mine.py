# ruff: noqa

ALGO_NAME = "BC_Diffusion_state_UNet"

import functools
import os
from pathlib import Path
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from optree import tree_map
import torch
import tyro

# from diffusion_policy.evaluate import evaluate
# from diffusion_policy.make_env import make_eval_envs

from manten.utils.utils_checkpointing import load_agent
from manten_evaluation.maniskill2.utils.utils_evaluation import (
    HorizonActionExecutor,
    TreeFrameStack,
    evaluate_via_agent,
    make_eval_envs,
)


@dataclass
class Args:
    obs_horizon: int = 2  # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = 16  # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal

    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v1"
    """the id of the environment"""

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = 350
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    num_eval_episodes: int = 30
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    control_mode: str = "pd_ee_delta_pose"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"evalmine_{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    Path(f"runs/{run_name}/videos").mkdir(parents=True, exist_ok=True)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    agent = load_agent(
        train_folder="/home/i53/student/yagmurlu/code/manten/outputs/training/manten_maniskill_PegInsertionSide-v1/2024-12-25/21-11-33/accelerate",
        use_ema=True,
        device=device,
    )
    agent.eval()

    class AgentWrapper:
        def __init__(self, agent, device):
            self.__agent = agent
            self.__device = device

        def step(self, obs):
            obs = tree_map(
                lambda x: x.to(self.__device)
                if isinstance(x, torch.Tensor)
                else torch.tensor(x, device=self.__device),
                obs,
            )
            batch = {"observations": {"state_obs": obs}}
            _metric, trajectory = self.__agent("eval", batch)
            return trajectory

        def __getattr__(self, attr):
            return getattr(self.__agent, attr)

    agent = AgentWrapper(agent, device)

    print("Agent loaded, creating evaluation environments...")

    # env setup
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array",
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    envs = make_eval_envs(
        args.env_id,
        num_envs=args.num_eval_envs,
        sim_backend=args.sim_backend,
        env_kwargs=env_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        save_video=args.capture_video,
        wrappers=[functools.partial(TreeFrameStack, num_stack=args.obs_horizon)],
    )

    # Evaluation
    eval_metrics = evaluate_via_agent(
        agent,
        envs,
        num_eval_episodes=args.num_eval_episodes,
        sim_backend=args.sim_backend,
        progress_bar=True,
        aex=HorizonActionExecutor(),
    )

    print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
    for k in eval_metrics.keys():
        eval_metrics[k] = np.mean(eval_metrics[k])
        print(f"{k}: {eval_metrics[k]:.4f}")

    envs.close()
