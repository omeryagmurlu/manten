from collections import deque

import einops
import torch
from optree import tree_map

from manten_evaluation.maniskill2.utils_maniskill_common import process_observation_from_raw


class LDDPAgentWrapper:
    def __init__(self, *, agent, obs_horizon, obs_mode, device="cuda"):
        self.__agent = agent
        self.__obs_horizon = obs_horizon
        self.__obs_mode = obs_mode
        self.__device = device

    def __getattr__(self, attr):
        return getattr(self.__agent, attr)

    def __add_to_obs_stack_and_get_obs_hist(self, obs_sing):
        # from dataloader: bs, hist, dim
        # from dataset: hist, dim
        # from npz: epi, dim

        # from agent_wrapper (here): bs (prolly 1), hist, dim
        # from env: bs (prolly 1), dim

        if not hasattr(self, "_LDDPAgentWrapper__obs_hist"):  # CPython name mangling
            self.__obs_hist = tree_map(lambda _: deque(maxlen=self.__obs_horizon), obs_sing)
            for _ in range(self.__obs_horizon):
                for k, v in obs_sing.items():
                    self.__obs_hist[k].append(v)

        for k, v in obs_sing.items():
            self.__obs_hist[k].append(v)  # append is the correct one [obs1, obs2...]

        return tree_map(
            lambda x: einops.rearrange(list(x), "hist bs ... -> bs hist ..."),
            self.__obs_hist,
            is_leaf=lambda x: isinstance(x, deque),
        )

    def __obs_dict(self, obs):
        obs_sing = process_observation_from_raw(obs, obs_mode=self.__obs_mode)
        obs_hist = self.__add_to_obs_stack_and_get_obs_hist(obs_sing)

        return obs_hist

    def step(self, obs):
        obs = tree_map(
            lambda x: x.to(self.__device)
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, device=self.__device),
            obs,
        )
        batch = {"observations": self.__obs_dict(obs)}
        _metric, trajectory = self.__agent("eval", batch)
        return trajectory
