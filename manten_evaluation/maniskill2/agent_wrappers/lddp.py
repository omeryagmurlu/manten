import einops
import torch
from optree import tree_flatten, tree_map, tree_unflatten

from manten_evaluation.maniskill2.lib.utils_maniskill_common import (
    process_observation_from_raw,
)


class LDDPAgentWrapper:
    def __init__(self, *, agent, obs_mode, device="cuda"):
        self.__agent = agent
        self.__obs_mode = obs_mode
        self.__device = device

    def __getattr__(self, attr):
        return getattr(self.__agent, attr)

    def __obs_dict(self, obs):
        return self.__map_dict_over_history_with_batches(
            obs, lambda ob: process_observation_from_raw(ob, self.__obs_mode)
        )

    @staticmethod
    def __map_dict_over_history_with_batches(obs, fn):
        # takes an input observation dict, each leaf tensor of shape (bs, nhist, ...)
        # converts it to a list (#nhist) of observation dicts, each leaf tensor of shape (bs, ...)
        # applies transformation to each observation dict
        # stacks the resulting mapped observation dicts into a single, mapped observation dict with leaf tensors of shape (bs, nhist, ...)

        # future me: vmap, VMAP! over history

        elems, in_obs_spec = tree_flatten(obs)
        elems_hists = [[elem[:, t] for t in range(elem.shape[1])] for elem in elems]
        hists_elems = LDDPAgentWrapper.transpose_list_of_lists(elems_hists)
        hists_obs = [tree_unflatten(in_obs_spec, elems) for elems in hists_elems]
        hists_mapped_obs = [fn(ob) for ob in hists_obs]
        hists_mapped_elems = [tree_flatten(mapped_obs) for mapped_obs in hists_mapped_obs]
        out_obs_spec = hists_mapped_elems[0][1]
        hists_mapped_elems = [elem[0] for elem in hists_mapped_elems]
        mapped_elems_hists = LDDPAgentWrapper.transpose_list_of_lists(hists_mapped_elems)
        mapped_elems_stacked = [
            einops.rearrange(elem, "t b ... -> b t ...") for elem in mapped_elems_hists
        ]

        return tree_unflatten(out_obs_spec, mapped_elems_stacked)

    @staticmethod
    def transpose_list_of_lists(outer):
        return [[inner[i] for inner in outer] for i in range(len(outer[0]))]

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
