from types import MappingProxyType

import einops
import torch
from optree import tree_flatten, tree_map, tree_unflatten

from manten_evaluation.maniskill2.lib.utils_maniskill_common import (
    process_observation_from_raw,
)

FULL_OBS_MODALITIES = MappingProxyType(
    {
        "pointcloud": ("rgb_obs", "pcd_obs", "pcd_mask", "state_obs"),
        "rgb": ("rgb_obs", "state_obs"),
        "state": ("state_obs",),
    }
)


class LDDPAgentWrapper:
    def __init__(
        self,
        *,
        agent,
        obs_mode,
        obs_modalities=None,
        state_modality_keys=None,
        rgb_modality_keys=None,
        device="cuda",
    ):
        if obs_modalities is None:
            obs_modalities = FULL_OBS_MODALITIES[obs_mode]
        if state_modality_keys is None:
            state_modality_keys = []
        if rgb_modality_keys is None and obs_mode == "rgb":
            raise ValueError("rgb_modality_keys must be provided for rgb obs_mode")

        self.__agent = agent
        self.__obs_mode = obs_mode
        self.__obs_modalities = obs_modalities
        self.__state_modality_keys = state_modality_keys
        self.__rgb_modality_keys = rgb_modality_keys
        self.__device = device

    def __getattr__(self, attr):
        return getattr(self.__agent, attr)

    def __get_aggregated_state(self, loaded_obs):
        states = [loaded_obs[smod] for smod in self.__state_modality_keys]
        return torch.cat(states, -1)

    def __proc_obs(self, obs):  # noqa: C901, PLR0912
        loaded_obs = process_observation_from_raw(obs, self.__obs_mode)
        obs_dict = {}
        for key in self.__obs_modalities:
            if key == "state_obs":
                if self.__obs_mode == "state":
                    # there is an np array called state_obs
                    obs_dict[key] = loaded_obs[key]
                else:
                    # we need to aggregate the state modalities into one
                    obs_dict[key] = self.__get_aggregated_state(loaded_obs)
            elif key == "rgb_obs":
                if self.__obs_mode == "rgb":
                    _, ncam, h, w, c = loaded_obs[key].shape
                    if ncam != len(self.__rgb_modality_keys):
                        raise ValueError(
                            f"Number of cameras in rgb_obs ({ncam}) does not match the number of rgb_modality_keys ({len(self.__rgb_modality_keys)})"
                        )
                    obs_dict[key] = {
                        k: loaded_obs[key][:, cam_i]
                        for cam_i, k in enumerate(self.__rgb_modality_keys)
                    }
                else:
                    # for pcd the colors are in rgb_obs
                    obs_dict[key] = loaded_obs[key]
            else:
                obs_dict[key] = loaded_obs[key]

        if "pcd_obs" in self.__obs_modalities:
            obs_dict["pcd_mask"] = loaded_obs["pcd_mask"]
            # TODO: also load camera intrinsics/extrinsics for stuff

        if self.__obs_mode == "rgb":
            for cam_key, cam_v in obs_dict["rgb_obs"].items():
                cam = einops.rearrange(cam_v, "t h w c -> t c h w")
                obs_dict["rgb_obs"][cam_key] = cam.float() / 255.0
        elif self.__obs_mode == "pointcloud":
            raise NotImplementedError

        return obs_dict

    def __obs_dict(self, obs):
        return self.__map_dict_over_history_with_batches(obs, self.__proc_obs)

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
        return self.__agent.predict_actions(observations=self.__obs_dict(obs))
