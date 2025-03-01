from types import MappingProxyType

import einops
import torch
from optree import tree_flatten, tree_map, tree_unflatten

from manten.networks.utils.rotation_transformer import RotationTransformer
from manten_evaluation.maniskill2.lib.utils_maniskill_common import (
    apply_static_transforms,
    back_transform_episode_actions,
    process_observation_from_raw,
    transform_episode_obs,
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
        rotation_transform=None,
        meta_2d_3d_mask=None,
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

        if rotation_transform is not None:
            self.__rotation_transformer = RotationTransformer(
                from_rep="euler_angles", to_rep=rotation_transform, from_convention="XYZ"
            )
        else:
            self.__rotation_transformer = None

        T = torch.Tensor([True]).bool().to("cuda")
        F = torch.Tensor([False]).bool().to("cuda")

        if meta_2d_3d_mask is None:
            self.__meta = {}
        elif meta_2d_3d_mask == "2d":
            self.__meta = {"3d_mask": F, "2d_mask": T}
        elif meta_2d_3d_mask == "3d":
            self.__meta = {"3d_mask": T, "2d_mask": T}
        else:
            raise ValueError(f"Invalid meta_2d_3d_mask: {meta_2d_3d_mask}")

    def __getattr__(self, attr):
        return getattr(self.__agent, attr)

    def __proc_obs(self, obs):
        loaded_obs = process_observation_from_raw(
            obs, self.__obs_mode, rgb_modality_keys=self.__rgb_modality_keys
        )
        obs_dict = transform_episode_obs(
            loaded_obs,
            obs_mode=self.__obs_mode,
            obs_modalities=self.__obs_modalities,
            rgb_modality_keys=self.__rgb_modality_keys,
            state_modality_keys=self.__state_modality_keys,
        )
        obs_dict = apply_static_transforms(obs_dict, obs_mode=self.__obs_mode)
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

    def __proc_actions(self, actions):
        actions = back_transform_episode_actions(
            actions, rotation_transformer=self.__rotation_transformer
        )
        return actions

    def step(self, obs):
        obs = tree_map(
            lambda x: x.to(self.__device)
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, device=self.__device),
            obs,
        )
        if len(self.__meta) == 0:
            actions = self.__agent.predict_actions(observations=self.__obs_dict(obs))
        else:
            actions = self.__agent.predict_actions(
                observations=self.__obs_dict(obs), meta=self.__meta
            )

        actions = self.__proc_actions(actions)
        return actions


if __name__ == "__main__":
    import functools

    from manten_evaluation.maniskill2.lib.evaluation import make_eval_envs
    from manten_evaluation.maniskill2.lib.utils_wrappers import TreeFrameStack

    env = make_eval_envs(
        env_id="PickCube-v1",
        num_envs=2,
        sim_backend="cpu",
        wrappers=[functools.partial(TreeFrameStack, num_stack=2)],
        env_kwargs={
            "control_mode": "pd_ee_delta_pose",
            "reward_mode": "sparse",
            "obs_mode": "rgb",
            "render_mode": "rgb_array",
            # max_episode_steps="300",
            "max_episode_steps": 100,
        },
    )

    class AgentDummy:
        def predict_actions(self, *, observations):  # noqa: ARG002
            act = torch.randn((1, 8, 10))
            print(act.shape)
            return act

    agent = LDDPAgentWrapper(
        agent=AgentDummy(),
        obs_mode="rgb",
        state_modality_keys=["goal_pos"],
        rgb_modality_keys=["camera1"],
        rotation_transform="rotation_6d",
    )

    obs, info = env.reset()
    print(obs)
    actions = agent.step(obs)
    print(actions.shape)
    obs, rew, _, _, info = env.step(actions.cpu().numpy())  # not a vec env?
    print(obs, rew, info)
