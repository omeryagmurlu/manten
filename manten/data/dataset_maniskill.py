import functools
from functools import partial
from types import MappingProxyType

import einops
import numpy as np
import torch
from optree import tree_map
from torch.utils.data import Dataset
from tqdm import tqdm

from manten.utils.utils_data import modulo_dataset


@modulo_dataset
class ManiSkillDataset(Dataset):
    OBS_MODE_FILE_NAMES = MappingProxyType(
        {
            "pointcloud": "pointcloud",
            "rgb": "pointcloud",  # has the same info, so just don't bother and use data in pointcloud
            "state": "state",
        }
    )
    FULL_OBS_MODALITIES = MappingProxyType(
        {
            "pointcloud": ("rgb_obs", "pcd_obs", "pcd_mask", "state_obs"),
            "rgb": ("rgb_obs", "state_obs"),
            "state": ("state_obs",),
        }
    )

    def __init__(
        self,
        *,
        train=True,
        test_ratio=0.01,
        # ...
        pack_root,
        task,
        obs_mode,
        obs_modalities=None,
        state_modality_keys=None,
        rgb_modality_keys=None,
        control_mode,
        # ...
        obs_horizon,
        pred_horizon,
        # ...
        load_count=None,
        use_mmap=False,
    ):
        if obs_modalities is None:
            obs_modalities = self.FULL_OBS_MODALITIES[obs_mode]
        if state_modality_keys is None:
            state_modality_keys = []
        if rgb_modality_keys is None and obs_mode == "rgb":
            raise ValueError("rgb_modality_keys must be provided for rgb obs_mode")

        self.obs_mode = obs_mode
        self.obs_modalities = obs_modalities
        self.state_modality_keys = state_modality_keys
        self.rgb_modality_keys = rgb_modality_keys

        path_obs_mode = self.OBS_MODE_FILE_NAMES[obs_mode]

        self.paths = {
            "action_lengths": f"{pack_root}/{task}/{path_obs_mode}/{control_mode}/traj_lengths.npy",
            "episode_format": partial(
                "{pack_root}/{task}/{path_obs_mode}/{control_mode}/episode_{episode_idx}/{key}.npy".format,
                pack_root=pack_root,
                task=task,
                path_obs_mode=path_obs_mode,
                control_mode=control_mode,
            ),
        }

        self.episode_cache = {}

        action_lengths = np.load(self.paths["action_lengths"])
        action_lengths = action_lengths[np.argsort(action_lengths[:, 0])]
        if load_count is not None:
            action_lengths = action_lengths[:load_count]

        split = int(len(action_lengths) * (1 - test_ratio))
        if train:
            action_lengths = action_lengths[:split]
        else:
            action_lengths = action_lengths[split:]

        self.pad_action_arm = None

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.slices = []
        total_transitions = 0
        for (
            episode_idx,
            action_length,
        ) in action_lengths:  # for each ep, so 30 in demo, do sliding windows
            total_transitions += action_length

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (episode_idx.item(), start, start + pred_horizon)
                for start in range(-pad_before, action_length - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.use_mmap = use_mmap
        for idx in tqdm(range(len(action_lengths))):
            self.get_episode(idx)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        episode = self.get_episode(traj_idx)
        L, act_dim = episode["actions"].shape

        obs_dict = tree_map(
            lambda obs: torch.tensor(obs[max(0, start) : start + self.obs_horizon]),
            episode["observations"],
        )
        obs_dict = self.apply_static_transforms(obs_dict, self.obs_mode)
        # start+self.obs_horizon is at least 1
        act_seq = torch.tensor(episode["actions"][max(0, start) : end])
        if start < 0:  # pad before the trajectory
            obs_dict = tree_map(
                lambda obs: torch.cat(
                    [einops.repeat(obs[0], "... -> k ...", k=-start), obs], dim=0
                ),
                obs_dict,
            )
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            gripper_action = act_seq[-1, -1]
            if self.pad_action_arm is None:
                self.pad_action_arm = torch.zeros((act_dim - 1,))
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert act_seq.shape[0] == self.pred_horizon

        # |o|o|                             observations: 2
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        #
        # | |a|a|a|a|a|a|a|a|               actions executed: 8
        return {
            "observations": obs_dict,
            "actions": act_seq,
        }

    @staticmethod
    def apply_static_transforms(obs_dict, obs_mode):
        if obs_mode == "rgb":
            for cam_key, cam_v in obs_dict["rgb_obs"].items():
                cam = einops.rearrange(cam_v, "t h w c -> t c h w")
                obs_dict["rgb_obs"][cam_key] = cam.float() / 255.0
        elif obs_mode == "pointcloud":
            raise NotImplementedError
            # use transformpcd from pointcloudmatters

        return obs_dict

    @functools.cache  # noqa: B019
    def get_episode(self, idx):
        epf = partial(self.paths["episode_format"], episode_idx=idx)
        load = partial(np.load, mmap_mode="r" if self.use_mmap else None)

        obs_dict = {}
        for key in self.obs_modalities:
            if key == "state_obs":
                if self.obs_mode == "state":
                    # there is an np array called state_obs
                    obs_dict[key] = load(epf(key=key))
                else:
                    # we need to aggregate the state modalities into one
                    obs_dict[key] = self.get_aggregated_state(load, epf)
            elif key == "rgb_obs":
                loaded = load(epf(key=key))
                if self.obs_mode == "rgb":
                    _, ncam, h, w, c = loaded.shape
                    if ncam != len(self.rgb_modality_keys):
                        raise ValueError(
                            f"Number of cameras in rgb_obs ({ncam}) does not match the number of rgb_modality_keys ({len(self.rgb_modality_keys)})"
                        )
                    obs_dict[key] = {
                        k: loaded[:, cam_i] for cam_i, k in enumerate(self.rgb_modality_keys)
                    }
                else:
                    # for pcd the colors are in rgb_obs
                    obs_dict[key] = loaded
            else:
                obs_dict[key] = load(epf(key=key))

        if "pcd_obs" in self.obs_modalities:
            obs_dict["pcd_mask"] = load(epf(key="pcd_mask"))
            # TODO: also load camera intrinsics/extrinsics for stuff

        return {
            "actions": load(epf(key="actions")),
            "observations": obs_dict,
        }

    def get_aggregated_state(self, load, epf):
        if not self.state_modality_keys:
            raise ValueError("No state modality keys provided")
        if len(self.state_modality_keys) == 1:
            # keep mmap untouched if not needed
            return load(epf(key=self.state_modality_keys[0]))

        states = [load(epf(key=smod)) for smod in self.state_modality_keys]
        return np.concatenate(states, axis=-1)

    @functools.cache  # noqa: B019
    def get_dataset_info(self):
        all_actions = []
        for idx in range(len(self.paths["action_lengths"])):
            episode = self.get_episode(idx)
            all_actions.append(episode["actions"])

        all_actions = np.concatenate(all_actions, axis=0)
        actions_stats = {
            "p01": np.percentile(all_actions, 1, axis=0).tolist(),
            "p99": np.percentile(all_actions, 99, axis=0).tolist(),
            "min": np.min(all_actions, axis=0).tolist(),
            "max": np.max(all_actions, axis=0).tolist(),
            "mean": np.mean(all_actions, axis=0).tolist(),
            "std": np.std(all_actions, axis=0).tolist(),
        }

        sample_batch = self[0]

        infos = {
            "actions_stats": actions_stats,
            "obs_horizon": self.obs_horizon,
            "pred_horizon": self.pred_horizon,
            "actions_shape": list(sample_batch["actions"].shape),
            "observations_shape": tree_map(
                lambda x: list(x.shape), sample_batch["observations"]
            ),
        }

        return infos


if __name__ == "__main__":
    dataset = ManiSkillDataset(
        simulated_length=10000000,
        test_ratio=0.05,
        task="PickCube-v1",
        pack_root="/home/i53/student/yagmurlu/code/manten/data/maniskill2/packed_demos",
        obs_horizon=2,
        pred_horizon=16,
        obs_mode="rgb",
        state_modality_keys=["goal_pos"],
        rgb_modality_keys=["camera1"],
        control_mode="pd_ee_delta_pose",
        use_mmap=True,
        load_count=35,
        # use_mmap=False,
    )

    print(dataset[0])
