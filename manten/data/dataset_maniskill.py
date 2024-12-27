import functools
from functools import partial
from types import MappingProxyType

import einops
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from manten.utils.utils_data import modulo_dataset


@modulo_dataset
class ManiSkillDataset(Dataset):
    FULL_OBS_MODALITIES = MappingProxyType(
        {
            "pointcloud": ("rgb_obs", "pcd_obs", "pcd_mask", "state_obs"),
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
        control_mode,
        # ...
        obs_horizon,
        pred_horizon,
        # ...
        load_count=None,
        use_mmap=False,
    ):
        self.obs_modalities = (
            obs_modalities
            if obs_modalities is not None
            else self.FULL_OBS_MODALITIES[obs_mode]
        )
        self.obs_mode = obs_mode

        self.paths = {
            "action_lengths": f"{pack_root}/{task}/{obs_mode}/{control_mode}/traj_lengths.npy",
            "episode_format": partial(
                "{pack_root}/{task}/{obs_mode}/{control_mode}/episode_{episode_idx}_{key}.npy".format,
                pack_root=pack_root,
                task=task,
                obs_mode=obs_mode,
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

        obs_dict = {
            key: torch.tensor(obs[max(0, start) : start + self.obs_horizon])
            for key, obs in episode["observations"].items()
        }
        # start+self.obs_horizon is at least 1
        act_seq = torch.tensor(episode["actions"][max(0, start) : end])
        if start < 0:  # pad before the trajectory
            obs_dict = {
                key: torch.cat([einops.repeat(obs[0], "... -> k ...", k=-start), obs], dim=0)
                for key, obs in obs_dict.items()
            }
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            gripper_action = act_seq[-1, -1]
            if self.pad_action_arm is None:
                self.pad_action_arm = torch.zeros((act_dim - 1,))
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        for obs in obs_dict.values():
            assert obs.shape[0] == self.obs_horizon
        assert act_seq.shape[0] == self.pred_horizon

        # |o|o|                             observations: 2
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        #
        # | |a|a|a|a|a|a|a|a|               actions executed: 8
        return {
            "observations": obs_dict,
            "actions": act_seq,
        }

    @functools.cache  # noqa: B019
    def get_episode(self, idx):
        epf = partial(self.paths["episode_format"], episode_idx=idx)
        load = partial(np.load, mmap_mode="r" if self.use_mmap else None)

        obs_dict = {}
        for key in self.obs_modalities:
            obs_dict[key] = load(epf(key=key))

        if "pcd_obs" in self.obs_modalities:
            obs_dict["pcd_mask"] = load(epf(key="pcd_mask"))

        return {
            "actions": load(epf(key="actions")),
            "observations": obs_dict,
        }

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
            "act_dim": sample_batch["actions"].shape[-1],
            "obs_horizon": self.obs_horizon,
            "pred_horizon": self.pred_horizon,
        }

        if self.obs_mode == "state":
            infos["obs_shape"] = list(sample_batch["observations"]["state_obs"].shape)

        return infos
