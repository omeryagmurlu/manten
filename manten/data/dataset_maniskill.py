import functools

import einops
import numpy as np
import torch
from optree import tree_map
from torch.utils.data import Dataset


class ManiSkillDataset(Dataset):
    def __init__(
        self,
        *,
        pack_root,
        task="PegInsertionSide-v1",
        obs_horizon=2,
        pred_horizon=16,
        obs_modalities=("pcd_obs", "rgb_obs", "state_obs"),
    ):
        self.obs_modalities = obs_modalities

        self.paths = {
            "action_lengths": f"{pack_root}/{task}/traj_lengths.npy",
            "episode_format": f"{pack_root}/{task}/episode_%d.npz",
        }

        action_lengths = np.load(self.paths["action_lengths"])
        action_lengths = action_lengths[np.argsort(action_lengths[:, 0])]

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
                (episode_idx, start, start + pred_horizon)
                for start in range(-pad_before, action_length - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        episode = self.get_episode(traj_idx)
        L, act_dim = episode["actions"].shape

        obs_seq = tree_map(
            lambda obs: obs[max(0, start) : start + self.obs_horizon], episode["observations"]
        )
        # start+self.obs_horizon is at least 1
        act_seq = episode["actions"][max(0, start) : end]
        if start < 0:  # pad before the trajectory
            obs_seq = tree_map(
                lambda obs: torch.cat(
                    [einops.repeat(obs[0], "... -> k ...", k=-start), obs], dim=0
                ),
                obs_seq,
            )
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            gripper_action = act_seq[-1, -1]
            if self.pad_action_arm is None:
                self.pad_action_arm = torch.zeros((act_dim - 1,))
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        for obs in obs_seq.values():
            assert obs.shape[0] == self.obs_horizon
        assert act_seq.shape[0] == self.pred_horizon

        # |o|o|                             observations: 2
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        #
        # | |a|a|a|a|a|a|a|a|               actions executed: 8
        return {
            "observations": obs_seq,
            "actions": act_seq,
        }

    @functools.lru_cache(maxsize=100)  # noqa: B019
    def get_episode(self, idx):
        # this cache is bad since it is inflated by ddp_gpus * num_workers in memory
        # shared memory would solve it but im lazy, so lru_cache it is
        npz = np.load(self.paths["episode_format"] % idx)
        dc = {k: npz[k] for k in npz}
        dc = tree_map(torch.from_numpy, dc)

        obs_dict = {}
        for modality in self.obs_modalities:
            if modality == "pcd_obs":
                obs_dict[modality] = dc["obs.pointcloud.xyzw"].view(-1, 2, 128, 128, 4)
            elif modality == "rgb_obs":
                obs_dict[modality] = dc["obs.pointcloud.rgb"].view(-1, 2, 128, 128, 3)
            elif modality == "state_obs":
                obs_dict[modality] = torch.cat(
                    [dc["obs.agent.qpos"], dc["obs.agent.qvel"], dc["obs.extra.tcp_pose"]],
                    dim=1,
                )

        return {
            "actions": dc["actions"],
            "observations": obs_dict,
        }
