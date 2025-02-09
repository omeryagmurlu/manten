import functools
import logging
from functools import partial
from pathlib import Path
from types import MappingProxyType

import einops
import numpy as np
import torch
import zarr
from optree import tree_map
from torch.utils.data import Dataset
from tqdm import tqdm

from manten.agents.utils.normalization import fit_consecutive
from manten.networks.utils.rotation_transformer import RotationTransformer
from manten.utils.utils_data import modulo_dataset
from manten.utils.utils_file import normalize_filename
from manten_evaluation.maniskill2.lib.utils_maniskill_common import (
    apply_static_transforms,
    transform_episode,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path("~/.cache/manten/dataset_maniskill").expanduser()


class LoaderDict:
    def __init__(self, load, epf):
        self.load = load
        self.epf = epf

    def __getitem__(self, key):
        return self.load(self.epf(key=key))


class ZarrLoaderDict:
    def __init__(self, epf):
        self.store = zarr.DirectoryStore(epf)

    def __getitem__(self, key):
        return zarr.open(self.store, mode="r")[key]


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
        rotation_transform=None,
        control_mode,
        # ...
        obs_horizon,
        pred_horizon,
        # ...
        load_count=None,
        use_mmap=False,
        cache_stats=True,
        ignore_stats_cache=False,
        use_zarr=False,
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
        if rotation_transform is not None:
            self.rotation_transformer = RotationTransformer(
                from_rep="euler_angles", to_rep=rotation_transform, from_convention="XYZ"
            )
        else:
            self.rotation_transformer = None

        path_obs_mode = self.OBS_MODE_FILE_NAMES[obs_mode]

        self.use_zarr = use_zarr
        self.paths = {
            "action_lengths": f"{pack_root}/{task}/{path_obs_mode}/{control_mode}/traj_lengths.npy",
            "episode_format": partial(
                "{pack_root}/{task}/{path_obs_mode}/{control_mode}/episode_{episode_idx}/{key}.npy".format
                if not use_zarr
                else "{pack_root}/{task}/{path_obs_mode}/{control_mode}/episode_{episode_idx}.zarr".format,
                pack_root=pack_root,
                task=task,
                path_obs_mode=path_obs_mode,
                control_mode=control_mode,
            ),
        }

        self.cache_stats = cache_stats
        self.ignore_stats_cache = ignore_stats_cache
        if cache_stats:
            stats_cache_identifier = f"{train}_{test_ratio}_{pack_root}_{task}_{obs_mode}_{obs_modalities}_{state_modality_keys}_{rgb_modality_keys}_{control_mode}_{load_count}"
            self.stats_cache_identifier = normalize_filename(stats_cache_identifier)

        action_lengths = np.load(self.paths["action_lengths"])
        action_lengths = action_lengths[np.argsort(action_lengths[:, 0])]
        if load_count is not None:
            action_lengths = action_lengths[:load_count]

        split = int(len(action_lengths) * (1 - test_ratio))
        if train:
            action_lengths = action_lengths[:split]
        else:
            action_lengths = action_lengths[split:]

        self.action_lengths = action_lengths

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
            lambda obs: obs[max(0, start) : start + self.obs_horizon],
            episode["observations"],
        )
        obs_dict = tree_map(
            lambda x: torch.tensor(x) if x.dtype != 'float64' else torch.tensor(x, dtype=torch.float32), obs_dict
        )
        # start+self.obs_horizon is at least 1
        act_seq = torch.tensor(episode["actions"][max(0, start) : end])

        obs_dict = apply_static_transforms(obs_dict, obs_mode=self.obs_mode)

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

        meta_dict = {
            "3d_mask": torch.Tensor([self.obs_mode == "pointcloud"]).bool(),
            "2d_mask": torch.Tensor(
                [self.obs_mode == "pointcloud" or self.obs_mode == "rgb"]
            ).bool(),
        }

        return {"observations": obs_dict, "actions": act_seq, "meta": meta_dict}

    @functools.cache  # noqa: B019
    def get_episode(self, idx):
        epf = partial(self.paths["episode_format"], episode_idx=idx)
        if not self.use_zarr:
            load = partial(np.load, mmap_mode="r" if self.use_mmap else None)
            loader_dict = LoaderDict(load, epf)
        else:
            loader_dict = ZarrLoaderDict(epf())

        return transform_episode(
            loader_dict,
            obs_mode=self.obs_mode,
            obs_modalities=self.obs_modalities,
            rgb_modality_keys=self.rgb_modality_keys,
            state_modality_keys=self.state_modality_keys,
            rotation_transformer=self.rotation_transformer,
        )

    @functools.cache  # noqa: B019
    def get_dataset_info(self):
        fit_results = self.__fit_stats()

        sample_batch = self[0]

        infos = {
            "actions_stats": fit_results["actions_fit"],
            "obs_horizon": self.obs_horizon,
            "pred_horizon": self.pred_horizon,
            "actions_shape": list(sample_batch["actions"].shape),
            "observations_shape": tree_map(
                lambda x: list(x.shape), sample_batch["observations"]
            ),
            "rotation_dim": (
                self.rotation_transformer.to_dim
                if self.rotation_transformer is not None
                else 3  # maniskill default euler angles
            ),
        }

        if "pcd_fit" in fit_results:
            infos["pcd_stats"] = fit_results["pcd_fit"]

        if "tcp_pose" in self.state_modality_keys:
            infos["tcp_pose_key"] = "tcp_pose"

        # agent wrappers need this, and for now just manually match
        # if self.rotation_transformer is not None:
        #     infos["rotation_transformer"] = self.rotation_transform

        return infos

    def __fit_stats(self):
        if self.cache_stats:
            cache_path = CACHE_DIR / "stats_cache"
            cache_path.mkdir(parents=True, exist_ok=True)

            cache_file = cache_path / f"{self.stats_cache_identifier}.pkl"

            if not self.ignore_stats_cache and cache_file.exists():
                import pickle

                logger.info("Loading stats from cache: %s", cache_file)
                with cache_file.open("rb") as f:
                    return pickle.load(f)  # noqa: S301

        logger.info("Fitting stats...")

        actions_fit = None
        pcd_fit = None
        for idx in tqdm(range(len(self.action_lengths))):
            episode = self.get_episode(idx)

            actions_fit = fit_consecutive(episode["actions"], actions_fit)

            if "pcd_obs" in episode["observations"]:
                pcd = episode["observations"]["pcd_obs"]
                pcd_mask = episode["observations"]["pcd_mask"]
                combined_pcd = np.concatenate([pcd[cam] for cam in pcd], axis=2)
                combined_pcd_mask = np.concatenate(
                    [pcd_mask[cam] for cam in pcd_mask], axis=2
                )
                combined_pcd = einops.rearrange(combined_pcd, "b c h w -> (b h w) c")
                combined_pcd_mask = einops.rearrange(
                    combined_pcd_mask, "b c h w -> (b h w) c"
                )
                masked_pcd = np.ma.masked_array(
                    combined_pcd, ~np.broadcast_to(combined_pcd_mask, combined_pcd.shape)
                )
                pcd_fit = fit_consecutive(masked_pcd, pcd_fit)

        out_dict = {}
        out_dict["actions_fit"] = tree_map(lambda x: x.tolist(), actions_fit[0])

        if pcd_fit is not None:
            out_dict["pcd_fit"] = tree_map(lambda x: x.tolist(), pcd_fit[0])

        if self.cache_stats:
            from accelerate import PartialState

            state = PartialState()

            if state.is_main_process:
                import pickle

                logger.info("Saving stats to cache: %s", cache_file)
                with cache_file.open("wb") as f:
                    pickle.dump(out_dict, f)

        return out_dict


if __name__ == "__main__":
    dataset = ManiSkillDataset(
        simulated_length=10000000,
        test_ratio=0.05,
        task="PickCube-v1",
        pack_root="/home/i53/student/yagmurlu/code/manten/data/maniskill2/packed_demos",
        obs_horizon=2,
        pred_horizon=16,
        obs_mode="pointcloud",
        # state_modality_keys=["goal_pos"],
        # rgb_modality_keys=["camera1", "gripper1"],
        rgb_modality_keys=["camera1"],
        control_mode="pd_ee_delta_pose",
        use_mmap=True,
        load_count=35,
        # use_mmap=False,
        rotation_transform="rotation_6d",
    )

    print(dataset[0])

    print(dataset.get_dataset_info())
