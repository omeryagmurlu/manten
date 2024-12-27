# ruff: noqa: C901, PLR0912, G004, PLR0915, PLR2004, S301, TRY300, PTH123, NPY002

import itertools
import math
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from pickle import UnpicklingError
from time import time

import blosc
import einops
import numpy as np
import torch
import torchvision.transforms.functional as transforms_f
from scipy.interpolate import CubicSpline, interp1d
from torch.utils.data import Dataset
from torchvision import transforms

from manten.agents.three_dda.utils.dda_utils_with_calvin import (
    convert_rotation,
    to_relative_action,
)
from manten.utils.logging import get_logger
from manten.utils.progbar import progbar

logger = get_logger(__name__)


def normalise_quat(x: torch.Tensor):
    return x / torch.clamp(x.square().sum(dim=-1).sqrt().unsqueeze(-1), min=1e-10)


def loader(file):
    if str(file).endswith(".npy"):
        try:
            content = np.load(file, allow_pickle=True)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".dat"):
        try:
            with open(file, "rb") as f:
                content = pickle.loads(blosc.decompress(f.read()))
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".pkl"):
        try:
            with open(file, "rb") as f:
                content = pickle.load(f)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    return None


class Resize:
    """Resize and pad/crop the image and aligned point cloud."""

    def __init__(self, scales):
        self.scales = scales

    def __call__(self, **kwargs):
        """Accept tensors as T, N, C, H, W."""
        keys = list(kwargs.keys())

        if len(keys) == 0:
            raise RuntimeError("No args")

        # Sample resize scale from continuous range
        sc = np.random.uniform(*self.scales)

        t, n, c, raw_h, raw_w = kwargs[keys[0]].shape
        kwargs = {n: arg.flatten(0, 1) for n, arg in kwargs.items()}
        resized_size = [int(raw_h * sc), int(raw_w * sc)]

        # Resize
        kwargs = {
            n: transforms_f.resize(arg, resized_size, transforms.InterpolationMode.NEAREST)
            for n, arg in kwargs.items()
        }

        # If resized image is smaller than original, pad it with a reflection
        if raw_h > resized_size[0] or raw_w > resized_size[1]:
            right_pad, bottom_pad = (
                max(raw_w - resized_size[1], 0),
                max(raw_h - resized_size[0], 0),
            )
            kwargs = {
                n: transforms_f.pad(
                    arg, padding=[0, 0, right_pad, bottom_pad], padding_mode="reflect"
                )
                for n, arg in kwargs.items()
            }

        # If resized image is larger than original, crop it
        i, j, h, w = transforms.RandomCrop.get_params(
            kwargs[keys[0]], output_size=(raw_h, raw_w)
        )
        kwargs = {n: transforms_f.crop(arg, i, j, h, w) for n, arg in kwargs.items()}

        kwargs = {
            n: einops.rearrange(arg, "(t n) c h w -> t n c h w", t=t)
            for n, arg in kwargs.items()
        }

        return kwargs


class TrajectoryInterpolator:
    """Interpolate a trajectory to have fixed length."""

    def __init__(self, use=False, interpolation_length=50):
        self._use = use
        self._interpolation_length = interpolation_length

    def __call__(self, trajectory):
        if not self._use:
            return trajectory
        trajectory = trajectory.numpy()
        # Calculate the current number of steps
        old_num_steps = len(trajectory)

        # Create a 1D array for the old and new steps
        old_steps = np.linspace(0, 1, old_num_steps)
        new_steps = np.linspace(0, 1, self._interpolation_length)

        # Interpolate each dimension separately
        resampled = np.empty((self._interpolation_length, trajectory.shape[1]))
        for i in range(trajectory.shape[1]):
            if i == (trajectory.shape[1] - 1):  # gripper opening
                interpolator = interp1d(old_steps, trajectory[:, i])
            else:
                interpolator = CubicSpline(old_steps, trajectory[:, i])

            resampled[:, i] = interpolator(new_steps)

        resampled = torch.tensor(resampled)
        if trajectory.shape[1] == 8:
            resampled[:, 3:7] = normalise_quat(resampled[:, 3:7])
        return resampled


def load_instructions(instructions, split):
    with Path(f"{instructions}/{split}.pkl").open("rb") as file:
        instructions = pickle.load(file)["embeddings"]
    return instructions


def traj_collate_fn(batch):
    keys = [
        "trajectory",
        "trajectory_mask",
        "rgbs",
        "pcds",
        "curr_gripper",
        "curr_gripper_history",
        "action",
        "instr",
        "has3d",
    ]
    ret_dict = {
        key: torch.cat(
            # [item[key].float() if key != "trajectory_mask" else item[key] for item in batch]
            [item[key] for item in batch]
        )
        for key in keys
    }

    # ret_dict["task"] = []
    # for item in batch:
    #     ret_dict["task"] += item["task"]
    return ret_dict


class CalvinDataset(Dataset):
    """Calvin dataset."""

    def __init__(
        self,
        # required
        root,
        instructions=None,
        # dataset specification
        taskvar=None,
        max_episode_length=5,
        cache_size=0,
        max_episodes_per_task=100,
        num_iters=None,
        cameras=("wrist", "left_shoulder", "right_shoulder"),
        # for augmentations
        training=True,
        image_rescale=(1.0, 1.0),
        # for trajectories
        return_low_lvl_trajectory=True,
        dense_interpolation=False,
        interpolation_length=100,
        relative_action=True,
    ):
        if taskvar is None:
            taskvar = [("A", 0), ("B", 0), ("C", 0), ("D", 0)]

        self._cache = {}
        self._cache_size = cache_size
        self._cameras = cameras
        self._max_episode_length = max_episode_length
        self._num_iters = num_iters
        self._training = training
        self._taskvar = taskvar
        self._return_low_lvl_trajectory = return_low_lvl_trajectory
        if isinstance(root, (Path, str)):
            root = [Path(root)]
        self._root = [Path(r).expanduser() for r in root]
        self._relative_action = relative_action

        # For trajectory optimization, initialize interpolation tools
        if return_low_lvl_trajectory:
            assert dense_interpolation
            self._interpolate_traj = TrajectoryInterpolator(
                use=dense_interpolation, interpolation_length=interpolation_length
            )

        if isinstance(instructions, (Path, str)):
            instructions = load_instructions(
                instructions, split="training" if training else "validation"
            )
        # Keep variations and useful instructions
        self._instructions = instructions
        self._num_vars = Counter()  # variations of the same task
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if data_dir.is_dir():
                self._num_vars[task] += 1

        # If training, initialize augmentation classes
        if self._training:
            self._resize = Resize(scales=image_rescale)

        # File-names of episodes per-task and variation
        episodes_by_task = defaultdict(list)
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if not data_dir.is_dir():
                print(f"Can't find dataset folder {data_dir}")
                continue
            npy_episodes = [(task, var, ep) for ep in data_dir.glob("*.npy")]
            dat_episodes = [(task, var, ep) for ep in data_dir.glob("*.dat")]
            pkl_episodes = [(task, var, ep) for ep in data_dir.glob("*.pkl")]
            episodes = npy_episodes + dat_episodes + pkl_episodes
            # Split episodes equally into task variations
            if max_episodes_per_task > -1:
                episodes = episodes[: max_episodes_per_task // self._num_vars[task] + 1]
            if len(episodes) == 0:
                print(f"Can't find episodes at folder {data_dir}")
                continue
            episodes_by_task[task] += episodes

        # Collect and trim all episodes in the dataset
        self._episodes = []
        self._num_episodes = 0
        for eps in episodes_by_task.values():
            if len(eps) > max_episodes_per_task and max_episodes_per_task > -1:
                eps = random.sample(eps, max_episodes_per_task)  # noqa: PLW2901
            self._episodes += eps
            self._num_episodes += len(eps)

        logger.info(f"created dataset from {root} with {self._num_episodes}")

    def __getitem__(self, episode_id):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
        ]
        """
        episode_id %= self._num_episodes
        task, variation, file = self._episodes[episode_id]

        # Load episode
        episode = self.read_from_cache(file)
        if episode is None:
            return None

        # Dynamic chunking so as not to overload GPU memory
        chunk = random.randint(0, math.ceil(len(episode[0]) / self._max_episode_length) - 1)

        # Get frame ids for this chunk
        frame_ids = episode[0][
            chunk * self._max_episode_length : (chunk + 1) * self._max_episode_length
        ]

        # Get the image tensors for the frame ids we got
        states = torch.stack(
            [
                episode[1][i]
                if isinstance(episode[1][i], torch.Tensor)
                else torch.from_numpy(episode[1][i])
                for i in frame_ids
            ]
        )

        # Camera ids
        if episode[3]:
            cameras = list(episode[3][0].keys())
            assert all(c in cameras for c in self._cameras)
            index = torch.tensor([cameras.index(c) for c in self._cameras])
            # Re-map states based on camera ids
            states = states[:, index]

        # Split RGB and XYZ
        rgbs = states[:, :, 0, :, 20:180, 20:180]
        pcds = states[:, :, 1, :, 20:180, 20:180]
        rgbs = self._unnormalize_rgb(rgbs)

        # Get action tensors for respective frame ids
        action = torch.cat([episode[2][i] for i in frame_ids])

        # Sample one instruction feature
        if self._instructions is not None:
            instr_ind = episode[6][0]
            instr = torch.as_tensor(self._instructions[instr_ind])
            instr = instr.repeat(len(rgbs), 1, 1)
        else:
            instr = torch.zeros((rgbs.shape[0], 53, 512))

        # Get gripper tensors for respective frame ids
        gripper = torch.cat([episode[4][i] for i in frame_ids])

        # gripper history
        if len(episode) > 7:
            gripper_history = torch.cat([episode[7][i] for i in frame_ids], dim=0)
        else:
            gripper_history = torch.stack(
                [
                    torch.cat([episode[4][max(0, i - 2)] for i in frame_ids]),
                    torch.cat([episode[4][max(0, i - 1)] for i in frame_ids]),
                    gripper,
                ],
                dim=1,
            )

        # Low-level trajectory
        traj, traj_lens = None, 0
        if self._return_low_lvl_trajectory:
            if len(episode) > 5:
                traj_items = [self._interpolate_traj(episode[5][i]) for i in frame_ids]
            else:
                traj_items = [
                    self._interpolate_traj(torch.cat([episode[4][i], episode[2][i]], dim=0))
                    for i in frame_ids
                ]
            max_l = max(len(item) for item in traj_items)
            traj = torch.zeros(len(traj_items), max_l, traj_items[0].shape[-1])
            traj_lens = torch.as_tensor([len(item) for item in traj_items])
            for i, item in enumerate(traj_items):
                traj[i, : len(item)] = item
            traj_mask = torch.zeros(traj.shape[:-1])
            for i, len_ in enumerate(traj_lens.long()):
                traj_mask[i, len_:] = 1

        # Augmentations
        if self._training:
            if traj is not None:
                for t, tlen in enumerate(traj_lens):
                    traj[t, tlen:] = 0
            modals = self._resize(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]

        # Compute relative action
        if self._relative_action and traj is not None:
            rel_traj = torch.zeros_like(traj)
            for i in range(traj.shape[0]):
                for j in range(traj.shape[1]):
                    rel_traj[i, j] = torch.as_tensor(
                        to_relative_action(traj[i, j].numpy(), traj[i, 0].numpy(), clip=False)
                    )
            traj = rel_traj

        # Convert Euler angles to Quarternion
        action = torch.cat(
            [
                action[..., :3],
                torch.as_tensor(convert_rotation(action[..., 3:6])),
                action[..., 6:],
            ],
            dim=-1,
        )
        gripper = torch.cat(
            [
                gripper[..., :3],
                torch.as_tensor(convert_rotation(gripper[..., 3:6])),
                gripper[..., 6:],
            ],
            dim=-1,
        )
        gripper_history = torch.cat(
            [
                gripper_history[..., :3],
                torch.as_tensor(convert_rotation(gripper_history[..., 3:6])),
                gripper_history[..., 6:],
            ],
            dim=-1,
        )
        if traj is not None:
            traj = torch.cat(
                [
                    traj[..., :3],
                    torch.as_tensor(convert_rotation(traj[..., 3:6])),
                    traj[..., 6:],
                ],
                dim=-1,
            )

        # assert all(anno == episode[6][0] for anno in episode[6])

        ret_dict = {
            # "task": [task for _ in frame_ids],
            "rgbs": rgbs,  # e.g. tensor (n_frames, n_cam, 3+1, H, W)
            "pcds": pcds,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            "action": action,  # e.g. tensor (n_frames, 8), target pose
            "instr": instr,  # a (n_frames, 53, 512) tensor
            "curr_gripper": gripper,
            "curr_gripper_history": gripper_history,
            # "annotation_id": [episode[6][0] for _ in frame_ids],
            "has3d": torch.ones(len(rgbs), dtype=torch.bool),
        }
        if self._return_low_lvl_trajectory:
            ret_dict.update(
                {
                    "trajectory": traj,  # e.g. tensor (n_frames, T, 8)
                    "trajectory_mask": traj_mask.bool(),  # tensor (n_frames, T)
                }
            )
        return ret_dict

    def read_from_cache(self, args):
        if self._cache_size == 0:
            return loader(args)

        if args in self._cache:
            return self._cache[args]

        value = loader(args)

        if len(self._cache) == self._cache_size:
            key = list(self._cache.keys())[int(time()) % self._cache_size]
            del self._cache[key]

        if len(self._cache) < self._cache_size:
            self._cache[args] = value

        return value

    @staticmethod
    def _unnormalize_rgb(rgb):
        # (from [-1, 1] to [0, 1]) to feed RGB to pre-trained backbone
        return rgb / 2 + 0.5

    def compute_statistics(self):
        ds_len = len(self)
        logger.info("computing dataset stats (len:%d)", ds_len)
        stat = torch.zeros(2, ds_len, self[0]["trajectory"].shape[-1])
        for i in progbar(range(ds_len), desc="computing dataset stats", leave=False):
            tj = self[i]["trajectory"]
            stat[0, i] = tj.amin(dim=[0, 1])
            stat[1, i] = tj.amax(dim=[0, 1])
        stats = torch.zeros(2, self[0]["trajectory"].shape[-1])
        stats[0] = stat[0].amin(dim=0)
        stats[1] = stat[1].amax(dim=0)
        return stats

    def __len__(self):
        if self._num_iters is not None:
            return self._num_iters
        return self._num_episodes
