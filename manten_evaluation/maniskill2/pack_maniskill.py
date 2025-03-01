# ruff: noqa: G004

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from shutil import rmtree

import h5py
import numpy as np
import zarr
from tqdm import tqdm

from manten_evaluation.maniskill2.lib.utils_maniskill_common import (
    process_observation_from_raw,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_json(filepath):
    with Path(filepath).open("r") as f:
        return json.load(f)


def load_content_from_h5_file(file):
    if isinstance(file, (h5py.File, h5py.Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, h5py.Dataset):
        return file[()]
    else:
        raise NotImplementedError(f"Unsupported h5 file type: {type(file)}")


def load_hdf5(
    path,
):
    # print("Loading HDF5 file", path)
    file = h5py.File(path, "r")
    ret = load_content_from_h5_file(file)
    file.close()
    # print("Loaded")
    return ret


def load_episode_hdf5(path, num_episode=None, single=False):
    # print("Loading HDF5 file", path)
    file = h5py.File(path, "r")
    keys = list(file.keys())
    if num_episode is not None:
        assert num_episode <= len(
            keys
        ), f"num_episode: {num_episode} > len(keys): {len(keys)}"
        keys = sorted(keys, key=lambda x: int(x.split("_")[-1]))
        if single:
            keys = keys[num_episode : num_episode + 1]
        else:
            keys = keys[:num_episode]
    ret = {key: load_content_from_h5_file(file[key]) for key in keys}
    file.close()
    # print("Loaded")
    return ret


ONE_MiB = 1024 * 1024


def pack_episode(
    episode_idx,
    *,
    h5filename,
    outdir,
    obs_mode,
    use_zarr,
    rgb_modality_keys: list[str],
    zarr_obs_chunk=2,
    zarr_actions_chunk=20,
):
    # logging.info(f"Packing episode {episode_idx}")

    episode = load_episode_hdf5(h5filename, episode_idx, single=True)[f"traj_{episode_idx}"]
    action_length = episode["actions"].shape[0]

    # paths, leaves, treespec = tree_flatten_with_path(episode)
    # cpaths = [".".join(elems) for elems in paths]
    # dc = dict(zip(cpaths, leaves, strict=True))

    pco = process_observation_from_raw(
        episode["obs"],
        obs_mode=obs_mode,
        rgb_modality_keys=rgb_modality_keys,
        slice_rgb_modality_to_ncam=True,
    )
    dc = {
        **pco,
        "actions": episode["actions"],
    }

    if not use_zarr:
        ep_dir = outdir / f"episode_{episode_idx}"
        if ep_dir.exists():
            rmtree(ep_dir)
        ep_dir.mkdir(parents=True, exist_ok=True)
        for key, value in dc.items():
            np.save(ep_dir / f"{key}.npy", value)
    else:
        ep_dir = outdir / f"episode_{episode_idx}.zarr"
        if ep_dir.exists():
            rmtree(ep_dir)
        zarr_store = zarr.DirectoryStore(str(ep_dir))
        zarr_group = zarr.group(zarr_store, overwrite=True)
        for key, value in dc.items():
            zarr_chunk_size = zarr_actions_chunk if key == "actions" else zarr_obs_chunk  # noqa: F841

            key_len = len(value)
            zarr_shard_size = min(
                key_len, np.ceil(key_len / (value.nbytes / ONE_MiB)).astype(int)
            )

            # # zarrV2 does not support shards, so just aim for min 1MB chunks
            # zarr_chunk_size = max(zarr_shard_size, zarr_chunk_size)

            zarr_group.create_dataset(
                name=key,
                data=value,
                chunks=(zarr_shard_size, *value.shape[1:]),
                # chunks=(zarr_chunk_size, *value.shape[1:]),
                # shards=(zarr_shard_size, *value.shape[1:]),
            )

    return [episode_idx, action_length]


def pack_episodes(chunk, **kwargs):
    rank, episode_idxs = chunk
    action_lengths = []
    for episode_idx in (
        progbar := tqdm(episode_idxs, total=len(episode_idxs), position=rank)
    ):
        progbar.set_description(f"Packing episode {episode_idx}")
        action_lengths.append(pack_episode(episode_idx, **kwargs))

    return np.array(action_lengths)


def pack_task(filename, outdir, n_proc, load_count=None, parallel=True, **kwargs):
    logging.info(f"Packing task from {filename} to {outdir}")
    json_data = load_json(str(filename).replace(".h5", ".json"))

    num_episodes = len(json_data["episodes"])
    if load_count is not None:
        num_episodes = min(num_episodes, load_count)

    chunks = np.array_split(np.arange(num_episodes), n_proc)

    if parallel:
        logging.info(f"Packing {num_episodes} episodes in {n_proc} processes")
        with ProcessPoolExecutor(max_workers=n_proc) as executor:
            action_lengths = executor.map(
                partial(pack_episodes, h5filename=filename, outdir=outdir, **kwargs),
                enumerate(chunks),
            )
    else:
        logging.info(f"Packing {num_episodes} episodes in a single process")
        action_lengths = [
            pack_episodes(chunk, h5filename=filename, outdir=outdir, **kwargs)
            for chunk in enumerate(chunks)
        ]
    logging.info(f"Packed {num_episodes} episodes")

    action_lengths = np.concatenate(list(action_lengths), axis=0)
    np.save(outdir / "traj_lengths.npy", action_lengths)


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        help="The task to perform, if not specified all tasks in the path will be processed",
        default=None,
    )
    parser.add_argument(
        "--path",
        type=str,
        required=False,
        help="The path to the file or directory",
        default=os.getcwd() + "/data/maniskill2",
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        required=False,
        help="The number of processes to use",
        default=10,
    )
    parser.add_argument(
        "--obs_mode",
        type=str,
        required=False,
        help="The observation mode, one of [state, pointcloud, rgb+depth+segmentation (or any combination of rgb, depth, segmentation)]",
        default="pointcloud",
    )
    parser.add_argument(
        "--control_mode",
        type=str,
        required=False,
        help="see: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html",
        default="pd_ee_delta_pose",
    )
    parser.add_argument(
        "--load_count",
        type=int,
        required=False,
        help="The number of episodes to load",
        default=None,
    )
    parser.add_argument(
        "--debug_run",
        action="store_true",
        help="If set, the output will be saved to a different directory",
        default=False,
    )
    parser.add_argument(
        "--demo_type",
        type=str,
        required=False,
        help="The type of the demo",
        default="motionplanning",
    )
    parser.add_argument(
        "--sim_backend",
        type=str,
        required=False,
        help="The simulation backend",
        default="physx_cpu",
    )
    parser.add_argument(
        "--use_zarr",
        action="store_true",
        help="If set, the output will be zarr files",
        default=False,
    )
    parser.add_argument(
        "--rgb_modality_keys",
        type=str,
        required=False,
        help="The keys of the rgb modalities",
        default=None,
    )

    args = parser.parse_args()

    if args.rgb_modality_keys is not None:
        args.rgb_modality_keys = args.rgb_modality_keys.split(",")
    else:
        args.rgb_modality_keys = [
            "camera1",
            "gripper1",
        ]  # this will be sliced afterwards to fit the number of cameras

    logging.info(f"Starting packing process with task: {args.task} and path: {args.path}")

    demos_path = Path(args.path) / "demos"
    out_path = (
        Path(args.path) / "packed_demos"
        if not args.debug_run
        else Path(args.path) / "packed_demos_debug_run"
    )
    if not out_path.exists():
        out_path.mkdir()

    if args.task is not None:
        tasks = [args.task]
    else:
        tasks = [task.name for task in demos_path.iterdir() if task.is_dir()]

    for task in tasks:
        logging.info(f"Processing task: {task}")
        task_path = demos_path / task
        out_task_path = out_path / task / args.obs_mode / args.control_mode
        if not out_task_path.exists():
            out_task_path.mkdir(parents=True)

        filename = (
            task_path
            / args.demo_type
            / f"trajectory.{args.obs_mode}.{args.control_mode}.{args.sim_backend}.h5"
        )

        pack_task(
            filename,
            out_task_path,
            n_proc=args.n_proc,
            load_count=args.load_count,
            obs_mode=args.obs_mode,
            parallel=not args.debug_run,
            use_zarr=args.use_zarr,
            rgb_modality_keys=args.rgb_modality_keys,
        )

    logging.info("Packing process completed")


if __name__ == "__main__":
    main()
