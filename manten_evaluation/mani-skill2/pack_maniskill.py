# ruff: noqa: G004

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import h5py
import numpy as np
from optree import tree_flatten_with_path
from tqdm import tqdm

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


def pack_episode(
    episode_idx,
    *,
    h5filename,
    outdir,
):
    logging.info(f"Packing episode {episode_idx}")

    episode = load_episode_hdf5(h5filename, episode_idx, single=True)[f"traj_{episode_idx}"]
    action_length = episode["actions"].shape[0]

    paths, leaves, treespec = tree_flatten_with_path(episode)
    cpaths = [".".join(elems) for elems in paths]
    dc = dict(zip(cpaths, leaves, strict=True))

    np.savez(outdir / f"episode_{episode_idx}.npz", **dc)

    return [episode_idx, action_length]


def pack_task(filename, outname):
    logging.info(f"Packing task from {filename} to {outname}")
    json_data = load_json(str(filename).replace(".h5", ".json"))

    num_episodes = len(json_data["episodes"])

    with ProcessPoolExecutor() as executor:
        action_lengths = list(
            tqdm(
                executor.map(
                    partial(
                        pack_episode,
                        h5filename=filename,
                        outdir=outname,
                    ),
                    range(num_episodes),
                ),
                total=num_episodes,
                desc="Packing episodes",
            )
        )
    # action_lengths = []
    # for episode_idx in range(num_episodes):
    #     action_lengths.append(pack_episode(episode_idx, h5filename=filename, outdir=outname))

    np.save(outname / "traj_lengths.npy", np.array(action_lengths))


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        help="The task to perform",
        default=None,
    )
    parser.add_argument(
        "--path",
        type=str,
        required=False,
        help="The path to the file or directory",
        default="/home/i53/student/yagmurlu/code/manten/data/maniskill2",
    )

    args = parser.parse_args()

    logging.info(f"Starting packing process with task: {args.task} and path: {args.path}")

    demos_path = Path(args.path) / "demos"
    out_path = Path(args.path) / "packed_demos"
    if not out_path.exists():
        out_path.mkdir()

    if args.task is not None:
        tasks = [args.task]
    else:
        tasks = [task.name for task in demos_path.iterdir() if task.is_dir()]

    for task in tasks:
        logging.info(f"Processing task: {task}")
        task_path = demos_path / task
        out_task_path = out_path / task
        if not out_task_path.exists():
            out_task_path.mkdir()

        filename = (
            task_path / "motionplanning" / "trajectory.pointcloud.pd_ee_delta_pose.cpu.h5"
        )

        pack_task(filename, out_task_path)

    logging.info("Packing process completed")


if __name__ == "__main__":
    main()
