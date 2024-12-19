# ruff: noqa: G004

import argparse
import json
import logging
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import h5py
import numpy as np
from optree import tree_flatten_with_path, tree_map
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_json(filepath):
    with Path(filepath).open("r") as f:
        return json.load(f)


def inflate_h5_data(data):
    out = {}
    for k in data:
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = inflate_h5_data(data[k])
    return out


def load_episode(data, episode_idx):
    return inflate_h5_data(data[f"traj_{episode_idx}"])


def pack_episode(episode_idx, *, filename, outdir, trajectory_length=20, pad_zeros=True):
    logging.info(f"Packing episode {episode_idx}")
    data = h5py.File(filename, "r")

    episode = load_episode(data, episode_idx)
    episode["trajectories"] = []

    for i in range(len(episode["actions"])):
        chunk = episode["actions"][i : i + trajectory_length]
        if pad_zeros:
            if len(chunk) < trajectory_length:
                # pad with last zeroes # assuming delta
                chunk = np.concatenate(
                    [chunk, np.zeros((trajectory_length - len(chunk), chunk.shape[1]))]
                )
        else:
            raise NotImplementedError("No padding")
        episode["trajectories"].append(chunk)

    del episode["actions"]
    episode["trajectories"] = np.array(episode["trajectories"])

    len_traj = episode["trajectories"].shape[0]
    episode = tree_map(
        lambda x: x[:len_traj], episode
    )  # obs include last state without action

    paths, leaves, treespec = tree_flatten_with_path(episode)
    dc = dict(zip([".".join(elems) for elems in paths], leaves, strict=True))
    np.savez(outdir / f"episode_{episode_idx}.npz", **dc)

    return len_traj, treespec


def pack_task(filename, outname):
    logging.info(f"Packing task from {filename} to {outname}")
    json_data = load_json(str(filename).replace(".h5", ".json"))

    # this is easily parallelizable
    num_episodes = len(json_data["episodes"])

    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(
                    partial(
                        pack_episode,
                        filename=filename,
                        outdir=outname,
                        trajectory_length=20,
                        pad_zeros=True,
                    ),
                    range(num_episodes),
                ),
                total=num_episodes,
                desc="Packing episodes",
            )
        )

    traj_lengths, treespecs = zip(*results, strict=True)
    traj_lengths = list(traj_lengths)
    treespec = treespecs[0]  # assuming all episodes have the same treespec

    np.save(outname / "traj_lengths.npy", np.array(traj_lengths))
    with Path(outname / "treespec.pkl").open("wb") as f:
        pickle.dump(treespec, f)


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
