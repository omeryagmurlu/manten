import einops
import numpy as np
import optree
import torch


def get_to_remove_indices(pcd):
    FAR_THRESHOLD = 0.5  # 0 for camera far, 1 camera near
    FOCUS_THRESHOLD = 1  # meters

    abs_fn = np.abs if isinstance(pcd, np.ndarray) else torch.abs

    far_indices = pcd[..., 3] < FAR_THRESHOLD
    unfocused_indices = (
        (abs_fn(pcd[..., 0]) > FOCUS_THRESHOLD)
        | (abs_fn(pcd[..., 1]) > FOCUS_THRESHOLD)
        | (abs_fn(pcd[..., 2]) > FOCUS_THRESHOLD)
    )
    to_remove_indices = far_indices | unfocused_indices

    return to_remove_indices


def process_observation_from_raw(obs, obs_mode, image_size: int = 128, cam=None):
    if obs_mode == "state":
        return {"state_obs": obs}

    paths, elems, _ = optree.tree_flatten_with_path(
        {**obs["sensor_param"], **obs["agent"], **obs["extra"]}
    )
    paths = [".".join(path) for path in paths]
    extra_elems = dict(zip(paths, elems, strict=False))

    if obs_mode == "pointcloud":
        pcd = obs["pointcloud"]["xyzw"]

        if cam is None:
            _, n_points, _ = pcd.shape
            assert image_size is not None
            cam = n_points // (image_size**2)
        if image_size is None:
            _, n_points, _ = pcd.shape
            assert cam is not None
            image_size = int(np.sqrt(n_points // cam))

        to_remove_indices = get_to_remove_indices(pcd)

        return {
            **extra_elems,
            "rgb_obs": einops.rearrange(
                obs["pointcloud"]["rgb"],
                "ix (cam h w) c -> ix cam h w c",
                cam=cam,
                h=image_size,
                w=image_size,
                c=3,
            ),
            "pcd_obs": pcd[..., :3],
            "pcd_mask": ~to_remove_indices,
        }
    elif obs_mode == "rgb":
        rgbs = optree.tree_flatten(obs["sensor_data"])[0]
        return {
            **extra_elems,
            "rgb_obs": einops.rearrange(rgbs, "cam ix h w c -> ix cam h w c"),
        }

    raise ValueError(f"obs_mode {obs_mode} not recognized")
