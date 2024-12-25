import einops
import numpy as np
import torch


def get_to_remove_indices(pcd):
    FAR_THRESHOLD = 0.5
    FOCUS_THRESHOLD = 1

    abs_fn = np.abs if isinstance(pcd, np.ndarray) else torch.abs

    far_indices = pcd[..., 3] < FAR_THRESHOLD
    unfocused_indices = (
        (abs_fn(pcd[..., 0]) > FOCUS_THRESHOLD)
        | (abs_fn(pcd[..., 1]) > FOCUS_THRESHOLD)
        | (abs_fn(pcd[..., 2]) > FOCUS_THRESHOLD)
    )
    to_remove_indices = far_indices | unfocused_indices

    return to_remove_indices


def process_observation_from_raw(obs, obs_mode, image_size: int = 128, keep_keys=None):
    if obs_mode == "state":
        retval = {"state_obs": obs}
    elif obs_mode == "pointcloud":
        pcd = obs["pointcloud"]["xyzw"]

        to_remove_indices = get_to_remove_indices(pcd)
        retval = {
            "rgb_obs": einops.rearrange(
                obs["pointcloud"]["rgb"],
                "ix (cam h w) c -> ix cam h w c",
                cam=2,
                h=image_size,
                w=image_size,
                c=3,
            ),
            "pcd_obs": pcd[..., :3],
            "pcd_mask": ~to_remove_indices,
            "state_obs": einops.pack(  # for [0] at the end, see #pack return in einops
                [obs["agent"]["qpos"], obs["agent"]["qvel"], obs["extra"]["tcp_pose"]], "ix *"
            )[0],
        }
    else:
        raise NotImplementedError(f"obs_mode {obs_mode} not implemented")

    if keep_keys is not None:
        retval = {k: v for k, v in retval.items() if k in keep_keys}

    return retval
