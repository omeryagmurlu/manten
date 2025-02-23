import einops
import numpy as np
import optree
import torch

from manten.networks.utils.rotation_transformer import RotationTransformer


def _get_backend(thing):
    if isinstance(thing, np.ndarray):
        return np
    if isinstance(thing, torch.Tensor):
        return torch
    raise ValueError(f"Unknown type {type(thing)}")


def _get_cat_fn(thing):
    if isinstance(thing, np.ndarray):
        return np.concatenate
    if isinstance(thing, torch.Tensor):
        return torch.cat
    raise ValueError(f"Unknown type {type(thing)}")


def get_to_remove_indices(pcd):
    FAR_THRESHOLD = 0.5  # 0 for camera far, 1 camera near
    FOCUS_THRESHOLD = 1  # meters

    abs_fn = _get_backend(pcd).abs

    far_indices = pcd[..., 3] < FAR_THRESHOLD
    unfocused_indices = (
        (abs_fn(pcd[..., 0]) > FOCUS_THRESHOLD)
        | (abs_fn(pcd[..., 1]) > FOCUS_THRESHOLD)
        | (abs_fn(pcd[..., 2]) > FOCUS_THRESHOLD)
    )
    to_remove_indices = far_indices | unfocused_indices

    return to_remove_indices


def process_observation_from_raw(  # noqa: C901
    obs,
    obs_mode,
    image_size: int = 128,
    cam=None,
    rgb_modality_keys=None,
    slice_rgb_modality_to_ncam=False,
):
    if obs_mode == "state":
        return {"state_obs": obs}

    if rgb_modality_keys is None:
        rgb_modality_keys = []

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
            image_size = int(_get_backend(pcd).sqrt(n_points // cam))

        if slice_rgb_modality_to_ncam:
            rgb_modality_keys = rgb_modality_keys[:cam]

        if len(rgb_modality_keys) != cam:
            raise ValueError(
                f"Number of cameras in rgb_modality_keys ({len(rgb_modality_keys)}) does not match the number of cameras in the pointcloud ({cam})"
            )

        to_remove_indices = get_to_remove_indices(pcd)

        rgb_obs = (
            einops.rearrange(
                obs["pointcloud"]["rgb"],
                "ix (cam h w) c -> cam ix c h w",
                cam=cam,
                h=image_size,
                w=image_size,
                c=3,
            )
            / 255.0
        )
        pcd_obs = einops.rearrange(
            pcd[..., :3],
            "ix (cam h w) c -> cam ix c h w",
            cam=cam,
            h=image_size,
            w=image_size,
            c=3,
        )
        pcd_mask = einops.rearrange(
            ~to_remove_indices,
            "ix (cam h w) -> cam ix 1 h w",
            cam=cam,
            h=image_size,
            w=image_size,
        )

        vision_elems = {}
        for cam_i, cam in enumerate(rgb_modality_keys):
            vision_elems[f"rgb_obs.{cam}"] = rgb_obs[cam_i]
            vision_elems[f"pcd_obs.{cam}"] = pcd_obs[cam_i]
            vision_elems[f"pcd_mask.{cam}"] = pcd_mask[cam_i]

        return {
            **extra_elems,
            **vision_elems,
        }
    elif obs_mode == "rgb":
        rgbs = optree.tree_flatten(obs["sensor_data"])[0]
        # einops.rearrange(rgbs, "cam ix h w c -> ix cam h w c")

        if slice_rgb_modality_to_ncam:
            rgb_modality_keys = rgb_modality_keys[: len(rgbs)]

        if len(rgb_modality_keys) != len(rgbs):
            raise ValueError(
                f"Number of cameras in rgb_modality_keys ({len(rgb_modality_keys)}) does not match the number of cameras in the rgb data ({len(rgbs)})"
            )

        vision_elems = {}
        for cam_i, cam in enumerate(rgb_modality_keys):
            vision_elems[f"rgb_obs.{cam}"] = rgbs[cam_i]

        return {
            **extra_elems,
            **vision_elems,
        }

    raise ValueError(f"obs_mode {obs_mode} not recognized")


def apply_static_transforms(obs_dict, *, obs_mode):  # noqa: ARG001
    # if obs_mode == "rgb":
    #     # rbg_obs is a dict of key(cam_name): (obs_horizon, C, H, W)
    #     for cam_key, cam_v in obs_dict["rgb_obs"].items():
    #         cam = einops.rearrange(cam_v, "t h w c -> t c h w")
    #         obs_dict["rgb_obs"][cam_key] = cam.float() / 255.0
    # elif obs_mode == "pointcloud":
    #     # rgb_obs is a tensor of shape (obs_horizon, ncam, C, H, W)
    #     # TODO: use transformpcd from pointcloudmatters
    #     raise NotImplementedError("Pointcloud transformation not implemented")

    return obs_dict


def transform_episode_obs(
    loader_dict, *, obs_mode, obs_modalities, rgb_modality_keys, state_modality_keys
):
    obs_dict = {}
    for key in obs_modalities:
        if key == "state_obs" and obs_mode != "state":
            # we need to aggregate the state modalities into one
            if state_modality_keys:  # only if there are state modalities
                obs_dict[key] = get_state_from_modality(loader_dict, state_modality_keys)
        elif key == "rgb_obs" or key == "pcd_obs" or key == "pcd_mask":
            # need to combine cams
            obs_dict[key] = segregate_vision_by_modality(
                loader_dict, rgb_modality_keys, main_key=key
            )
        else:
            obs_dict[key] = loader_dict[key]

    return obs_dict


def transform_episode_actions(loader_dict, *, rotation_transformer=None):
    actions = loader_dict["actions"]
    if rotation_transformer is not None:
        actions = forward_rotation_transformer(actions, rotation_transformer)

    return actions


def transform_episode(
    loader_dict,
    *,
    obs_mode,
    obs_modalities,
    rgb_modality_keys,
    state_modality_keys,
    rotation_transformer,
):
    return {
        "actions": transform_episode_actions(
            loader_dict, rotation_transformer=rotation_transformer
        ),
        "observations": transform_episode_obs(
            loader_dict,
            obs_mode=obs_mode,
            obs_modalities=obs_modalities,
            rgb_modality_keys=rgb_modality_keys,
            state_modality_keys=state_modality_keys,
        ),
    }


def back_transform_episode_actions(actions, *, rotation_transformer=None):
    if rotation_transformer is not None:
        actions = inverse_rotation_transformer(actions, rotation_transformer)

    return actions


def forward_rotation_transformer(actions, rotation_transformer: RotationTransformer):
    """Maniskill actions are (x, y, z, R..., gripper)"""
    from_dim = rotation_transformer.from_dim
    rotations = actions[..., 3 : 3 + from_dim]
    transformed_rotations = rotation_transformer.forward(rotations)
    return _get_cat_fn(rotations)(
        [actions[..., :3], transformed_rotations, actions[..., 3 + from_dim :]], -1
    )


def inverse_rotation_transformer(actions, rotation_transformer: RotationTransformer):
    to_dim = rotation_transformer.to_dim
    rotations = actions[..., 3 : 3 + to_dim]
    transformed_rotations = rotation_transformer.inverse(rotations)
    return _get_cat_fn(rotations)(
        [actions[..., :3], transformed_rotations, actions[..., 3 + to_dim :]], -1
    )


def segregate_vision_by_modality(loader_dict, rgb_modality_keys, main_key):
    if not rgb_modality_keys:
        raise ValueError("No rgb modality keys provided")

    return {k: loader_dict[f"{main_key}.{k}"] for k in rgb_modality_keys}


def get_state_from_modality(loader_dict, state_modality_keys):
    if not state_modality_keys:
        raise ValueError("No state modality keys provided")

    return {k: loader_dict[k] for k in state_modality_keys}
