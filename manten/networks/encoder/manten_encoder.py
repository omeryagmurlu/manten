from collections.abc import Callable
from typing import Literal

import einops
import optree
import torch
from torch import nn

from manten.utils.utils_pytree import tree_rearrange


class MantenEncoder(nn.Module):
    def __init__(
        self,
        train_modes: tuple[Literal["2d", "3d"]],
        rgb_shape: dict[str, tuple[int]] | None = None,
        rgb_encoder: Callable[..., nn.Module] | None = None,
        state_shape: dict[str, tuple[int]] | None = None,
        state_encoder: Callable[..., nn.Module] | None = None,
        pcd_shape: dict[str, tuple[int]] | None = None,
        pcd_encoder: Callable[..., nn.Module] | None = None,
        pcd_scaler=None,
    ):
        super().__init__()

        assert len(train_modes) > 0, "At least one train mode must be provided"
        assert all(mode in ["2d", "3d"] for mode in train_modes), (
            "Train modes must be either 2d or 3d"
        )

        if state_shape is not None:
            self.state_encoder = state_encoder(state_shape=state_shape)
        else:
            self.state_encoder = None

        if "2d" in train_modes:
            assert rgb_shape is not None, "RGB shape must be provided"
            self.rgb_encoder = rgb_encoder(rgb_shape=rgb_shape)
        else:
            self.rgb_encoder = None

        if "3d" in train_modes:
            assert pcd_shape is not None, "PCD shape must be provided"
            self.pcd_encoder = pcd_encoder(pcd_shape=pcd_shape, pcd_scaler=pcd_scaler)
        else:
            self.pcd_encoder = None

    def forward(self, *, rgb_obs, pcd_obs, pcd_mask, state_obs):
        shape = next(iter((rgb_obs or pcd_obs or state_obs).values())).shape
        B = shape[0]
        obs_horizon = shape[1]

        if self.state_encoder is not None:
            state_cond = einops.rearrange(
                self.state_encoder(state_obs), "b t ... -> b t 1 ...", b=B, t=obs_horizon
            )
        else:
            state_cond = torch.tensor([], device=self.device, dtype=self.dtype)

        if self.rgb_encoder is not None:
            global_rgb_cond, local_rgb_cond = optree.tree_map(
                lambda x: einops.rearrange(x, "(b t) n d -> b t n d", b=B, t=obs_horizon),
                self.rgb_encoder(tree_rearrange(rgb_obs, "b t c h w -> (b t) c h w")),
            )
        else:
            global_rgb_cond = torch.tensor([], device=self.device, dtype=self.dtype)
            local_rgb_cond = torch.tensor([], device=self.device, dtype=self.dtype)

        if self.pcd_encoder is not None:
            pcd_features = self.pcd_encoder(
                tree_rearrange(rgb_obs, "b t ... -> (b t) ..."),
                tree_rearrange(pcd_obs, "b t ... -> (b t) ..."),
                tree_rearrange(pcd_mask, "b t ... -> (b t) ..."),
            )
            pcd_cond = einops.rearrange(
                pcd_features, "(b t) n d -> b t n d", b=B, t=obs_horizon
            )
        else:
            pcd_cond = torch.tensor([], device=self.device, dtype=self.dtype)

        return state_cond, global_rgb_cond, local_rgb_cond, pcd_cond
