from collections.abc import Callable

import einops
import torch
from torch import nn

import manten.networks.vendor.openpoints.backbone.layers as pointnet2_utils
from manten.networks.vendor.sugar.knn import knn_point
from manten.networks.vendor.sugar.point_ops import PointGroup, fps


class MaskedPointGroup(nn.Module):  # FPS + KNN
    def __init__(self, num_groups, group_size, mask_ident, knn=True, radius=None):
        """
        Whether use knn or radius to get the neighborhood
        """
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.knn = knn
        self.radius = radius
        self.mask_ident = mask_ident

    def forward(self, pc_fts):
        """
        input: B N 3+x
        ---------------------------
        output: B G M 3+x
        center : B G 3
        """
        batch_size, num_points, _ = pc_fts.shape
        xyz = pc_fts[..., :3].contiguous()

        # fps the centers out
        # centers = fps(xyz, self.num_groups)  # B G 3

        fulls = (xyz != self.mask_ident).all(dim=[1, 2])

        centers = torch.empty(
            device=xyz.device,
            dtype=xyz.dtype,
            size=(xyz.shape[0], self.num_groups, 3),
        )
        if xyz[fulls].numel() != 0:  # I don't like this, probably breaks compile support
            centers[fulls] = fps(xyz[fulls], self.num_groups)
        if xyz[~fulls].numel() != 0:  # not this either
            non_full_centers = fps(xyz[~fulls], self.num_groups + 1)
            centers[~fulls] = non_full_centers[non_full_centers != self.mask_ident].view(
                -1, self.num_groups, 3
            )

        if self.knn:  # knn to get the neighborhood, shape=(batch, num_groups, group_size)
            idx = knn_point(self.group_size, xyz, centers)
        else:  # use radius to get the neighborhood (ball query), shape=(batch, num_groups, group_size)
            idx = pointnet2_utils.ball_query(self.radius, self.group_size, xyz, centers)
        assert idx.size(1) == self.num_groups
        assert idx.size(2) == self.group_size
        # shape=(batch, 1, 1)
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhoods = pc_fts.view(batch_size * num_points, -1)[idx, :]
        neighborhoods = neighborhoods.view(
            batch_size, self.num_groups, self.group_size, -1
        ).contiguous()
        # normalize
        neighborhoods[..., :3] = neighborhoods[..., :3] - centers.unsqueeze(2)
        return neighborhoods, centers


class MantenPCDEncoder(nn.Module):
    def __init__(
        self,
        *,
        pcd_shape: dict[str, tuple[int]],  # noqa: ARG002
        pcd_model: Callable[..., nn.Module],
        pcd_scaler=None,
        use_color_in_pcd: bool = False,
        use_mask_in_pcd: bool = True,
        mask_ident: int = -1e9,
        use_pcd_grouper: bool = True,
        pcd_grouper_kwargs=None,
    ):
        super().__init__()
        self.pcd_scaler = pcd_scaler

        self.use_color_in_pcd = use_color_in_pcd
        if self.use_color_in_pcd:
            self.pcd_model = pcd_model(in_channels=6)
        else:
            self.pcd_model = pcd_model(in_channels=3)

        self.use_mask_in_pcd = use_mask_in_pcd
        self.mask_ident = mask_ident

        if use_pcd_grouper:
            if pcd_grouper_kwargs is None:
                pcd_grouper_kwargs = {}
            if self.use_mask_in_pcd:
                self.pcd_grouper = MaskedPointGroup(
                    mask_ident=mask_ident, **pcd_grouper_kwargs
                )
            else:
                self.pcd_grouper = PointGroup(**pcd_grouper_kwargs)

    def forward(self, rgb_obs, pcd_obs, pcd_mask):
        shape = next(iter(pcd_obs.values())).shape
        B = shape[0]

        pcd_obs = einops.rearrange(list(pcd_obs.values()), "cam b c h w -> b (cam h w) c")
        pcd_mask = einops.rearrange(list(pcd_mask.values()), "cam b c h w -> b (cam h w) c")

        if self.pcd_scaler is not None:
            pcd_obs = self.pcd_scaler.scale(pcd_obs)

        if self.use_color_in_pcd:
            rgb_obs = einops.rearrange(list(rgb_obs.values()), "cam b c h w -> b (cam h w) c")
            pointcloud = torch.cat([pcd_obs, rgb_obs], dim=-1)
        else:
            pointcloud = pcd_obs

        self.apply_mask(pointcloud, pcd_mask)  # b n c

        if self.pcd_grouper is not None:
            pointcloud, _ = self.pcd_grouper(pointcloud)  # b g n c
            _, G, N, _ = pointcloud.shape
            pointcloud = einops.rearrange(pointcloud, "b g n c -> (b g) n c")
        else:  # b n c
            _, N, _ = pointcloud.shape
            G = 1

        pcd_features = self.pcd_model(pointcloud)  # btg d
        return einops.rearrange(pcd_features, "(b g) d -> b g d", b=B, g=G)

    def apply_mask(self, pointcloud, pcd_mask):
        if not self.use_mask_in_pcd:
            return

        pointcloud[~pcd_mask.squeeze(-1)] = self.mask_ident
