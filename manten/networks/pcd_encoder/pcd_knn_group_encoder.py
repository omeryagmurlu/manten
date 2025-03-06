import einops
import optree
import torch
from torch import nn

from manten.networks.vendor.sugar.pc_transformer import PCGroupEncoder
from manten.networks.vendor.sugar.point_ops import PointGroup


class PCDKNNGroupEncoder(nn.Module):
    def __init__(
        self,
        hidden_size=384,
        input_size=6,
        num_groups=64,
        group_size=32,
        group_use_knn=True,
        group_radius=None,
        **_kwargs,
    ):
        super().__init__()

        self.num_groups = num_groups
        self.group_size = group_size
        self.group_use_knn = group_use_knn
        self.group_radius = group_radius
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.group_divider = PointGroup(
            num_groups=self.num_groups,
            group_size=self.group_size,
            knn=self.group_use_knn,
            radius=self.group_radius,
        )
        self.encoder = PCGroupEncoder(self.input_size, self.hidden_size)

    def forward(self, pcd_obs, _pcd_mask):
        pcd_obs = optree.tree_map(
            lambda x: einops.rearrange(x, "b c h w -> b (h w) c"), pcd_obs
        )

        pcd = torch.cat([*pcd_obs.values()], dim=-1).contiguous()

        neighborhoods, _centers = self.group_divider(pcd)

        group_input_tokens = self.encoder(neighborhoods)  # B G C

        return group_input_tokens
