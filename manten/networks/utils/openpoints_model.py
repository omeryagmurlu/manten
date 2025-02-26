import einops
import optree
from omegaconf import DictConfig, ListConfig, OmegaConf
from openpoints.models.build import build_model_from_cfg
from openpoints.utils.config import EasyConfig

import torch
import torch.nn as nn

# def create_openpoints_model(**kwargs):
#     kwargs = optree.tree_map(
#         lambda x: OmegaConf.to_object(x) if isinstance(x, (DictConfig, ListConfig)) else x,
#         kwargs,
#     )

#     cfg = EasyConfig()
#     cfg.update(kwargs)

#     model = build_model_from_cfg(cfg)
#     return model


class OpenPointsModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        kwargs = optree.tree_map(
            lambda x: OmegaConf.to_object(x)
            if isinstance(x, (DictConfig, ListConfig))
            else x,
            kwargs,
        )

        cfg = EasyConfig()
        cfg.update(kwargs)

        self.model = build_model_from_cfg(cfg)

    def forward(self, pcd_obs, pcd_mask):
        # pcd_obs['camera1'].shape
        # torch.Size([1, 3, 128, 128])
        # pcd_mask['camera1'].shape
        # torch.Size([1, 1, 128, 128])

        pcd_obs = optree.tree_map(
            lambda x: einops.rearrange(x, "b c h w -> b (h w) c"), pcd_obs
        )
        pcd_mask = optree.tree_map(
            lambda x: einops.rearrange(x, "b c h w -> b (h w) c"), pcd_mask
        )

        # handle mask for now by setting all pos to 0
        pcd_obs = optree.tree_map(lambda x, mask: x * mask, pcd_obs, pcd_mask)

        pcd = torch.cat([*pcd_obs.values()], dim=-1).contiguous()

        res = self.model(pcd)

        res = einops.reduce(res, "b f c -> b f 1", reduction="mean").squeeze(-1)

        return res
