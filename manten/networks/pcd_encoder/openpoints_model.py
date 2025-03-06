import einops
import optree
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from openpoints.models.build import build_model_from_cfg
from openpoints.utils.config import EasyConfig
from torch import nn

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
    def __init__(self, model, aggregate="mean", **_kwargs):
        super().__init__()
        self.model = model

        if aggregate is None:
            self.agg_fn = lambda x: x
        elif aggregate == "mean":
            self.agg_fn = lambda x: einops.reduce(
                x, "b f c -> b f 1", reduction="mean"
            ).squeeze(-1)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")

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

        # # handle mask for now by setting all pos to 0
        # pcd_obs = optree.tree_map(lambda x, mask: x * mask, pcd_obs, pcd_mask)

        pcd = torch.cat([*pcd_obs.values()], dim=-1).contiguous()

        res = self.model(pcd)

        res = self.agg_fn(res)

        return res
