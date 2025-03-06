import math

import einops
import optree
import torch
from torch import nn
from x_transformers.x_transformers import (
    Decoder,
    Encoder,
)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def get_named_embed(input_dim, dim):
    if isinstance(input_dim, int):
        return nn.Linear(input_dim, dim)
    else:
        raise ValueError(f"input_dim {input_dim} not recognized")  # noqa: TRY004


def get_named_pos_embed(num_token, dim):
    if isinstance(num_token, int):
        return nn.Parameter(torch.randn(num_token, dim))
    else:
        for i in num_token:
            assert isinstance(i, int)
        return nn.Parameter(torch.randn(*num_token, dim))


class MantenTransformer(nn.Module):
    def __init__(
        self,
        *,
        act_dim,
        pred_horizon,
        obs_horizon,
        cond_type_num_tokens: dict[str, int],
        cond_type_input_dims: dict,
        cond_types=("rgb", "pcd", "state"),
        depth: int = 12,
        dim: int = 768,
        emb_dropout: float = 0.1,
        depth_encoder: int = 0,
        ff_mult: int = 4,
        layer_kwargs: dict | None = None,
        concat_cond_dims_into_one=False,
    ):
        if layer_kwargs is None:
            layer_kwargs = {}

        super().__init__()

        cond_type_num_tokens = optree.tree_map(
            lambda x: (obs_horizon, x) if isinstance(x, int) else x,
            cond_type_num_tokens,
        )

        self.cond_concat_hack = concat_cond_dims_into_one
        if self.cond_concat_hack:
            cond_type_num_tokens = {"_".join(cond_types): (obs_horizon, 1)}
            cond_type_input_dims = {
                "_".join(cond_types): sum(cond_type_input_dims[name] for name in cond_types)
            }
            self.orig_cond_types = cond_types
            cond_types = ["_".join(cond_types)]

        self.sample_emb = nn.Linear(act_dim, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, pred_horizon, dim))
        self.drop = nn.Dropout(emb_dropout)

        self.time_emb = SinusoidalPosEmb(dim)
        self.cond_embs = nn.ModuleDict(
            {
                f"cond_{name}_embed": get_named_embed(cond_type_input_dims[name], dim)
                for name in cond_types
            }
        )
        self.cond_pos_embs = nn.ParameterDict(
            {
                f"cond_{name}_pos_embed": get_named_pos_embed(cond_type_num_tokens[name], dim)
                for name in cond_types
            }
        )

        if depth_encoder > 0:
            self.encoder = Encoder(
                dim=dim,
                depth=depth_encoder,
                ff_mult=ff_mult,
                **layer_kwargs,
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(dim, ff_mult * dim), nn.Mish(), nn.Linear(ff_mult * dim, dim)
            )

        self.decoder = Decoder(
            dim=dim,
            depth=depth,
            ff_mult=ff_mult,
            cross_attend=True,
            **layer_kwargs,
        )

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, act_dim)

        self.init_()

    def init_(self):
        torch.nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        for emb in self.cond_pos_embs.values():
            torch.nn.init.normal_(emb, mean=0.0, std=0.02)

    def forward(self, sample, conds, timestep=None):
        timestep = timestep.expand(sample.shape[0])
        time_emb = self.time_emb(timestep)  # (B,n_emb)

        if self.cond_concat_hack:
            comb_cond_name = "_".join(self.orig_cond_types)
            conds = {
                comb_cond_name: torch.cat(
                    [conds[name] for name in self.orig_cond_types], dim=-1
                )
            }

        # encoder
        conds = {
            k: self.cond_embs[f"cond_{k}_embed"](v)
            + self.cond_pos_embs[f"cond_{k}_pos_embed"]
            for k, v in conds.items()
        }
        x = torch.cat(
            [einops.rearrange(time_emb, "b d -> b 1 d")]
            + [einops.rearrange(elem, "b t n d -> b (t n) d") for elem in conds.values()],
            dim=1,
        )  # (B, obs_horizon*N, n_emb)
        x = self.drop(x)
        x = self.encoder(x)
        context = x

        # if self.training and self.cross_attn_tokens_dropout > 0:
        #     context, _ = dropout_seq(context, None, self.cross_attn_tokens_dropout)

        x = self.sample_emb(sample) + self.pos_emb  # (B,pred_horizon,n_emb)
        x = self.drop(x)
        x = self.decoder(x, context=context)
        # , context_mask=mask

        x = self.ln_f(x)
        x = self.head(x)
        return x
