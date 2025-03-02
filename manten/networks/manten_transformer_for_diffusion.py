import math
from collections.abc import Callable
from contextlib import nullcontext
from random import randrange

import einops
import optree
import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import (
    AttentionLayers,
    Decoder,
    LayerIntermediates,
    LayerNorm,
    LinearNoBias,
    ScaledSinusoidalEmbedding,
    at_most_one_of,
    calc_entropy,
    calc_z_loss,
    cast_tuple,
    default,
    exists,
    first,
    masked_mean,
    pad_at_dim,
)


class MantenTransformerForDiffusion(nn.Module):
    def __init__(
        self,
        *,
        act_dim: int,
        pred_horizon: int,
        obs_horizon: int,
        cond_type_num_tokens: dict[str, int],
        cond_type_input_dims: dict[str, int],
        dim: int | None = None,
        attn_layers: Callable[..., AttentionLayers] | AttentionLayers,
        cond_types: tuple[str] = ("state", "rgb", "pcd"),
        emb_dim=None,
        max_mem_len=0,
        shift_mem_down=0,
        emb_dropout=0.0,
        post_emb_norm=False,
        num_memory_tokens=None,
        memory_tokens_interspersed_every=None,
        logits_dim=None,
        return_only_embed=False,
        num_output_heads=1,
        recycling=False,  # from Jumper et al. - Alphafold2
        train_max_recycle_steps=4,  # saw a benefit for language modeling up to 3 recycling steps, so let's default this to 4
        emb_frac_gradient=1.0,  # GLM-130B and Cogview successfully used this, set at 0.1
        average_pool_embed=False,
        use_cls_token=False,
        num_cls_tokens=1,
        squeeze_out_last_dim=False,
    ):
        super().__init__()

        dim = dim if dim is not None else attn_layers.dim
        emb_dim = default(emb_dim, dim)
        self.emb_dim = emb_dim
        self.num_cls_tokens = num_cls_tokens

        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim

        self.sample_emb = nn.Linear(act_dim, emb_dim)
        # pred horizon is taken as num_token
        self.sample_pos_emb = nn.Parameter(torch.randn(pred_horizon, emb_dim))
        self.cond_embs = nn.ModuleDict(
            {
                f"cond_{name}_embed": nn.Linear(cond_type_input_dims[name], emb_dim)
                for name in cond_types
            }
        )
        self.cond_pos_embs = nn.ParameterDict(
            {
                f"cond_{name}_pos_embed": nn.Parameter(
                    torch.randn(obs_horizon, cond_type_num_tokens[name], emb_dim)
                )
                for name in cond_types
            }
        )
        self.time_emb = ScaledSinusoidalEmbedding(emb_dim)

        # fraction of the gradient that should go to the embedding, https://arxiv.org/abs/2105.13290

        self.emb_frac_gradient = emb_frac_gradient

        self.post_emb_norm = LayerNorm(emb_dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()

        if isinstance(attn_layers, AttentionLayers):
            self.attn_layers = attn_layers
        else:
            self.attn_layers = attn_layers(dim=dim)

        self.init_()

        assert num_output_heads > 0

        assert at_most_one_of(average_pool_embed, use_cls_token)

        # maybe recycling

        self.recycling = recycling
        self.recycled_proj = LinearNoBias(dim, dim) if recycling else None

        self.train_max_recycle_steps = train_max_recycle_steps

        # classic cls token from the bert days

        self.cls_token = None

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(num_cls_tokens, dim))
            nn.init.normal_(self.cls_token, std=0.02)

        # whether to average pool the embed (`global average pool`)

        self.average_pool_embed = average_pool_embed

        # output head, usually to logits of num_tokens

        logits_dim = default(logits_dim, self.act_dim)

        self.has_multiple_heads = num_output_heads > 1

        if return_only_embed:
            self.to_logits = None
        elif num_output_heads > 1:
            self.to_logits = nn.ModuleDict(
                [LinearNoBias(dim, logits_dim) for _ in range(num_output_heads)]
            )
        else:
            self.to_logits = LinearNoBias(dim, logits_dim)

        # memory tokens (like [cls]) from Memory Transformers paper

        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.memory_tokens_interspersed_every = memory_tokens_interspersed_every

        # squeeze out last dimension if possible

        self.squeeze_out_last_dim = squeeze_out_last_dim

        # whether can do cached kv decoding

        self.can_cache_kv = (
            self.num_memory_tokens == 0 and not recycling and self.attn_layers.can_cache_kv
        )
        self.can_cache_kv_outside_max_seq_len = False

    def init_(self):
        # init embs
        nn.init.normal_(self.sample_pos_emb, std=1e-5)
        for emb in self.cond_pos_embs.values():
            nn.init.normal_(emb, std=1e-5)

    def forward(
        self,
        sample,
        timestep,
        conds,
        return_embeddings=False,
        return_logits_and_embeddings=False,
        return_intermediates=False,
        return_logit_entropies=False,
        return_sample_tokens=True,
        mask=None,
        return_mems=False,
        return_attn=False,
        mems=None,
        mem_masks=None,
        recycle_steps=None,
        return_attn_z_loss=False,
        attn_z_loss_weight=1e-4,
        seq_start_pos=None,
        cache: LayerIntermediates | None = None,
        **kwargs,
    ):
        sample_cond = next(iter(conds.values()))
        (
            b,
            obs_horizon,
            pred_horizon,
            device,
            num_mems,
            has_memory_tokens,
            emb_frac_gradient,
            orig_mask,
        ) = (
            sample.shape[0],
            sample_cond.shape[-3],
            sample.shape[-2],
            sample.device,
            self.num_memory_tokens,
            self.num_memory_tokens > 0,
            self.emb_frac_gradient,
            mask,
        )

        assert self.obs_horizon == obs_horizon, "input shape mismatch, check `obs_horizon`"
        assert self.pred_horizon == pred_horizon, "input shape mismatch, check `pred_horizon`"

        return_hiddens = return_mems | return_attn | return_intermediates | return_attn_z_loss
        return_embeddings = return_embeddings | (not exists(self.to_logits))

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time = self.time_emb(timesteps.unsqueeze(0)).unsqueeze(1)  # B 1 D

        sample = self.sample_emb(sample) + self.sample_pos_emb  # B T D

        conds = {
            k: self.cond_embs[f"cond_{k}_embed"](v)
            + self.cond_pos_embs[f"cond_{k}_pos_embed"]
            for k, v in conds.items()
        }
        conds = torch.cat(
            [einops.rearrange(elem, "b n t d -> b (n t) d") for elem in conds.values()],
            dim=-2,
        )  # B N*T D

        x = torch.cat([sample, time, conds], dim=-2)  # B num_token D

        sample_slice = (slice(None), slice(0, sample.shape[1]), slice(None))
        n = x.shape[1]

        if mask is not None:
            # extend mask to include conditioning as ones
            mask = torch.cat(
                [
                    mask,
                    torch.ones(time.shape[:-1], device=device, dtype=torch.bool),
                    torch.ones(conds.shape[:-1], device=device, dtype=torch.bool),
                ],
                dim=-1,
            )

        # post embedding norm, purportedly leads to greater stabilization

        x = self.post_emb_norm(x)

        # whether to reduce the gradient going to the embedding, from cogview paper, corroborated by GLM-130B model

        if emb_frac_gradient < 1:
            assert emb_frac_gradient > 0
            x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)

        # embedding dropout

        x = self.emb_dropout(x)

        x = self.project_emb(x)

        # maybe cls token

        if exists(self.cls_token):
            cls_tokens = einops.repeat(self.cls_token, "... -> b ...", b=b)
            x, cls_packed_shape = einops.pack([cls_tokens, x], "b * d")

            if exists(mask):
                mask = F.pad(mask, (self.num_cls_tokens, 0), value=True)

        # maybe memory / register tokens

        if has_memory_tokens:
            mem_seq = x.shape[-2]
            mem_every = self.memory_tokens_interspersed_every

            if exists(mem_every):
                assert mem_every > 0
                assert isinstance(self.attn_layers, Decoder), "only for decoder"
                next_seq_len = math.ceil(n / mem_every) * mem_every

                x = pad_at_dim(x, (0, next_seq_len - n), dim=-2, value=0.0)
                x = einops.rearrange(x, "b (n m) d -> (b n) m d", m=mem_every)

            mem = einops.repeat(self.memory_tokens, "n d -> b n d", b=x.shape[0])
            x, mem_packed_shape = einops.pack((mem, x), "b * d")

            # auto-handle masking after appending memory tokens
            if not exists(mem_every) and exists(mask):
                mask = pad_at_dim(mask, (num_mems, 0), dim=-1, value=True)

            if exists(mem_every):
                x = einops.rearrange(x, "(b n) m d -> b (n m) d", b=b)

        # handle maybe shifting of memories

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[: self.shift_mem_down], mems[self.shift_mem_down :]
            mems = [*mems_r, *mems_l]

        # attention layers

        if not self.recycling:
            assert not exists(recycle_steps) or recycle_steps == 1, (
                "you did not train with recycling"
            )

            # regular

            attended, intermediates = self.attn_layers(
                x,
                mask=mask,
                mems=mems,
                mem_masks=mem_masks,
                cache=cache,
                return_hiddens=True,
                seq_start_pos=seq_start_pos,
                **kwargs,
            )

        else:
            # recycling

            recycle_steps = default(
                recycle_steps,
                (randrange(self.train_max_recycle_steps) + 1) if self.training else None,
            )
            assert exists(recycle_steps) and recycle_steps > 0, (  # noqa: PT018
                "`recycle_steps` must be provided on forward if recycling is turned on and not training"
            )

            for i in range(recycle_steps):
                first_step = i == 0
                last_step = i == (recycle_steps - 1)

                context = nullcontext if last_step else torch.no_grad

                with context():
                    maybe_recycled = (
                        self.recycled_proj(attended.detach()) if not first_step else 0.0
                    )

                    attended, intermediates = self.attn_layers(
                        x + maybe_recycled,
                        mask=mask,
                        mems=mems,
                        mem_masks=mem_masks,
                        cache=cache,
                        return_hiddens=True,
                        seq_start_pos=seq_start_pos,
                        **kwargs,
                    )

        x = attended

        # handle memories post-attention

        if has_memory_tokens:
            if exists(mem_every):
                x = einops.rearrange(x, "b (n m) d -> (b n) m d", m=(mem_every + num_mems))

            mem, x = einops.unpack(x, mem_packed_shape, "b * d")

            intermediates.memory_tokens = mem

            if exists(mem_every):
                x = einops.rearrange(x, "(b n) m d -> b (n m) d", b=b)

            x = x[:, :mem_seq]

        # global average pool

        if self.average_pool_embed:
            x = masked_mean(x, mask=orig_mask, dim=1)

        if exists(self.cls_token):
            x, _ = einops.unpack(x, cls_packed_shape, "b * d")
            x = x.squeeze(
                1
            )  # Remove sequence dimension if num_cls_tokens=1 to keep previous behavior

        # projecting to logits

        if not return_embeddings:
            if self.has_multiple_heads:
                logits = tuple(fn(x) for fn in self.to_logits)
            else:
                logits = self.to_logits(x)

        # maybe squeeze out last dimension of logits

        if self.squeeze_out_last_dim:
            logits = tuple(
                (einops.rearrange(t, "... 1 -> ...") if t.shape[-1] == 1 else t)
                for t in cast_tuple(logits)
            )

            if not self.has_multiple_heads:
                logits = first(logits)

        # different returns

        if return_logits_and_embeddings:
            out = (logits, x)
        elif return_embeddings:
            out = x
        else:
            out = logits

        if return_sample_tokens:
            out = optree.tree_map(lambda x: x[sample_slice], out)

        # logit entropies

        if return_logit_entropies:
            intermediates.logit_entropies = calc_entropy(logits)
            return_intermediates = True

        # aux loss

        if return_attn_z_loss:
            pre_softmax_attns = [t.pre_softmax_attn for t in intermediates.attn_intermediates]
            intermediates.attn_z_loss = calc_z_loss(
                pre_softmax_attns, weight=attn_z_loss_weight
            )
            return_intermediates = True

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = (
                [torch.cat(pair, dim=-2) for pair in zip(mems, hiddens, strict=False)]
                if exists(mems)
                else hiddens
            )
            new_mems = [t[..., -self.max_mem_len :, :].detach() for t in new_mems]

            if not return_intermediates:
                return out, new_mems

            intermediates.mems = new_mems

        if return_intermediates:
            return out, intermediates

        if return_attn:
            attn_maps = [t.post_softmax_attn for t in intermediates.attn_intermediates]
            return out, attn_maps

        return out
