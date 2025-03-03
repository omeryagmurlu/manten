import logging

import einops
import optree
import torch
from torch import nn

from manten.networks.utils.mixins import ModuleAttrMixin
from manten.networks.vendor.diffusion_policy.diffusion.positional_embedding import (
    SinusoidalPosEmb,
)

logger = logging.getLogger(__name__)


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


class MantenTransformerV2(ModuleAttrMixin):
    def __init__(
        self,
        *,
        act_dim,
        pred_horizon,
        obs_horizon,
        cond_type_num_tokens: dict[str, int],
        cond_type_input_dims: dict,
        cond_types=("rgb", "pcd", "state"),
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        n_cond_layers: int = 0,
    ) -> None:
        super().__init__()

        cond_type_num_tokens = optree.tree_map(
            lambda x: (obs_horizon, x) if isinstance(x, int) else x,
            cond_type_num_tokens,
        )

        # hack for now to combine conds with the signature
        cond_type_num_tokens = {"_".join(cond_types): (obs_horizon, 1)}
        cond_type_input_dims = {
            "_".join(cond_types): sum(cond_type_input_dims[name] for name in cond_types)
        }
        self.orig_cond_types = cond_types
        cond_types = ["_".join(cond_types)]

        # input embedding stem
        self.sample_emb = nn.Linear(act_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, pred_horizon, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_embs = nn.ModuleDict(
            {
                f"cond_{name}_embed": get_named_embed(cond_type_input_dims[name], n_emb)
                for name in cond_types
            }
        )
        self.cond_pos_embs = nn.ParameterDict(
            {
                f"cond_{name}_pos_embed": get_named_pos_embed(
                    cond_type_num_tokens[name], n_emb
                )
                for name in cond_types
            }
        )

        # if n_cond_layers > 0:
        #     encoder_layer = nn.TransformerEncoderLayer(
        #         d_model=n_emb,
        #         nhead=n_head,
        #         dim_feedforward=4 * n_emb,
        #         dropout=p_drop_attn,
        #         activation="gelu",
        #         batch_first=True,
        #         norm_first=True,
        #     )
        #     self.encoder = nn.TransformerEncoder(
        #         encoder_layer=encoder_layer, num_layers=n_cond_layers
        #     )
        # else:
        self.encoder = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb)
        )
        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # important for stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layer)

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = seq_len_input
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = (
                mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
            )
            self.register_buffer("mask", mask)

            S = seq_len_cond
            t, s = torch.meshgrid(torch.arange(seq_len_input), torch.arange(S), indexing="ij")
            mask = t >= (s - 1)  # add one dimension since time is the first token in cond
            mask = (
                mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
            )
            self.register_buffer("memory_mask", mask)
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, act_dim)

        # init
        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):  # noqa: C901
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            nn.ModuleDict,
            nn.ParameterDict,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, MantenTransformerV2):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            for emb in module.cond_pos_embs.values():
                torch.nn.init.normal_(emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise TypeError(f"Unaccounted module {module}")

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = dict(self.named_parameters())
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            f"parameters {inter_params!s} made it into both decay/no_decay sets!"
        )
        assert len(param_dict.keys() - union_params) == 0, (
            f"parameters {(param_dict.keys() - union_params)!s} were not separated into either decay/no_decay set!"
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(self, sample, conds, timestep=None, **_kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps)  # (B,n_emb)

        # hack for conds:
        comb_cond_name = "_".join(self.orig_cond_types)
        conds = {
            comb_cond_name: torch.cat([conds[name] for name in self.orig_cond_types], dim=-1)
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
        memory = x  # (B,T_cond,n_emb)

        # decoder
        x = self.sample_emb(sample) + self.pos_emb  # (B,pred_horizon,n_emb)
        x = self.drop(x)
        x = self.decoder(
            tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask
        )

        # head
        x = self.ln_f(x)
        x = self.head(x)  # (B,T,n_out)
        return x


def test():
    # GPT with time embedding
    transformer = MantenTransformerV2(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4, 8, 16))
    out = transformer(sample, timestep)

    # GPT with time embedding and obs cond
    transformer = MantenTransformerV2(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4, 8, 16))
    cond = torch.zeros((4, 4, 10))
    out = transformer(sample, timestep, cond)

    # GPT with time embedding and obs cond and encoder
    transformer = MantenTransformerV2(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        n_cond_layers=4,
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4, 8, 16))
    cond = torch.zeros((4, 4, 10))
    out = transformer(sample, timestep, cond)

    # BERT with time embedding token
    transformer = MantenTransformerV2(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        # causal_attn=True,
        time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4, 8, 16))
    out = transformer(sample, timestep)

    print(out.shape, opt)
