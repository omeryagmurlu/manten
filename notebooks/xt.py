import optree
import torch
from torch import nn
from x_transformers import Encoder

from manten.networks.manten_transformer_for_diffusion import MantenTransformerForDiffusion

# model = MantenTransformerForDiffusion(
#     act_dim=10,
#     pred_horizon=16,
#     obs_horizon=1,
#     cond_type_num_tokens={"rgb": 1, "pcd": 64, "state": 1},
#     cond_type_input_dims={"rgb": 512, "pcd": 256, "state": 3},
#     attn_layers=lambda dim: Encoder(dim=dim, depth=12, heads=8),
#     dim=512,
# ).cuda()

# x = torch.rand((48, 16, 10)).cuda()
# conds = {
#     "rgb": torch.rand((48, 1, 1, 512)).cuda(),
#     "pcd": torch.rand((48, 1, 64, 256)).cuda(),
#     "state": torch.rand((48, 1, 1, 3)).cuda(),
# }
# timesteps = torch.arange(48).cuda()
# mask = torch.ones(x.shape[:-1], device=x.device).bool()

# print(16 + 1 + 64 + 1 + 1)

# out = model(
#     x, timesteps, conds=conds, mask=mask, return_logits_and_embeddings=True
# )  # (1, 128, 20000)

# print(optree.tree_map(lambda x: x.shape, out))

t1 = torch.tensor([1, 3, 5])
t2 = torch.tensor([2, 4, 6])

odd = torch.tensor([True, False, True, False, True, False])
even = ~odd

combined = torch.zeros(6, dtype=t1.dtype)
combined[odd] = t1
combined[even] = t2

print(combined)
