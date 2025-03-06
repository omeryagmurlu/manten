import torch


def noop_encoder(x, **_kwargs):
    return x


# def cat_all_encoder(x, **_kwargs):
#     return torch.cat(list(x.values()), dim=-1)


def cat_all_encoder(x, excluded_keys=(), **_kwargs):
    return torch.cat([x[key] for key in x if key not in excluded_keys], dim=-1)
