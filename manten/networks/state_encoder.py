import itertools

import torch
from torch import nn


def with_keys(x, keys):
    return {key: x[key] for key in x if key in keys}


def without_keys(x, keys):
    return {key: x[key] for key in x if key not in keys}


class NoopEncoder(nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()

    def forward(self, x):
        return x


class CatAllEncoder(nn.Module):
    def __init__(self, *, excluded_keys=(), **_kwargs):
        super().__init__()
        self.excluded_keys = excluded_keys

    def forward(self, x):
        x = without_keys(x, self.excluded_keys)
        return torch.cat(x.values(), dim=-1)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        *,
        state_shape: dict[str, list],  # list of shapes w/o batch dim
        excluded_keys=(),
        hidden_dims=(64,),
        output_dim=64,
        activation=nn.ReLU,
        **_kwargs,
    ):
        super().__init__()
        valid_shape = without_keys(state_shape, excluded_keys)
        input_dim = sum(shape[-1] for shape in valid_shape.values())

        self.net = nn.Sequential(
            nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), activation()),
            *[
                nn.Sequential(nn.Linear(hid_in, hid_out), activation())
                for hid_in, hid_out in itertools.pairwise(hidden_dims)
            ],
            nn.Linear(hidden_dims[-1], output_dim),
        )

        self.keys = list(valid_shape.keys())

    def forward(self, x):
        x = with_keys(x, self.keys)
        x = torch.cat(list(x.values()), dim=-1)
        return self.net(x)
