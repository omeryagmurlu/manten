from abc import ABC, abstractmethod

import torch
from torch import nn

DEFAULT_MAX = 1.0  # with tolerance


class Scaler(nn.Module, ABC):
    def __init__(self, **_):
        super().__init__()

    @abstractmethod
    def scale(self, x):
        raise NotImplementedError

    @abstractmethod
    def descale(self, x):
        raise NotImplementedError


class NoopScaler(Scaler):
    def __init__(self, **_):
        super().__init__()

    def scale(self, x):
        return x

    def descale(self, x):
        return x


class MinMaxScaler(Scaler):
    """Min-max scaling."""

    def __init__(self, *, min, max, min_value=-DEFAULT_MAX, max_value=DEFAULT_MAX, **_):  # noqa: A002
        super().__init__()
        self.register_buffer("min", torch.tensor(min))
        self.register_buffer("max", torch.tensor(max))
        self.register_buffer("min_value", torch.tensor(min_value))
        self.register_buffer("max_value", torch.tensor(max_value))

    def scale(self, x):
        return ((x - self.min) / (self.max - self.min)) * (
            self.max_value - self.min_value
        ) + self.min_value

    def descale(self, x):
        return ((x - self.min_value) / (self.max_value - self.min_value)) * (
            self.max - self.min
        ) + self.min


class P01P99Scaler(Scaler):
    """Scaling based on the 1st and 99th percentiles.
    It is often useful to set min/max_value to ~0.5 to avoid saturation, or just use MinMaxScaler.
    """

    def __init__(self, *, p01, p99, min_value=-DEFAULT_MAX, max_value=DEFAULT_MAX, **_):
        super().__init__()
        self.register_buffer("p01", torch.tensor(p01))
        self.register_buffer("p99", torch.tensor(p99))
        self.register_buffer("min_value", torch.tensor(min_value))
        self.register_buffer("max_value", torch.tensor(max_value))

    def scale(self, x):
        return ((x - self.p01) / (self.p99 - self.p01)) * (
            self.max_value - self.min_value
        ) + self.min_value

    def descale(self, x):
        return ((x - self.min_value) / (self.max_value - self.min_value)) * (
            self.p99 - self.p01
        ) + self.p01


# class NormalScaler(Scaler):
#     def __init__(self, *, mean, std, target_mean=0.0, target_std=DEFAULT_MAX, **_):
#         super().__init__()
#         self.register_buffer("mean", torch.tensor(mean))
#         self.register_buffer("std", torch.tensor(std))
#         self.register_buffer("target_mean", torch.tensor(target_mean))
#         self.register_buffer("target_std", torch.tensor(target_std))

#     def scale(self, x):
#         return ((x - self.mean) / self.std) * self.target_std + self.target_mean

#     def descale(self, x):
#         return ((x - self.target_mean) / self.target_std) * self.std + self.mean
