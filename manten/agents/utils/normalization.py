from abc import ABC, abstractmethod

import torch
from torch import nn

DEFAULT_MAX = 1.0  # with tolerance


def fit_single(single_data):
    import numpy as np

    stats = {
        "p01": np.percentile(single_data, 1, axis=0),
        "p99": np.percentile(single_data, 99, axis=0),
        "min": np.min(single_data, axis=0),
        "max": np.max(single_data, axis=0),
        "mean": np.mean(single_data, axis=0),
        "std": np.std(single_data, axis=0),
    }

    return {k: v.tolist() for k, v in stats.items()}


def fit_consecutive(data, prev_fit_data=None):
    """Consecutively calculate statistics for batches from a stream of data with each invocation.
    Previous results should be passed as prev_fit_data to continue the calculation.

    for now only works with min/max
    """
    import numpy as np

    data_for_min = np.ma.filled(data, fill_value=float("inf"))
    data_for_max = np.ma.filled(data, fill_value=float("-inf"))

    if prev_fit_data is None:
        prev_fit_data = (
            {
                "min": np.min(data_for_min, axis=0),
                "max": np.max(data_for_max, axis=0),
            },
            len(data),
        )
        return prev_fit_data

    prev_stats, prev_len = prev_fit_data
    new_len = prev_len + len(data)
    new_stats = {
        "min": np.minimum(prev_stats["min"], np.min(data_for_min, axis=0)),
        "max": np.maximum(prev_stats["max"], np.max(data_for_max, axis=0)),
    }
    return new_stats, new_len


def auto_reshape_to_end(dim_size):
    def decorator(fn):
        def wrapper(self, x):
            dim = (torch.tensor(x.shape) == dim_size).nonzero()[-1].item()
            dim_count = len(x.shape)

            if dim == dim_count - 1:
                return fn(self, x)

            x = x.transpose(dim, dim_count - 1)
            x = fn(self, x)
            return x.transpose(dim, dim_count - 1)

        return wrapper

    return decorator


class Scaler(nn.Module, ABC):
    def __init__(self, *, slices=None, **_):
        super().__init__()

        self.slices = slices

    def scale(self, x):
        if self.slices is None:
            return self._scale(x)

        return self.sliced_op(x, self._scale)

    def descale(self, x):
        if self.slices is None:
            return self._descale(x)

        return self.sliced_op(x, self._descale)

    def sliced_op(self, x, op):
        out_x = x.clone()
        op_x = op(x)
        for s in self.slices:
            out_x[..., s] = op_x[..., s]
        return out_x

    @abstractmethod
    def _scale(self, x):
        raise NotImplementedError

    @abstractmethod
    def _descale(self, x):
        raise NotImplementedError


class NoopScaler(Scaler):
    def __init__(self, **_):
        super().__init__()

    def _scale(self, x):
        return x

    def _descale(self, x):
        return x


class MinMaxScaler(Scaler):
    """Min-max scaling."""

    def __init__(self, *, min, max, min_value=-DEFAULT_MAX, max_value=DEFAULT_MAX, **_):  # noqa: A002
        super().__init__()
        self.register_buffer("min", torch.tensor(min))
        self.register_buffer("max", torch.tensor(max))
        self.register_buffer("min_value", torch.tensor(min_value))
        self.register_buffer("max_value", torch.tensor(max_value))

    def _scale(self, x):
        return ((x - self.min) / (self.max - self.min)) * (
            self.max_value - self.min_value
        ) + self.min_value

    def _descale(self, x):
        return ((x - self.min_value) / (self.max_value - self.min_value)) * (
            self.max - self.min
        ) + self.min


class T3DMinMaxScaler(Scaler):
    """3D min-max scaling.
    The scaling is done independently for each dimension.
    """

    def __init__(
        self,
        *,
        min,  # noqa: A002
        max,  # noqa: A002
        min_value=-DEFAULT_MAX,
        max_value=DEFAULT_MAX,
        preserve_aspect_ratio=True,
        **_,
    ):
        super().__init__()
        self.register_buffer("min", torch.tensor(min))
        self.register_buffer("max", torch.tensor(max))
        self.register_buffer("min_value", torch.tensor(min_value))
        self.register_buffer("max_value", torch.tensor(max_value))
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def __get_scale_and_center(self):
        ranges = self.max - self.min
        if self.preserve_aspect_ratio:
            scale = (self.max_value - self.min_value) / torch.max(ranges)
        else:
            scale = (self.max_value - self.min_value) / ranges
        center = (self.max + self.min) / 2

        return scale, center

    @auto_reshape_to_end(3)
    def _scale(self, x):
        scale, center = self.__get_scale_and_center()

        return (x - center) * scale + (self.max_value + self.min_value) / 2

    @auto_reshape_to_end(3)
    def _descale(self, x):
        scale, center = self.__get_scale_and_center()

        return (x - (self.max_value + self.min_value) / 2) * scale + center

    @auto_reshape_to_end(3)
    def scale_without_translation(self, x):
        scale, _ = self.__get_scale_and_center()

        return x * scale

    @auto_reshape_to_end(3)
    def descale_without_translation(self, x):
        scale, _ = self.__get_scale_and_center()

        return x / scale


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

    def _scale(self, x):
        return ((x - self.p01) / (self.p99 - self.p01)) * (
            self.max_value - self.min_value
        ) + self.min_value

    def _descale(self, x):
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

#     def _scale(self, x):
#         return ((x - self.mean) / self.std) * self.target_std + self.target_mean

#     def _descale(self, x):
#         return ((x - self.target_mean) / self.target_std) * self.std + self.mean
