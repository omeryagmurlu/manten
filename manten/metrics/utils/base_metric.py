from abc import ABC, abstractmethod

from manten.utils.utils_mixins import (
    recursive_copy_mixin_factory,
    recursive_state_dict_mixin_factory,
)


class BaseMetric(
    recursive_copy_mixin_factory("ground", "prediction"),
    recursive_state_dict_mixin_factory("ground", "prediction"),
    ABC,
):
    def __init__(self):
        self.ground = None
        self.prediction = None

    def feed(self, ground, prediction):
        self.ground = ground
        self.prediction = prediction

    def reset(self):
        self.ground = None
        self.prediction = None

    @abstractmethod
    def loss(self):
        raise NotImplementedError

    @abstractmethod
    def metrics(self) -> dict:
        raise NotImplementedError

    def summary_metrics(self) -> dict:
        """Return a summary of metrics. Useful for use in tqdm post_fix"""
        return self.metrics()

    def visualize(self, *_, **__) -> dict:
        """Return a visualization of the metric. Useful for tensorboard"""
        return


class BaseStats(
    recursive_copy_mixin_factory("stats"),
    recursive_state_dict_mixin_factory("stats"),
    BaseMetric,
    ABC,
):
    def __init__(self):
        super().__init__()
        self.stats = None

    def feed(self, stats):
        self.stats = stats

    def reset(self):
        self.stats = None

    def loss(self):
        raise ValueError("BaseStats does not support loss()")
