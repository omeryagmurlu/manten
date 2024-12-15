from abc import ABC, abstractmethod

from manten.utils.utils_decorators import with_shallow_copy, with_state_dict


@with_state_dict("ground", "prediction")
@with_shallow_copy("ground", "prediction")
class BaseMetric(ABC):
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

    def visualize(self, **_) -> dict:
        """Return a visualization of the metric. Useful for tensorboard"""
        return


@with_state_dict("stats")
@with_shallow_copy("stats")
class BaseStats(BaseMetric, ABC):
    def __init__(self):
        self.stats = None

    def feed(self, stats):
        self.stats = stats

    def reset(self):
        self.stats = None

    def loss(self):
        raise ValueError("BaseStats does not support loss()")
