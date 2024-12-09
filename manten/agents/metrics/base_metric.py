from abc import ABC, abstractmethod


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

    def copy(self):
        cpy = self.__class__()
        cpy.ground = self.ground
        cpy.prediction = self.prediction
        return cpy

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


class BaseStats(BaseMetric, ABC):
    def __init__(self):
        self.stats = None

    def feed(self, stats):
        self.stats = stats

    def reset(self):
        self.stats = None

    def copy(self):
        cpy = self.__class__()
        cpy.stats = self.stats
        return cpy

    def loss(self):
        raise ValueError("BaseStats does not support loss()")
