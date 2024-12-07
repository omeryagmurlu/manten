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

    @abstractmethod
    def loss(self):
        raise NotImplementedError

    @abstractmethod
    def metrics(self):
        raise NotImplementedError

    def summary_metrics(self):
        """Return a summary of metrics. Useful for use in tqdm post_fix"""
        return self.metrics()

    def visualize(self):
        """Return a visualization of the metric. Useful for tensorboard"""
        return


class BaseStats(BaseMetric, ABC):
    def __init__(self):
        self.stats = None

    def feed(self, stats):
        self.stats = stats

    def reset(self):
        self.stats = None

    def loss(self):
        raise ValueError("BaseStats does not support loss()")
