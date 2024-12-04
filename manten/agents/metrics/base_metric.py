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
