from abc import ABC, abstractmethod
import torch.nn as nn

from manten.agents.metrics.base_metric import BaseMetric


class BaseAgent(nn.Module, ABC):
    """
    Base class for all training agents
    """

    def __init__(self, metric: BaseMetric):
        super().__init__()
        self.metric = metric

    def reset(self):
        self.metric.reset()

    @abstractmethod
    def train_step(self, batch: dict) -> BaseMetric:
        raise NotImplementedError

    @abstractmethod
    def validate_step(self, batch: dict) -> BaseMetric:
        raise NotImplementedError

    @abstractmethod
    def test_step(self, batch: dict) -> BaseMetric:
        raise NotImplementedError
