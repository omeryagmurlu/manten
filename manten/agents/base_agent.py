from abc import ABC, abstractmethod

import torch
from torch import nn

from manten.metrics.base_metric import BaseMetric


class BaseAgent(nn.Module, ABC):
    """
    Base class for all training agents
    """

    def __init__(self, metric: BaseMetric):
        super().__init__()
        self.metric = metric

    def reset(self):
        self.metric.reset()

    def forward(self, agent_mode: str, *args, **kwargs):
        """
        forward method
        DDP adds some hooks to the forward method, so route the calls through it
        """
        self.reset()
        match agent_mode:
            case "train":
                return self.train_step(*args, **kwargs)
            case "validate":
                return self.validate_step(*args, **kwargs)
            case "eval":
                return self.eval_step(*args, **kwargs)
            case _:
                raise NotImplementedError

    @abstractmethod
    def train_step(self, batch: dict) -> BaseMetric:
        raise NotImplementedError

    @abstractmethod
    def validate_step(self, batch: dict) -> BaseMetric:
        raise NotImplementedError

    @abstractmethod
    def eval_step(
        self, batch: dict, *, compare_gt: bool = False
    ) -> tuple[BaseMetric, torch.Tensor]:
        raise NotImplementedError
