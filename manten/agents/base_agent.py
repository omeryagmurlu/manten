from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch import nn

from manten.agents.metrics.base_metric import BaseMetric


class AgentMode(Enum):
    TRAIN = "train"
    VALIDATE = "validate"
    EVAL = "eval"


class BaseAgent(nn.Module, ABC):
    """
    Base class for all training agents
    """

    def __init__(self, metric: BaseMetric):
        super().__init__()
        self.metric = metric

    def reset(self):
        self.metric.reset()

    def forward(self, agent_mode: AgentMode, *args, **kwargs):
        """
        forward method
        DDP adds some hooks to the forward method, so route the calls through it
        """
        self.reset()
        match agent_mode:
            case AgentMode.TRAIN:
                return self.train_step(*args, **kwargs)
            case AgentMode.VALIDATE:
                return self.validate_step(*args, **kwargs)
            case AgentMode.EVAL:
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
