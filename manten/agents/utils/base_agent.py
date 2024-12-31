from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from manten.metrics.utils.base_metric import BaseMetric
from manten.networks.utils.mixins import ModuleAttrMixin
from manten.utils.utils_hydra import to_object_graceful


class BaseAgent(ModuleAttrMixin, nn.Module, ABC):
    """Base class for agents.
    Attributes:
        metric : BaseMetric
            The metric used for training/loss
        dataset_info : dict | None
            Information about the dataset, if available
    Methods:
        reset()
            Resets the metric
        forward(agent_mode : str, *args, **kwargs)
            Routes calls based on the agent mode
        train_step(batch : dict) -> BaseMetric
            Abstract method for the training step
        eval_step(batch : dict, compare_gt : bool) -> tuple[BaseMetric, torch.Tensor]
            Abstract method for the evaluation step
        validate_step(*a, **k) -> BaseMetric
            Default implementation calls train_step without gradients
    """

    def __init__(self, metric: BaseMetric, dataset_info: Any = None):
        super().__init__()
        self.metric = metric
        self.dataset_info = to_object_graceful(dataset_info)

    def reset(self):
        """Resets the metric."""
        self.metric.reset()

    def forward(self, agent_mode: str, *args, **kwargs):
        """Routes calls based on the agent mode.
        Args:
            agent_mode (str): The mode ('train', 'validate', or 'eval').
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            Any: The result from the corresponding step method.
        Raises:
            ValueError: If an unknown agent mode is provided.
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
                raise ValueError(f"Unknown agent mode: {agent_mode}")

    @abstractmethod
    def train_step(self, batch: dict) -> BaseMetric:
        """Training step.
        Args:
            batch (dict): The batch of training data.
        Returns:
            BaseMetric: The metric after the training step.
        """
        raise NotImplementedError

    @abstractmethod
    def eval_step(
        self, batch: dict, *, compare_gt: bool = False
    ) -> tuple[BaseMetric, torch.Tensor]:
        """Evaluation step.
        Args:
            batch (dict): The evaluation data.
            compare_gt (bool, optional): Compare with ground truth.
        Returns:
            tuple[BaseMetric, torch.Tensor]: The metric and output tensor.
        """
        raise NotImplementedError

    @torch.no_grad()
    def validate_step(self, *a, **k) -> BaseMetric:
        """Validation step. You may override this method. Default implementation calls train_step without gradients."""
        return self.train_step(*a, **k)
