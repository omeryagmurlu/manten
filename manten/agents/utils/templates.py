# pyright: reportAbstractUsage=false, reportIncompatibleMethodOverride=false

from abc import ABC, abstractmethod

import torch

from manten.agents.utils.base_agent import BaseAgent
from manten.utils.utils_decorators import with_partial


class AdaptActionsMixin:
    def adapt_actions_from_ds_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Adapts dataset actions if necessary. Override this method if needed. Noop by default.
        Args:
            actions (torch.Tensor): The dataset actions.
        Returns:
            torch.Tensor: Adapted actions.
        """
        return actions


# see an example agent for usage
class BatchObservationActionAgentTemplate(AdaptActionsMixin, ABC):
    @staticmethod
    @with_partial
    def make_agent(
        template_cls: type["BatchObservationActionAgentTemplate"],
        evaluation_metric_cls,
        evaluation_stats_cls,
    ):
        """Decorator for agents that predict actions based on batches with
        'observations' and 'actions' keys."""

        class BatchObservationActionAgent(template_cls, BaseAgent):
            def __init__(
                self,
                *args,
                evaluation_metric_cls=evaluation_metric_cls,
                evaluation_stats_cls=evaluation_stats_cls,
                **kwargs,
            ):
                super().__init__(*args, **kwargs)

                self.__evaluation_metric_cls = evaluation_metric_cls
                self.__evaluation_stats_cls = evaluation_stats_cls

            @torch.no_grad()
            def predict_actions(self, *, observations):
                """Predicts actions based on observations.
                Args:
                    observations (torch.Tensor): Input observations.
                Returns:
                    torch.Tensor: The predicted actions.
                """
                # This (and pretty much all public facing methods) needs to be routed
                # through the forward method because accelerate has some hooks there to
                # handle the device placement, autocast etc. TrainLoops directly call
                # self("mode", ...) instead of these methods, so this is mostly only
                # concern for inference.
                _metric, pred_actions = self("eval", {"observations": observations})
                return pred_actions

            def train_step(self, batch: dict):
                gt, pred = super().compute_train_gt_and_pred(
                    observations=batch["observations"], actions=batch["actions"]
                )
                self.metric.feed(ground=gt, prediction=pred)
                return self.metric

            @torch.no_grad()
            def eval_step(self, batch: dict, *_, **__):
                observations = batch["observations"]
                pred_actions = super().predict_actions(observations)

                if "actions" in batch:
                    gt_actions = super().adapt_actions_from_ds_actions(batch["actions"])
                    metric = self.__evaluation_metric_cls()
                    metric.feed(ground=gt_actions, prediction=pred_actions)
                else:
                    metric = self.__evaluation_stats_cls()
                    metric.feed(stats=pred_actions)
                return metric, pred_actions

        return BatchObservationActionAgent

    @abstractmethod
    def predict_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """Predicts actions based on observations.
        Args:
            observations (torch.Tensor): Input observations.
        Returns:
            torch.Tensor: The predicted actions.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_train_gt_and_pred(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes ground truth and predictions for loss computation.
        Args:
            observations (torch.Tensor): The input observations.
            actions (torch.Tensor): Dataset actions.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Ground truth and predictions.
        """
        raise NotImplementedError


class BatchStateObservationActionAgentTemplate(AdaptActionsMixin, ABC):
    @staticmethod
    @with_partial
    def make_agent(
        template_cls: type["BatchStateObservationActionAgentTemplate"], *args, **kwargs
    ):
        # ObsActAgent -> BSOAATmplToBOAATmpl -> SObsActTmpl (# just to get default impl -> ObsActTmpl)
        @BatchObservationActionAgentTemplate.make_agent(*args, **kwargs)
        class BSOAATmplToBOAATmpl(template_cls):
            def predict_actions_from_state(self, state_obs):
                return self.predict_actions(observations={"state_obs": state_obs})

            def compute_train_gt_and_pred(self, observations, actions):
                state_obs = observations["state_obs"]
                return super().compute_train_gt_and_pred(state_obs, actions)

            def predict_actions(self, observations):
                state_obs = observations["state_obs"]
                return super().predict_actions(state_obs)

        return BSOAATmplToBOAATmpl

    @abstractmethod
    def predict_actions(self, state_obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_train_gt_and_pred(
        self, state_obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class BatchRGBObservationActionAgentTemplate(AdaptActionsMixin, ABC):
    @staticmethod
    @with_partial
    def make_agent(
        template_cls: type["BatchRGBObservationActionAgentTemplate"], *args, **kwargs
    ):
        @BatchObservationActionAgentTemplate.make_agent(*args, **kwargs)
        class BRGBOAATmplToBOAATmpl(template_cls):
            def compute_train_gt_and_pred(self, observations, actions):
                rgb_obs = observations["rgb_obs"]
                return super().compute_train_gt_and_pred(rgb_obs, actions)

            def predict_actions(self, observations):
                rgb_obs = observations["rgb_obs"]
                return super().predict_actions(rgb_obs)

        return BRGBOAATmplToBOAATmpl

    @abstractmethod
    def predict_actions(self, rgb_obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_train_gt_and_pred(
        self, rgb_obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError