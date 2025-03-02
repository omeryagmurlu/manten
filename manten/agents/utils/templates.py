# pyright: reportAbstractUsage=false, reportIncompatibleMethodOverride=false

from abc import ABC, abstractmethod
from typing import Any

import optree
import torch

from manten.agents.utils.base_agent import BaseAgent
from manten.utils.utils_decorators import with_partial
from manten.utils.utils_pytorch import get_ones_shape_like


def select_keys(dc, *keys):
    return {k: dc[k] for k in keys}


def omit_keys(dc, *keys):
    return {k: v for k, v in dc.items() if k not in keys}


class AdaptActionsMixin:
    def adapt_actions_from_ds_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Adapts dataset actions if necessary. Override this method if needed. Noop by default.
        Args:
            actions (torch.Tensor): The dataset actions.
        Returns:
            torch.Tensor: Adapted actions.
        """
        return actions


class DatasetShapesMixin:
    """
    Mixin for agents that need to know the shapes of the observations and actions.

    The attributes are often implemented in the respective templates, so it may be
    that you don't need to implement them in the agent itself.
    """

    observations_shape: Any
    "The shape of the observations. This attribute is often implemented in the respective templates, so it may be that you don't need to implement it in the agent itself."
    actions_shape: Any
    "The shape of the actions. This attribute is often implemented in the respective templates, so it may be that you don't need to implement it in the agent itself."


class PreTemplateInitMixin:
    """Mixin for agents that need to run some code before the template init.

    __init__ runs the _pre_template_init method (which can be overridden and thus runs before template init)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pre_template_init()

    def _pre_template_init(self):
        """This is usually implemented in templates."""


class AgentActionTemplateMixins(AdaptActionsMixin, DatasetShapesMixin, PreTemplateInitMixin):
    """Combined mixins"""


# see an example agent for usage
class BatchObservationActionAgentTemplate(AgentActionTemplateMixins, ABC):
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

            def _pre_template_init(self):
                self.actions_shape = self.__list_to_size(self.dataset_info["actions_shape"])
                self.observations_shape = self.__list_to_size(
                    self.dataset_info["observations_shape"]
                )
                super()._pre_template_init()

            @staticmethod
            def __list_to_size(cont):
                return optree.tree_map(
                    lambda x: torch.Size(x), cont, is_leaf=lambda x: isinstance(x, list)
                )

            @torch.no_grad()
            def predict_actions(self, *, observations, **kwargs):
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
                _metric, pred_actions = self("eval", {"observations": observations, **kwargs})
                return pred_actions

            def train_step(self, batch: dict):
                gt, pred = super().compute_train_gt_and_pred(**batch)
                self.metric.feed(ground=gt, prediction=pred)
                return self.metric

            @torch.no_grad()
            def eval_step(self, batch: dict, *_, **__):
                batch_wo_actions = omit_keys(batch, "actions")
                pred_actions = super().predict_actions(**batch_wo_actions)

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


class BatchStateObservationActionAgentTemplate(AgentActionTemplateMixins, ABC):
    @staticmethod
    @with_partial
    def make_agent(
        template_cls: type["BatchStateObservationActionAgentTemplate"], *args, **kwargs
    ):
        # ObsActAgent -> BSOAATmplToBOAATmpl -> SObsActTmpl (# just to get default impl -> ObsActTmpl)
        @BatchObservationActionAgentTemplate.make_agent(*args, **kwargs)
        class BSOAATmplToBOAATmpl(template_cls):
            def _pre_template_init(self):
                self.observations_shape = self.observations_shape["state_obs"]
                super()._pre_template_init()

            def compute_train_gt_and_pred(self, observations, actions, **_):
                state_obs = observations["state_obs"]
                return super().compute_train_gt_and_pred(state_obs, actions)

            def predict_actions(self, observations, **_):
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


class BatchRGBObservationActionAgentTemplate(AgentActionTemplateMixins, ABC):
    @staticmethod
    @with_partial
    def make_agent(
        template_cls: type["BatchRGBObservationActionAgentTemplate"], *args, **kwargs
    ):
        @BatchObservationActionAgentTemplate.make_agent(*args, **kwargs)
        class BRGBOAATmplToBOAATmpl(template_cls):
            def _pre_template_init(self):
                self.observations_shape = select_keys(
                    self.observations_shape, "rgb_obs", "state_obs"
                )

            def compute_train_gt_and_pred(self, observations, actions, **_):
                rgb_obs = observations["rgb_obs"]
                state_obs = observations["state_obs"]
                return super().compute_train_gt_and_pred(rgb_obs, state_obs, actions)

            def predict_actions(self, observations, **_):
                rgb_obs = observations["rgb_obs"]
                state_obs = observations["state_obs"]
                return super().predict_actions(rgb_obs, state_obs)

        return BRGBOAATmplToBOAATmpl

    @abstractmethod
    def predict_actions(self, rgb_obs: torch.Tensor, state_obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_train_gt_and_pred(
        self, rgb_obs: torch.Tensor, state_obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class BatchPCDObservationActionAgentTemplate(AgentActionTemplateMixins, ABC):
    @staticmethod
    @with_partial
    def make_agent(
        template_cls: type["BatchPCDObservationActionAgentTemplate"], *args, **kwargs
    ):
        @BatchObservationActionAgentTemplate.make_agent(*args, **kwargs)
        class BPCDOAATmplToBOAATmpl(template_cls):
            def _pre_template_init(self):
                self.observations_shape = select_keys(
                    self.observations_shape, "pcd_obs", "rgb_obs", "pcd_mask", "state_obs"
                )

            def compute_train_gt_and_pred(self, observations, actions, **_):
                pcd_obs = observations["pcd_obs"]
                rgb_obs = observations["rgb_obs"]
                pcd_mask = observations["pcd_mask"]
                state_obs = observations["state_obs"]
                return super().compute_train_gt_and_pred(
                    pcd_obs, rgb_obs, pcd_mask, state_obs, actions
                )

            def predict_actions(self, observations, **_):
                pcd_obs = observations["pcd_obs"]
                rgb_obs = observations["rgb_obs"]
                pcd_mask = observations["pcd_mask"]
                state_obs = observations["state_obs"]
                return super().predict_actions(pcd_obs, rgb_obs, pcd_mask, state_obs)

        return BPCDOAATmplToBOAATmpl

    @abstractmethod
    def predict_actions(
        self,
        pcd_obs: torch.Tensor,
        rgb_obs: torch.Tensor,
        pcd_mask: torch.Tensor,
        state_obs: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_train_gt_and_pred(
        self,
        pcd_obs: torch.Tensor,
        rgb_obs: torch.Tensor,
        pcd_mask: torch.Tensor,
        state_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class BatchPCDOrRGBObservationActionAgentTemplate(AgentActionTemplateMixins, ABC):
    @staticmethod
    @with_partial
    def make_agent(
        template_cls: type["BatchPCDOrRGBObservationActionAgentTemplate"], *args, **kwargs
    ):
        @BatchObservationActionAgentTemplate.make_agent(*args, **kwargs)
        class BPCDOAATmplToBOAATmpl(template_cls):
            def _pre_template_init(self):
                self.observations_shape = select_keys(
                    self.observations_shape, "pcd_obs", "rgb_obs", "pcd_mask", "state_obs"
                )

            def compute_train_gt_and_pred(self, observations, actions, meta):
                observations = self.__mask_observations_by_vision_modality(
                    observations, meta["3d_mask"]
                )

                pcd_obs = observations["pcd_obs"]
                rgb_obs = observations["rgb_obs"]
                pcd_mask = observations["pcd_mask"]
                state_obs = observations["state_obs"]

                B = next(iter(rgb_obs.values())).shape[0]

                return super().compute_train_gt_and_pred(
                    pcd_obs,
                    rgb_obs,
                    pcd_mask,
                    state_obs,
                    actions,
                    meta["3d_mask"].view(-1).expand(B),
                )

            def predict_actions(self, observations, meta):
                # this is used as the modality for the prediction
                observations = self.__mask_observations_by_vision_modality(
                    observations, meta["3d_mask"]
                )

                pcd_obs = observations["pcd_obs"]
                rgb_obs = observations["rgb_obs"]
                pcd_mask = observations["pcd_mask"]
                state_obs = observations["state_obs"]

                B = next(iter(rgb_obs.values())).shape[0]

                return super().predict_actions(
                    pcd_obs, rgb_obs, pcd_mask, state_obs, meta["3d_mask"].view(-1).expand(B)
                )

            def __mask_observations_by_vision_modality(self, observations, keep_mask_3d):
                bs, dtype, device = self.__get_bs_dtype_device(observations)
                if "pcd_obs" not in observations:
                    observations["pcd_obs"] = {
                        cam: torch.zeros(bs, *list(shp), dtype=dtype, device=device)
                        for cam, shp in self.observations_shape["pcd_obs"].items()
                    }
                if "pcd_mask" not in observations:
                    observations["pcd_mask"] = {
                        cam: torch.ones(bs, *list(shp), dtype=dtype, device=device)
                        for cam, shp in self.observations_shape["pcd_mask"].items()
                    }

                observations["pcd_obs"] = {
                    cam: torch.where(
                        keep_mask_3d.view(
                            keep_mask_3d.shape[0], *get_ones_shape_like(val)[1:]
                        ),
                        val,
                        torch.zeros_like(val),
                    )
                    for cam, val in observations["pcd_obs"].items()
                }

                observations["pcd_mask"] = {
                    cam: torch.where(
                        keep_mask_3d.view(
                            keep_mask_3d.shape[0], *get_ones_shape_like(val)[1:]
                        ),
                        val,
                        torch.ones_like(val),
                    )
                    for cam, val in observations["pcd_mask"].items()
                }

                return observations

            @staticmethod
            def __get_bs_dtype_device(observations):
                first_rgb = next(iter(observations["rgb_obs"].values()))
                return first_rgb.shape[0], first_rgb.dtype, first_rgb.device

        return BPCDOAATmplToBOAATmpl

    @abstractmethod
    def predict_actions(
        self,
        pcd_obs: torch.Tensor,
        rgb_obs: torch.Tensor,
        pcd_mask: torch.Tensor,
        state_obs: torch.Tensor,
        keep_mask_3d: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_train_gt_and_pred(
        self,
        pcd_obs: torch.Tensor,
        rgb_obs: torch.Tensor,
        pcd_mask: torch.Tensor,
        state_obs: torch.Tensor,
        actions: torch.Tensor,
        keep_mask_3d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
