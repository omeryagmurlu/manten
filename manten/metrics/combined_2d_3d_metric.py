import torch

from manten.metrics.utils.base_metric import BaseMetric
from manten.utils.utils_mixins import recursive_copy_mixin_factory
from manten.utils.utils_pytree import with_tree_map


class Combined2D3DMetric(
    recursive_copy_mixin_factory(
        "weight_action_2d",
        "weight_action_3d",
        "weight_action_consistency",
        "metric_action_2d",
        "metric_action_3d",
        "metric_action_consistency",
        "sigmoid_openness_in_action_consistency",
    ),
    BaseMetric,
):
    def __init__(
        self,
        metric_action_2d=None,
        metric_action_3d=None,
        metric_action_consistency=None,
        weight_action_2d=None,
        weight_action_3d=None,
        weight_action_consistency=None,
    ):
        super().__init__()

        self.metric_action_2d = metric_action_2d
        self.metric_action_3d = metric_action_3d
        self.metric_action_consistency = metric_action_consistency

        self.weight_action_2d = weight_action_2d
        self.weight_action_3d = weight_action_3d
        self.weight_action_consistency = weight_action_consistency

    def feed(self, ground, prediction):
        super().feed(ground, prediction)

        self.metric_action_2d.feed(self.ground["2d"], self.prediction["2d"])
        self.metric_action_3d.feed(self.ground["3d"], self.prediction["3d"])

        self.metric_action_consistency.feed(
            self.prediction["2d_for_3d"], self.prediction["3d"]
        )

    def loss(self):
        # only action loss for now
        return self.action_loss()

    def action_loss(self):
        return (
            self.weight_action_2d * self.action_2d_loss()
            + self.weight_action_3d * self.action_3d_loss()
            + self.weight_action_consistency * self.action_consistency_loss()
        )

    def action_2d_loss(self):
        return self.metric_action_2d.loss()

    def action_3d_loss(self):
        return self.metric_action_3d.loss()

    def action_consistency_loss(self):
        return self.metric_action_consistency.loss()

    @with_tree_map(lambda x: x.detach().cpu().numpy())
    def metrics(self):
        return {
            "loss": self.loss(),
            "action_loss": self.action_loss(),
            "action_2d_loss": self.action_2d_loss(),
            "action_3d_loss": self.action_3d_loss(),
            "action_consistency_loss": self.action_consistency_loss(),
            "3d_available_ratio": torch.tensor(
                len(self.ground["3d"]) / len(self.ground["2d"])
            ),
        }
