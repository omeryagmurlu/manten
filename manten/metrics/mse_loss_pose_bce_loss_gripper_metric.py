import torch
import torch.nn.functional as F

from manten.metrics.utils.base_metric import BaseMetric
from manten.metrics.utils.utils_loss import (
    binary_cross_entropy_with_logits_with_hinge_domain,
)
from manten.utils.utils_mixins import recursive_copy_mixin_factory
from manten.utils.utils_pytree import with_tree_map


class MSELossPoseBCEWithLogitsLossGripperMetric(
    recursive_copy_mixin_factory("pos_weight", "rot_weight", "gripper_weight"), BaseMetric
):
    """
    This metric is a combination of MSE loss for position and rotation and BCE loss for gripper.
    Gripper is expected to be -1 or 1.

    Important: This metric is not symmetric, i.e. it is not guaranteed that loss(a, b) == loss(b, a).
    Use a non-with-logits version of the gripper loss if you need a symmetric metric.
    """

    def __init__(self, pos_weight=None, rot_weight=None, gripper_weight=None):
        super().__init__()

        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.gripper_weight = gripper_weight

    def loss(self):
        return (
            self.pos_loss() * self.pos_weight
            + self.rot_loss() * self.rot_weight
            + self.gripper_loss() * self.gripper_weight
        )

    def pos_loss(self):
        return F.mse_loss(self.prediction[..., :3], self.ground[..., :3])

    def rot_loss(self):
        # infer rot dim from shape, we'll use mse anyways so concrete type doesn't matter
        rot_dim_end = self.prediction.shape[-1] - 1
        return F.mse_loss(
            self.prediction[..., 3:rot_dim_end], self.ground[..., 3:rot_dim_end]
        )

    def gripper_loss(self):
        """requires the gripper to be -1 or 1"""
        return binary_cross_entropy_with_logits_with_hinge_domain(
            self.prediction[..., -1:], self.ground[..., -1:]
        )

    @with_tree_map(lambda x: x.detach().cpu().numpy())
    def metrics(self):
        return {"loss": self.loss()}


# required for example when comparing two different predictions
class MSELossPoseBCEWithLogitsLossGripperSymmetricMetric(
    MSELossPoseBCEWithLogitsLossGripperMetric
):
    """
    Symmetric version of MSELossPoseBCEWithLogitsLossGripperMetric.
    """

    def gripper_loss(self):
        """requires the gripper to be -1 or 1"""
        target = (self.ground[..., -1:] + 1) / 2
        inp = (self.prediction[..., -1:] + 1) / 2
        target = torch.sigmoid(target)
        inp = torch.sigmoid(inp)
        return F.binary_cross_entropy(inp, target)
