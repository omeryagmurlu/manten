import torch.nn.functional as F

from manten.utils.utils_decorators import with_shallow_copy
from manten.utils.utils_pytree import with_tree_map

from .base_metric import BaseMetric


@with_shallow_copy("pos_weight", "ortho6d_weight", "open_weight")
class ThreeDDAMetric(BaseMetric):
    def __init__(self, pos_weight=30.0, ortho6d_weight=10.0, open_weight=1.0):
        super().__init__()

        self.pos_weight = pos_weight
        self.ortho6d_weight = ortho6d_weight
        self.open_weight = open_weight

    def loss(self):
        return self.pos_loss() + self.ortho6d_loss() + self.open_loss()

    @with_tree_map(lambda x: x.detach().cpu().numpy())
    def metrics(self):
        # # pred/gt are (B, L, 7) (3 ee + 4 joint)
        # pred = self.prediction
        # gt = self.ground

        # pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
        # # symmetric quaternion eval
        # quat_l1 = (pred[..., 3:7] - gt[..., 3:7]).abs().sum(-1)
        # quat_l1_ = (pred[..., 3:7] + gt[..., 3:7]).abs().sum(-1)
        # select_mask = (quat_l1 < quat_l1_).float()
        # quat_l1 = select_mask * quat_l1 + (1 - select_mask) * quat_l1_
        # # gripper openness
        # openness = ((pred[..., 7:] >= 0.5) == (gt[..., 7:] > 0.0)).bool()
        # tr = "traj_"

        # return {
        #     tr + "action_mse": F.mse_loss(pred, gt),
        #     tr + "pos_l2": pos_l2.mean(),
        #     tr + "pos_acc_001": (pos_l2 < 0.01).float().mean(),
        #     tr + "rot_l1": quat_l1.mean(),
        #     tr + "rot_acc_0025": (quat_l1 < 0.025).float().mean(),
        #     tr + "gripper": openness.flatten().float().mean(),
        # }

        pred = self.prediction
        gt = self.ground

        ret = {
            "loss_total": self.loss(),
            "loss_pos": self.pos_loss(),
            "loss_ortho6d": self.ortho6d_loss(),
            "loss_open": self.open_loss(),
            "mae_pos": F.l1_loss(pred[..., :3], gt[..., :3]),
            "mae_ortho6d": F.l1_loss(pred[..., 3:9], gt[..., 3:9]),
            "bce_open": F.binary_cross_entropy_with_logits(pred[..., 9:10], gt[..., 9:10]),
        }

        return ret

    @with_tree_map(lambda x: x.detach().cpu().numpy())
    def summary_metrics(self):
        return {
            "loss_pos": self.pos_loss(),
            "loss_ortho6d": self.ortho6d_loss(),
            "loss_open": self.open_loss(),
        }

    def pos_loss(self):
        # why mae? wouldn't mse be better? since it optimizes for the mean
        return self.pos_weight * F.l1_loss(self.prediction[..., :3], self.ground[..., :3])

    def ortho6d_loss(self):  # ortho6d
        return self.ortho6d_weight * F.l1_loss(
            self.prediction[..., 3:9], self.ground[..., 3:9]
        )

    def open_loss(self):
        return self.open_weight * F.binary_cross_entropy_with_logits(
            self.prediction[..., 9:10], self.ground[..., 9:10]
        )
