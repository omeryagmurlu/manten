from manten.agents.metrics.base_metric import BaseMetric
import torch.nn.functional as F


class ThreeDDAMetric(BaseMetric):
    def __init__(
        self, translation_weight=30.0, rotation_weight=10.0, openness_weight=1.0
    ):
        super().__init__()

        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.openness_weight = openness_weight

    def loss(self):
        return (
            self.translation_weight * self.translation_loss()
            + self.rotation_weight * self.rotation_loss()
            + self.openness_weight * self.openness_loss()
        )

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
        # # gripper openess
        # openess = ((pred[..., 7:] >= 0.5) == (gt[..., 7:] > 0.0)).bool()
        # tr = "traj_"

        # return {
        #     tr + "action_mse": F.mse_loss(pred, gt),
        #     tr + "pos_l2": pos_l2.mean(),
        #     tr + "pos_acc_001": (pos_l2 < 0.01).float().mean(),
        #     tr + "rot_l1": quat_l1.mean(),
        #     tr + "rot_acc_0025": (quat_l1 < 0.025).float().mean(),
        #     tr + "gripper": openess.flatten().float().mean(),
        # }

        pred = self.prediction
        gt = self.ground

        ret = dict(
            loss_translation=self.translation_loss().item(),
            loss_rotation=self.rotation_loss().item(),
            loss_openness=self.openness_loss().item(),
            traj_x_mae=F.l1_loss(pred[..., 0], gt[..., 0]).item(),
            traj_y_mae=F.l1_loss(pred[..., 1], gt[..., 1]).item(),
            traj_z_mae=F.l1_loss(pred[..., 2], gt[..., 2]).item(),
            traj_pos_mae=F.l1_loss(pred[..., :3], gt[..., :3]).item(),
            traj_openness_mae=F.l1_loss(pred[..., 9:], gt[..., 9:]).item(),
        )

        return ret

    def translation_loss(self):
        # why mae? wouldn't mse be better? since it optimizes for the mean
        return F.l1_loss(self.prediction[..., :3], self.ground[..., :3])

    def rotation_loss(self):  # ortho6d
        return F.l1_loss(self.prediction[..., 3:9], self.ground[..., 3:9])

    def openness_loss(self):
        return F.binary_cross_entropy_with_logits(
            self.prediction[..., 9:], self.ground[..., 9:]
        )
