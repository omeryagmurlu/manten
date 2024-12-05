from manten.agents.metrics.base_metric import BaseMetric
import torch.nn.functional as F


class ThreeDDAMetric(BaseMetric):
    def __init__(self, pos_weight=30.0, ortho6d_weight=10.0, open_weight=1.0):
        super().__init__()

        self.pos_weight = pos_weight
        self.ortho6d_weight = ortho6d_weight
        self.open_weight = open_weight

    def loss(self):
        return self.pos_loss() + self.ortho6d_loss() + self.open_loss()

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

        ret = dict(
            loss_total=self.loss().item(),
            loss_pos=self.pos_loss().item(),
            loss_ortho6d=self.ortho6d_loss().item(),
            loss_open=self.open_loss().item(),
            mae_x=F.l1_loss(pred[..., 0], gt[..., 0]).item(),
            mae_y=F.l1_loss(pred[..., 1], gt[..., 1]).item(),
            mae_z=F.l1_loss(pred[..., 2], gt[..., 2]).item(),
            mae_pos=F.l1_loss(pred[..., :3], gt[..., :3]).item(),
            mae_ortho6d=F.l1_loss(pred[..., 3:9], gt[..., 3:9]).item(),
            bce_open=F.binary_cross_entropy_with_logits(
                pred[..., 9:10], gt[..., 9:10]
            ).item(),
        )

        return ret

    def summary_metrics(self):
        return dict(
            loss_pos=self.pos_loss().item(),
            loss_ortho6d=self.ortho6d_loss().item(),
            loss_open=self.open_loss().item(),
        )

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
