from manten.agents.metrics.base_metric import BaseMetric, BaseStats
import torch.nn.functional as F


class TrajectoryStats(BaseStats):
    def metrics(self):
        pos, quat, openn = self.stats
        ret = dict(
            mean_x=pos[..., 0].mean().item(),
            mean_y=pos[..., 1].mean().item(),
            mean_z=pos[..., 2].mean().item(),
            mean_quat1=quat[..., 0].mean().item(),
            mean_quat2=quat[..., 1].mean().item(),
            mean_quat3=quat[..., 2].mean().item(),
            mean_quat4=quat[..., 3].mean().item(),
            mean_open=openn.mean().item(),
        )
        return ret

    def summary_metrics(self):
        metrics = self.metrics()
        return {key: metrics[key] for key in ["mean_x", "mean_y", "mean_z"]}


class TrajectoryMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.pred_stats = TrajectoryStats()

    def feed(self, ground, prediction):
        super().feed(ground, prediction)
        self.pred_stats.feed(prediction)

    def reset(self):
        super().reset()
        self.pred_stats.reset()

    def loss(self):
        raise ValueError("TrajectoryMetric does not support loss()")

    def metrics(self):
        pred_pos, pred_quat, pred_open = self.prediction
        gt_pos, gt_quat, gt_open = self.ground

        ret = dict(
            mae_x=F.l1_loss(pred_pos[..., 0], gt_pos[..., 0]).item(),
            mae_y=F.l1_loss(pred_pos[..., 1], gt_pos[..., 1]).item(),
            mae_z=F.l1_loss(pred_pos[..., 2], gt_pos[..., 2]).item(),
            mae_pos=F.l1_loss(pred_pos[..., :3], gt_pos[..., :3]).item(),
            mae_quat=F.l1_loss(pred_quat[..., :4], gt_quat[..., :4]).item(),
            bce_open=F.binary_cross_entropy_with_logits(
                pred_open[..., :], gt_open[..., :]
            ).item(),
        )

        ret.update({f"pred_{key}": value for key, value in self.pred_stats.metrics().items()})

        return ret

    def summary_metrics(self):
        metrics = self.metrics()
        return {key: metrics[key] for key in ["mae_pos", "bce_open"]}
