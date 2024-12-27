import numpy as np
import torch
import torch.nn.functional as F

from manten.utils.utils_pytree import with_tree_map

from .utils.base_metric import BaseMetric, BaseStats


class TrajectoryStats(BaseStats):
    def metrics(self):
        pos, quat, openn = self.stats
        ret = {
            "mean_x": pos[..., 0].mean(),
            "mean_y": pos[..., 1].mean(),
            "mean_z": pos[..., 2].mean(),
            "mean_quat1": quat[..., 0].mean(),
            "mean_quat2": quat[..., 1].mean(),
            "mean_quat3": quat[..., 2].mean(),
            "mean_quat4": quat[..., 3].mean(),
            "mean_open": openn.mean(),
        }
        return ret

    def summary_metrics(self):
        metrics = self.metrics()
        return {key: metrics[key] for key in ["mean_x", "mean_y", "mean_z"]}


# @with_state_dict("pred_stats")
# @with_shallow_copy("pred_stats")
class TrajectoryMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        # self.pred_stats = TrajectoryStats()

    def feed(self, ground, prediction):
        super().feed(ground, prediction)
        # self.pred_stats.feed(prediction)

    def reset(self):
        super().reset()
        # self.pred_stats.reset()

    def loss(self):
        raise ValueError("TrajectoryMetric does not support loss()")

    @with_tree_map(lambda x: x.detach().cpu().numpy())
    def metrics(self):
        pred_pos, pred_quat, pred_open = self.prediction
        gt_pos, gt_quat, gt_open = self.ground

        pos_l2 = ((pred_pos - gt_pos) ** 2).sum(-1).sqrt()
        # symmetric quaternion eval
        quat_l1 = (pred_quat - gt_quat).abs().sum(-1)
        quat_l1_ = (pred_quat + gt_quat).abs().sum(-1)
        select_mask = (quat_l1 < quat_l1_).float()
        quat_l1 = select_mask * quat_l1 + (1 - select_mask) * quat_l1_
        # gripper openness
        openness = ((pred_open >= 0.5) == (gt_open > 0.0)).bool()  # noqa: PLR2004

        ret = {
            "mse_traj": F.mse_loss(
                torch.cat([pred_pos, pred_quat, pred_open], dim=-1),
                torch.cat([gt_pos, gt_quat, gt_open], dim=-1),
            ),
            "pos_l2": pos_l2.mean(),
            "pos_acc_001": (pos_l2 < 0.01).float().mean(),  # noqa: PLR2004
            "rot_l1": quat_l1.mean(),
            "rot_acc_0025": (quat_l1 < 0.025).float().mean(),  # noqa: PLR2004
            "gripper": openness.flatten().float().mean(),
        }

        # ret.update({f"pred_{key}": value for key, value in self.pred_stats.metrics().items()})

        return ret

    def summary_metrics(self):
        metrics = self.metrics()
        return {key: metrics[key] for key in ["mse_traj", "pos_l2"]}

    def visualize(self, sort_key=None, reduction_hint=None):
        """
        Returns an image showing both prediction and ground truth trajectory positions
        for the first sample in the batch in 3D.
        """
        from manten.utils.utils_visualization import visualize_2_pos_traj

        pred_pos, pred_quat, pred_open = self.prediction
        gt_pos, gt_quat, gt_open = self.ground

        if sort_key is None:
            samples = [(0, "batch_first/vis_pos")]
        else:
            samples = []
            values = self.metrics()[sort_key]
            if reduction_hint == "min" or reduction_hint is None:
                index = np.argmin(values)
                samples.append((index, f"batch_min:{sort_key}/vis_pos"))
            if reduction_hint == "max" or reduction_hint is None:
                index = np.argmax(values)
                samples.append((index, f"batch_max:{sort_key}/vis_pos"))

        retval = {
            key_name: visualize_2_pos_traj(
                pred_pos[sample_idx].cpu().numpy(),
                gt_pos[sample_idx].cpu().numpy(),
            )
            for sample_idx, key_name in samples
        }

        return retval
