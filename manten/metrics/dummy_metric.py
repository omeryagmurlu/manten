import numpy as np
import torch.nn.functional as F

from manten.utils.utils_pytree import with_tree_map

from .base_metric import BaseMetric, BaseStats


class MSELossDummyMetric(BaseMetric):
    def loss(self):
        return F.mse_loss(self.prediction, self.ground)

    @with_tree_map(lambda x: x.detach().cpu().numpy())
    def metrics(self):
        return {"loss": self.loss()}


class PosTrajStats(BaseStats):
    def metrics(self):
        pos = self.stats
        ret = {
            "mean_x": pos[..., 0].mean(),
            "mean_y": pos[..., 1].mean(),
            "mean_z": pos[..., 2].mean(),
        }
        return ret

    def summary_metrics(self):
        metrics = self.metrics()
        return {key: metrics[key] for key in ["mean_x", "mean_y", "mean_z"]}


class PosTrajMetric(BaseMetric):
    def loss(self):
        raise ValueError("PosTrajMetric does not support loss()")

    @with_tree_map(lambda x: x.detach().cpu().numpy())
    def metrics(self):
        pred_pos = self.prediction
        gt_pos = self.ground

        pos_l2 = ((pred_pos - gt_pos) ** 2).sum(-1).sqrt()

        ret = {
            "pos_l2": pos_l2.mean(),
            "pos_acc_001": (pos_l2 < 0.01).float().mean(),  # noqa: PLR2004
        }

        return ret

    def summary_metrics(self):
        metrics = self.metrics()
        return {key: metrics[key] for key in ["pos_l2"]}

    def visualize(self, sort_key=None, reduction_hint=None):
        """
        Returns an image showing both prediction and ground truth Postraj positions
        for the first sample in the batch in 3D.
        """
        from manten.utils.utils_visualization import visualize_2_pos_traj

        pred_pos = self.prediction
        gt_pos = self.ground

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
