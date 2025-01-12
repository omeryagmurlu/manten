import numpy as np
import torch.nn.functional as F

from manten.metrics.utils.utils_loss import binary_cross_entropy_with_logits_with_hinge_domain
from manten.utils.utils_pytree import with_tree_map

from .utils.base_metric import BaseMetric, BaseStats


class PosRotGripperStats(BaseStats):
    def metrics(self):
        pos = self.stats[..., :3]
        gripper = self.stats[..., 9:]
        ret = {
            "mean_x": pos[..., 0].mean(),
            "mean_y": pos[..., 1].mean(),
            "mean_z": pos[..., 2].mean(),
            "mean_gripper": gripper.mean(),
        }
        return ret

    def summary_metrics(self):
        metrics = self.metrics()
        return {key: metrics[key] for key in ["mean_x", "mean_y", "mean_z"]}


class PosRotGripperMetric(BaseMetric):
    def loss(self):
        raise ValueError("PosTrajMetric does not support loss()")

    @with_tree_map(lambda x: x.detach().cpu().numpy())
    def metrics(self):
        pred_pos = self.prediction[..., :3]
        gt_pos = self.ground[..., :3]
        pos_l2 = F.mse_loss(pred_pos, gt_pos, reduction="none").sqrt()

        # rot_diff = self.rotation_diff_max_magnitude()

        pred_gripper = self.prediction[..., -1:]
        gt_gripper = self.ground[..., -1:]

        if gt_gripper.min() < -1 or gt_gripper.max() > 1:
            raise ValueError(
                f"Gripper range is not [-1, 1], it is [{gt_gripper.min()}, {gt_gripper.max()}]"
            )

        gripper_bce = binary_cross_entropy_with_logits_with_hinge_domain(
            pred_gripper, gt_gripper, reduction="none"
        )

        ret = {
            "pos_l2": pos_l2.mean(),
            "pos_l2_acc_001": (pos_l2 < 0.01).float().mean(),  # noqa: PLR2004
            "pos_l2_acc_010": (pos_l2 < 0.10).float().mean(),  # noqa: PLR2004
            # "rot_mmag": rot_diff.mean(),
            # "rot_mmag_acc_001": (rot_diff < 0.01).float().mean(),
            # "rot_mmag_acc_010": (rot_diff < 0.10).float().mean(),
            "gripper_bce": gripper_bce.mean(),
            "gripper_bce_acc_001": (gripper_bce < 0.01).float().mean(),  # noqa: PLR2004
            "gripper_bce_acc_010": (gripper_bce < 0.10).float().mean(),  # noqa: PLR2004
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

        pred_pos = self.prediction[:3]
        gt_pos = self.ground[:3]

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

    # @abstractmethod
    # def rotation_diff_max_magnitude(self):
    #     raise NotImplementedError


# class PosRot6DGripperMetric(PosRotGripperMetric):
#     def rotation_diff_max_magnitude(self):
#         pred_rot = self.prediction[..., 3:9]
#         pred_rot = rotation_6d_to_matrix(pred_rot)
#         gt_rot = self.ground[..., 3:9]
#         gt_rot = rotation_6d_to_matrix(gt_rot)

#         diff_matrix = einops.einsum(gt_rot, pred_rot.transpose(-2, -1), "... i j, ... j k -> ... i k")
#         diff_axis_angle = matrix_to_axis_angle(diff_matrix)
#         # return (gt_rot * pred_rot.inv()).magnitude().max(dim=0)


# def quaternion_magnitude(q):
