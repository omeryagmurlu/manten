import torch
import torch.nn.functional as F

from manten.utils.utils_pytree import with_tree_map
from manten.utils.utils_visualization import visualize_image

from .utils.base_metric import BaseMetric, BaseStats


class MNISTMetric(BaseMetric):
    def loss(self):
        return F.mse_loss(self.prediction, self.ground)

    @with_tree_map(lambda x: x.detach().cpu().numpy())
    def metrics(self):
        return {"loss": self.loss()}


class MNISTImageStats(BaseStats):
    def metrics(self):
        return {"h": torch.tensor(self.stats.shape[2], dtype=torch.float32)}

    def visualize(self, *_, **__) -> dict:
        return {"image": visualize_image(self.stats)}


class MNISTConditionalImageStats(BaseStats):
    def metrics(self):
        images, labels = self.stats
        return {"h": torch.tensor(images.shape[2], dtype=torch.float32)}

    def visualize(self, *_, **__) -> dict:
        images, labels = self.stats
        return {"image": visualize_image(images, labels)}
