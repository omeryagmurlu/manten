import torch.nn.functional as F

from manten.metrics.utils.base_metric import BaseMetric
from manten.utils.utils_pytree import with_tree_map


class MSELossMetric(BaseMetric):
    """
    Simple MSE Loss
    """

    def __init__(self):
        super().__init__()

    def loss(self):
        return F.mse_loss(self.prediction, self.ground)

    @with_tree_map(lambda x: x.detach().cpu().numpy())
    def metrics(self):
        return {"loss": self.loss()}
