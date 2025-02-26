import torch
from torch import nn


class ModuleAttrMixin(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.__dummy_variable = nn.Parameter(
            requires_grad=False, data=torch.tensor(0.0, device=device)
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
