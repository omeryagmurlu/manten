from torch import nn


class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self.__dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
