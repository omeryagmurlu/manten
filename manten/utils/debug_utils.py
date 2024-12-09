import logging

logger = logging.getLogger(__name__)


class DebugUtils:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key) and callable(getattr(self, key)):
                getattr(self, key)(value)

    @staticmethod
    def monkeypatch_tensor_shape(*_args, **_kwargs):
        import torch

        logger.info("Monkeypatching torch.Tensor.__repr__ to include shape")

        def custom_repr(self):
            return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)} {{Tensor:{tuple(self.shape)}}}"

        original_repr = torch.Tensor.__repr__
        torch.Tensor.__repr__ = custom_repr
