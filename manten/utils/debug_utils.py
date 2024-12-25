import logging
import multiprocessing
import os
from time import time

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class DebugUtils:
    @staticmethod
    def log(*args, **kwargs):
        if multiprocessing.parent_process() is None:
            logger.info(*args, **kwargs)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key) and callable(getattr(self, key)):
                val = (
                    value
                    if not OmegaConf.is_config(value)
                    else OmegaConf.to_container(value, resolve=True)
                )
                getattr(self, key)(val)

    @staticmethod
    def env_set(env_vars):
        for key, value in env_vars.items():
            old_value = os.environ.get(key)
            DebugUtils.log("setting env var %s from %s to %s", key, old_value, value)
            os.environ[key] = value

        DebugUtils.log("set env vars:")
        for key in env_vars:
            DebugUtils.log("%s: %s", key, os.environ.get(key))

    @staticmethod
    def monkeypatch_tensor_shape(*_args, **_kwargs):
        import torch

        DebugUtils.log("Monkeypatching torch.Tensor.__repr__ to include shape")

        def custom_repr(self):
            return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)} {{Tensor:{tuple(self.shape)}}}"

        original_repr = torch.Tensor.__repr__
        torch.Tensor.__repr__ = custom_repr


class TrainTimer:
    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.end_time = time()

    def before_forward(self):
        self.data_load_duration = (time() - self.end_time) * 1e3
        self._sync()
        self.pre_forward_time = time()

    def after_forward(self):
        self._sync()
        self.post_forward_time = time()

    def after_backward(self):
        self._sync()
        self.post_backward_time = time()

    def before_step_end(self):
        self._sync()
        forward_duration = (self.post_forward_time - self.pre_forward_time) * 1e3
        backward_duration = (self.post_backward_time - self.post_forward_time) * 1e3
        logger.info(
            "forward time: %.2fms, backward time: %.2fms, data load time: %.2fms",
            forward_duration,
            backward_duration,
            self.data_load_duration,
        )

        self.end_time = time()

    def _sync(self):
        import torch

        self.accelerator.wait_for_everyone()
        torch.cuda.synchronize()
