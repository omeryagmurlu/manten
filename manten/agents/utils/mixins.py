from abc import ABC

from manten.agents.utils.base_agent import BaseAgent
from manten.agents.utils.normalization import (
    MinMaxScaler,
    NoopScaler,
    Scaler,
    T3DMinMaxScaler,
)


class DatasetActionScalerMixin(BaseAgent, ABC):
    """Mixin for agents that use action scalers built from dataset_info.
    Attributes:
        action_scaler : Scaler
            The action scaler used for normalizing or denormalizing actions
    """

    def __init__(self, *args, action_scaler: type[Scaler] | None = MinMaxScaler, **kwargs):
        super().__init__(*args, **kwargs)
        self.__setup_scaler_from_dataset_info(action_scaler)

    def __setup_scaler_from_dataset_info(self, action_scaler):
        assert self.dataset_info is not None, "Dataset info is required for action scaling"
        assert (
            self.dataset_info["actions_stats"] is not None
        ), "Action stats are required for action scaling"

        if self.dataset_info is not None and self.dataset_info["actions_stats"] is not None:
            kwargs = {**self.dataset_info["actions_stats"]["stats"]}
            if (
                "slice" in self.dataset_info["actions_stats"]
                and self.dataset_info["actions_stats"]["slice"]
            ):
                s_begin, s_end = self.dataset_info["actions_stats"]["slice"]
                kwargs["slice"] = slice(s_begin, s_end)
            self.action_scaler = action_scaler(**kwargs)
        else:
            self.action_scaler = NoopScaler()


class DatasetPCDScalerMixin(BaseAgent, ABC):
    """Mixin for agents that use PCD scalers built from dataset_info.
    Attributes:
        pcd_scaler : Scaler
            The PCD scaler used for normalizing or denormalizing PCD
    """

    def __init__(self, *args, pcd_scaler: type[Scaler] | None = T3DMinMaxScaler, **kwargs):
        super().__init__(*args, **kwargs)
        self.__setup_scaler_from_dataset_info(pcd_scaler)

    def __setup_scaler_from_dataset_info(self, pcd_scaler):
        assert self.dataset_info is not None, "Dataset info is required for PCD scaling"
        assert (
            self.dataset_info["pcd_stats"] is not None
        ), "PCD stats are required for PCD scaling"

        if self.dataset_info is not None and self.dataset_info["pcd_stats"] is not None:
            kwargs = {**self.dataset_info["pcd_stats"]}
            self.pcd_scaler = pcd_scaler(**kwargs)
        else:
            self.pcd_scaler = NoopScaler()
