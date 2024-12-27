from abc import ABC

from manten.agents.utils.base_agent import BaseAgent
from manten.agents.utils.normalization import MinMaxScaler, NoopScaler, Scaler


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
            self.action_scaler = action_scaler(**self.dataset_info["actions_stats"])
        else:
            self.action_scaler = NoopScaler()
