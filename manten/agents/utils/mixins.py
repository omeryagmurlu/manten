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

    def __init__(
        self,
        *args,
        action_scaler: type[Scaler] | None = MinMaxScaler,
        action_scaler_scale_pos: bool | None = None,
        action_scaler_scale_rot: bool | None = None,
        action_scaler_scale_gripper: bool | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.__scale_pos = (
            action_scaler_scale_pos if action_scaler_scale_pos is not None else True
        )
        self.__scale_rot = (
            action_scaler_scale_rot if action_scaler_scale_rot is not None else False
        )
        self.__scale_gripper = (
            action_scaler_scale_gripper if action_scaler_scale_gripper is not None else False
        )

        self.__setup_scaler_from_dataset_info(action_scaler)

    def __setup_scaler_from_dataset_info(self, action_scaler):
        assert self.dataset_info is not None, "Dataset info is required for action scaling"
        assert (
            self.dataset_info["actions_stats"] is not None
        ), "Action stats are required for action scaling"

        if self.dataset_info is not None and self.dataset_info["actions_stats"] is not None:
            kwargs = {**self.dataset_info["actions_stats"]}

            if not self.__scale_pos or not self.__scale_rot or not self.__scale_gripper:
                # default behavior is to scale all
                # selective scaling, we need to adjust the slices
                kwargs["slices"] = []
                if self.__scale_pos:
                    kwargs["slices"].append(slice(0, 3))
                if self.__scale_rot:
                    kwargs["slices"].append(slice(3, -1))
                if self.__scale_gripper:
                    kwargs["slices"].append(slice(-1, None))

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
