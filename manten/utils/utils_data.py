import logging
from typing import Protocol, TypeVar

import torch

logger = logging.getLogger(__name__)


class UhaDataModuleProtocol(Protocol):
    def create_train_dataloader(self): ...
    def create_test_dataloader(self): ...
    def get_dataset_statistics(self): ...


class DummyDataModule(UhaDataModuleProtocol):
    def __init__(self, train_dataloader, test_dataloader):
        self.train_dataloader_fn = train_dataloader
        self.test_dataloader_fn = test_dataloader

    def create_train_dataloader(self, **kwargs):
        self.train_dataloader = self.train_dataloader_fn(**kwargs)
        return self.train_dataloader

    def create_test_dataloader(self, **kwargs):
        self.test_dataloader = self.test_dataloader_fn(**kwargs)
        return self.test_dataloader

    def get_dataset_info(self):
        info = self.train_dataloader.dataset.get_dataset_info()
        return info


T = TypeVar("T", bound=type[torch.utils.data.Dataset])


def modulo_dataset(cls: T) -> T:
    class ModuloDataset(cls):
        def __init__(
            self,
            *args,
            simulated_length=None,
            simulated_length_multiplier=None,
            simulated_batch_sizes=None,
            simulated_train_iterations=None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            if simulated_length is not None:
                target_length = simulated_length
            elif simulated_length_multiplier is not None:
                target_length = int(simulated_length_multiplier * super().__len__())
            elif simulated_batch_sizes is not None:
                bs, sbs = simulated_batch_sizes
                target_length = int((bs / sbs) * super().__len__())
            elif simulated_train_iterations is not None:
                bs, target_iters = simulated_train_iterations
                target_length = max(int(bs * target_iters), super().__len__())
            else:
                target_length = super().__len__()

            if target_length != super().__len__():
                logger.info(
                    "Changing dataset length from %d to the simulated length %d, this is an increase of factor %.2f",
                    super().__len__(),
                    target_length,
                    target_length / super().__len__(),
                )
            if target_length < super().__len__():
                logger.warning("Simulated length is less than the dataset length.")

            self.__target_length = target_length
            self.__len_dataset = super().__len__()

        def __len__(self):
            return max(self.__len_dataset, self.__target_length)

        def __getitem__(self, idx):
            return super().__getitem__(idx % self.__len_dataset)

    return ModuloDataset
