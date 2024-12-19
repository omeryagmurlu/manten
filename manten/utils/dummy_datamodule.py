from typing import Protocol

import torch
from torch.utils.data import Subset


class UhaDataModuleProtocol(Protocol):
    def create_train_dataloader(self): ...
    def create_test_dataloader(self): ...
    def get_dataset_statistics(self): ...


class DummyDataModule(UhaDataModuleProtocol):
    def __init__(self, train_dataloader, test_dataloader):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def create_train_dataloader(self):
        return self.train_dataloader

    def create_test_dataloader(self):
        return self.test_dataloader

    def get_dataset_statistics(self):
        stats = self.train_dataloader.dataset.compute_statistics()
        # return {"min": stats[0].tolist(), "max": stats[1].tolist()}
        return stats.tolist()


class AutoTestDataModule(UhaDataModuleProtocol):
    def __init__(
        self,
        dataset,
        dataloader_fn,
        test_ratio=0.2,
        shuffle_train=False,
        shuffle_test=False,
    ):
        len_dataset = len(dataset)
        indices = torch.randperm(len_dataset).tolist()
        split = int(test_ratio * len_dataset)
        test_indices = indices[:split]
        train_indices = indices[split:]
        self.train_dataloader = dataloader_fn(
            dataset=Subset(dataset, train_indices), shuffle=shuffle_train
        )
        self.test_dataloader = dataloader_fn(
            dataset=Subset(dataset, test_indices), shuffle=shuffle_test
        )

    def create_train_dataloader(self):
        return self.train_dataloader

    def create_test_dataloader(self):
        return self.test_dataloader

    def get_dataset_statistics(self):
        raise NotImplementedError
