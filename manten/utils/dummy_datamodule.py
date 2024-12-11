from typing import Protocol


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
        stats = self.train_dataloader.dataset.stats
        # return {"min": stats[0].tolist(), "max": stats[1].tolist()}
        return stats.tolist()
