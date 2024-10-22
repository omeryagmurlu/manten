from abc import ABC, abstractmethod

import hydra


class BaseAgent(ABC):
    def __init__(self, config):
        self.config = config

        self.noise_scheduler = hydra.utils.instantiate(config.noise_scheduler)

    @abstractmethod
    def train_step(self, batch: dict):
        raise NotImplementedError
