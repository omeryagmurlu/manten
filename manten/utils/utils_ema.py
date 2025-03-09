from diffusers.training_utils import EMAModel


class EMAContextManager:
    def __init__(self, manager: EMAModel, agent):
        self.manager = manager
        self.agent = agent

    def __enter__(self):
        self.manager.store(self.agent.parameters())
        self.manager.copy_to(self.agent.parameters())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.restore(self.agent.parameters())


class EMA:
    def __init__(self, manager: EMAModel, accelerator):
        accelerator.register_for_checkpointing(manager)

        self.manager = manager
        self.accelerator = accelerator

    def context_manager(self, agent):
        return EMAContextManager(self.manager, self.accelerator.unwrap_model(agent))

    def step(self, agent):
        # no need to unwrap here? see: https://github.com/huggingface/diffusers/blob/1fddee211ea61edcbe5476f7fbc7ce35b8de5200/examples/unconditional_image_generation/train_unconditional.py#L592
        self.manager.step(agent.parameters())
