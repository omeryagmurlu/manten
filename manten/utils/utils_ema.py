from diffusers.training_utils import EMAModel


class EMA:
    def __init__(self, manager: EMAModel, agent, accelerator):
        agent = accelerator.prepare(agent)
        accelerator.register_for_checkpointing(manager)

        self.manager = manager
        self.__agent = agent

    @property
    def agent(self):
        self.manager.copy_to(self.__agent.parameters())
        return self.__agent
