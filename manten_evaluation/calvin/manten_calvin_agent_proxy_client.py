from calvin.calvin_models.calvin_agent.models.calvin_base_model import CalvinBaseModel
from manten_evaluation.agent_proxy_client import AgentProxyClient


class MantenCalvinAgentProxyClient(CalvinBaseModel):
    def __init__(self):
        self._agent = AgentProxyClient(
            # change address, port, authkey here if you changed them from the defaults of the server
        )

    def reset(self):
        """
        This is called
        """
        self._agent.reset()

    def step(self, obs, _goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        return obs[:7]
