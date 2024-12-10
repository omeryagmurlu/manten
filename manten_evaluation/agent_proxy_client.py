from multiprocessing.connection import Client
from typing import Protocol


class IMetric(Protocol):
    def feed(self, ground, prediction):
        pass

    def reset(self):
        pass

    def loss(self):
        pass

    def metrics(self) -> dict:
        pass

    def summary_metrics(self) -> dict:
        pass

    def visualize(self, **_) -> dict:
        pass


class IAgent(Protocol):
    def reset(self):
        pass

    def forward(self, agent_mode: str, *args, **kwargs) -> IMetric:
        pass


class AgentProxyClient(IAgent):
    def __init__(
        self,
        address=("localhost", 6000),
        authkey=bytes("manten_agent_server_secret", "utf-8"),
    ):
        self._conn = Client(address, authkey)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"Cannot access private attribute '{name}'")

        def method(*args):
            self._conn.send((name, args))  # Send method name and arguments
            return self._conn.recv()  # Receive the result

        return method

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._conn.send(("__shutdown__", ()))
        self._conn.close()
