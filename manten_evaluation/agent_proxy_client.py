import pickle
from logging import getLogger
from typing import Protocol

import requests

logger = getLogger(__name__)


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


class AgentProxyClient:
    def __init__(self, url="http://localhost:12567"):
        self._url = url

    def __call__(self, *args, **kwargs):
        return self._send_request("__call__", args, kwargs)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"Cannot access private attribute '{name}'")

        def method(*args, **kwargs):
            return self._send_request(name, args, kwargs)

        return method

    def close(self):
        self._send_request("__shutdown__", (), {})

    def _send_request(self, method, args, kwargs):
        try:
            response = requests.post(
                self._url,
                data=pickle.dumps({"method": method, "args": args, "kwargs": kwargs}),
                headers={"Content-Type": "application/octet-stream"},
                timeout=10,
            )
            response.raise_for_status()
            data = pickle.loads(response.content)  # noqa: S301
            if "result" in data:
                return data["result"]
            else:
                raise RuntimeError(data.get("error", "Unknown error"))
        except requests.RequestException as e:
            logger.exception("HTTP request failed", exc_info=e)
            raise


if __name__ == "__main__":
    client = AgentProxyClient()
    client.reset()
    print(1)
    client.reset()
    print(2)
    client.reset()
    print(3)
    client.reset()
    print("now calling directly")
    client("lol")
    print("done")
