from logging import getLogger

import requests

from manten.utils.utils_serialization import MantenAgentSerialization

logger = getLogger(__name__)


class AgentProxyClient:
    def __init__(self, url="http://localhost:12567", timeout=10):
        self._url = url
        self._timeout = None

        # disable timeout for __init__ request
        logger.info("sending __init__ request to proxy")
        self._send_request("__init__", (), {})
        logger.info("proxy responded to __init__")

        self._timeout = timeout

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
                data=MantenAgentSerialization.serialize(
                    {"method": method, "args": args, "kwargs": kwargs}
                ),
                headers={"Content-Type": "application/octet-stream"},
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = MantenAgentSerialization.deserialize(response.content)
            if "result" in data:
                return data["result"]
            else:
                raise RuntimeError(data.get("error", "Unknown error"))
        except requests.RequestException as e:
            logger.exception("HTTP request failed", exc_info=e)
            raise
