from typing import List, Protocol


class ParamConfig(Protocol):
    contains_substrings: List[str]
    # and rest of optimizer configuration, lr etc.


class OptimizerConfigurator:
    def __init__(
        self, *, agent, params_configs: List[ParamConfig] = [], default_params_config
    ):
        self.agent = agent
        self.params_configs = params_configs
        self.default_params_config = default_params_config

    def get_grouped_params(self):
        for name, param in self.agent.named_parameters():
            for config in self.params_configs:
                if any(substring in name for substring in config.contains_substrings):
                    if "params" not in config:
                        config["params"] = []
                    config["params"].append(param)
                    break
            else:
                if "params" not in self.default_params_config:
                    self.default_params_config["params"] = []
                self.default_params_config["params"].append(param)

        return [self.default_params_config] + self.params_configs
