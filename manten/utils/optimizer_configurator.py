from typing import Protocol


class ParamConfig(Protocol):
    contains_substrings: list[str]
    # and rest of optimizer configuration, lr etc.


class OptimizerConfigurator:
    def __init__(
        self, *, agent, params_configs: list[ParamConfig] | None = None, default_params_config
    ):
        if params_configs is None:
            self.params_configs = []
        else:
            self.params_configs = [
                dict(params_config_cfg.items()) for params_config_cfg in params_configs
            ]
        self.default_params_config = dict(default_params_config.items())
        self.agent = agent

    def get_grouped_params(self):
        for name, param in self.agent.named_parameters():
            for config in self.params_configs:
                if any(substring in name for substring in config["contains_substrings"]):
                    if "params" not in config:
                        config["params"] = []
                    config["params"].append(param)
                    break
            else:
                if "params" not in self.default_params_config:
                    self.default_params_config["params"] = []
                self.default_params_config["params"].append(param)

        return [self.default_params_config, *self.params_configs]
