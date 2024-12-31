from omegaconf import DictConfig, ListConfig, OmegaConf


def to_object_graceful(config):
    if isinstance(config, (DictConfig, ListConfig)):
        return OmegaConf.to_object(config)
    return config
