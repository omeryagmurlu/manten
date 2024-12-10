import hydra
from omegaconf import OmegaConf


def load_agent(
    train_folder: str,
    checkpoint: str | None,
    agent_override: dict | None,
    no_checkpoint_mode="last",
):
    if checkpoint is None:
        if no_checkpoint_mode == "last":
            checkpoint = "last_checkpoint"
        elif no_checkpoint_mode == "best":
            checkpoint = "best_checkpoint"
        else:
            raise ValueError(f"no_checkpoint_mode {no_checkpoint_mode} not recognized")

    agent_cfg = OmegaConf.load(f"{train_folder}/{checkpoint}/agent_config.yaml")
    if agent_override is not None:
        agent_cfg = OmegaConf.merge(agent_cfg, agent_override)
    agent = hydra.utils.instantiate(agent_cfg)
    return agent


def save_agent_config(checkpoint_path, agent_cfg):
    OmegaConf.save(
        config=agent_cfg,
        f=checkpoint_path + "/agent_config.yaml",
        resolve=True,
    )
