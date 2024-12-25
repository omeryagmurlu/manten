from logging import getLogger

import hydra
from omegaconf import OmegaConf

logger = getLogger(__name__)


def load_model_from_safetensors(model, path, device=None):
    from safetensors.torch import load_model

    load_model(model, path, device)
    logger.info("Loaded safetensors from %s", path)


def save_model_to_safetensors(accelerator, model, path):
    # we can directly use safetensors, but accelerate also provides a convenient wrapper for it
    from accelerate.utils import save

    state_dict = accelerator.get_state_dict(model, unwrap=True)
    save(state_dict, path, safe_serialization=True)


def load_agent(
    train_folder: str,
    checkpoint: str | None,
    agent_override: dict | None,
    no_checkpoint_mode="last",
    device=None,
):
    if checkpoint is None:
        if no_checkpoint_mode == "last":
            checkpoint = "last_checkpoint"
        elif no_checkpoint_mode == "best":
            checkpoint = "best_checkpoint"
        else:
            raise ValueError(f"no_checkpoint_mode {no_checkpoint_mode} not recognized")

    file_str = f"{train_folder}/{checkpoint}/%s"

    agent_cfg = OmegaConf.load(file_str % "agent_config.yaml")
    if agent_override is not None:
        agent_cfg = OmegaConf.merge(agent_cfg, agent_override)
    agent = hydra.utils.instantiate(agent_cfg)
    if device is not None:
        agent.to(device)

    load_model_from_safetensors(agent, file_str % "model.safetensors", device=device)

    return agent


def save_agent_config(checkpoint_dir, agent_cfg):
    OmegaConf.save(
        config=agent_cfg,
        f=checkpoint_dir + "/agent_config.yaml",
        resolve=True,
    )
