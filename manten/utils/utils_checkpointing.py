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
    *,
    mode="last",
    checkpoint: str | None = None,
    agent_override: dict | None = None,
    device=None,
    use_ema=False,
):
    if checkpoint is None:
        if mode == "last":
            checkpoint = "last_checkpoint"
        elif mode == "best":
            checkpoint = "best_checkpoint"
        elif mode == "resume":
            raise ValueError("can't load checkpoints within resume, use other checkpoints")
        else:
            raise ValueError(f"mode {mode} not recognized")

    file_str = f"{train_folder}/{checkpoint}/%s"

    if use_ema:
        model_filename = file_str % "ema_model.safetensors"
    else:
        model_filename = file_str % "model.safetensors"

    agent_cfg = OmegaConf.load(file_str % "agent_config.yaml")
    if agent_override is not None:
        agent_cfg = OmegaConf.merge(agent_cfg, agent_override)

    agent = hydra.utils.instantiate(agent_cfg)
    if device is not None:
        agent.to(device)
    load_model_from_safetensors(agent, model_filename, device=device)

    return agent


def save_agent_config(checkpoint_dir, agent_cfg):
    OmegaConf.save(
        config=agent_cfg,
        f=checkpoint_dir + "/agent_config.yaml",
        resolve=True,
    )
