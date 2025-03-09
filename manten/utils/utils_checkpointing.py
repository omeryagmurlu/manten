from logging import getLogger
from pathlib import Path

import hydra
import omegaconf
from omegaconf import OmegaConf

logger = getLogger(__name__)


def load_model_from_safetensors(model, path, device=None):
    from safetensors.torch import load_model

    if device is not None:
        load_model(model, path, device=device, strict=False)
    else:
        load_model(model, path, strict=False)
    logger.info("Loaded safetensors from %s", path)


def save_model_to_safetensors(accelerator, model, path):
    # we can directly use safetensors, but accelerate also provides a convenient wrapper for it
    from accelerate.utils import save

    state_dict = accelerator.get_state_dict(model, unwrap=True)
    save(state_dict, path, safe_serialization=True)


def get_checkpoint_dir(train_folder, checkpoint):
    checkpoint_dir = Path(f"{train_folder}/{checkpoint}")
    if not Path(checkpoint_dir).exists():
        raise FileNotFoundError(f"Checkpoint directory not found at {checkpoint_dir}")
    if Path(checkpoint_dir).is_symlink():
        resolved_checkpoint_dir = Path(checkpoint_dir).resolve()
        logger.info(
            f"Resolved checkpoint alias {checkpoint_dir.name} to {resolved_checkpoint_dir}"
        )
        checkpoint_dir = resolved_checkpoint_dir
        del resolved_checkpoint_dir

    return checkpoint_dir


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

    checkpoint_dir = get_checkpoint_dir(train_folder, checkpoint)

    file_str = f"{checkpoint_dir}/%s"
    if use_ema:
        model_filepath = file_str % "ema_model.safetensors"
    else:
        model_filepath = file_str % "model.safetensors"
    agent_cfg_filepath = file_str % "agent_config.yaml"

    if not Path(model_filepath).exists():
        raise FileNotFoundError(f"Model file not found at {model_filepath}")
    if not Path(agent_cfg_filepath).exists():
        raise FileNotFoundError(f"Agent config file not found at {agent_cfg_filepath}")

    agent_cfg = OmegaConf.load(agent_cfg_filepath)
    if agent_override is not None:
        agent_cfg = OmegaConf.merge(agent_cfg, agent_override)

    agent = hydra.utils.instantiate(agent_cfg)
    if device is not None:
        agent.to(device)
    load_model_from_safetensors(agent, model_filepath, device=device)

    return agent


def save_agent_config(checkpoint_dir, agent_cfg):
    OmegaConf.save(
        config=agent_cfg,
        f=checkpoint_dir + "/agent_config.yaml",
        resolve=True,
    )


def add_omegaconf_to_safe_globals():
    import typing
    from collections import defaultdict

    import torch

    torch.serialization.add_safe_globals(
        [
            omegaconf.listconfig.ListConfig,
            omegaconf.dictconfig.DictConfig,
            omegaconf.base.ContainerMetadata,
            typing.Any,
            list,
            dict,
            defaultdict,
            int,
            omegaconf.nodes.AnyNode,
            omegaconf.base.Metadata,
        ]
    )
