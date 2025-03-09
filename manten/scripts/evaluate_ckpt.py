import argparse
import datetime
import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tabulate import tabulate

from manten.utils.logging import setup_custom_logging
from manten.utils.utils_checkpointing import get_checkpoint_dir, load_agent
from manten.utils.utils_seeding_wo_accelerate import set_seed_wo_accelerate
from manten.utils.utils_visualization import handle_rich_media_for_logs

logger = logging.getLogger(__name__)


def load_hydra_configs(train_path):
    """Load the Hydra configuration files from the training directory."""
    train_path = Path(train_path)

    # Load main config
    config_path = train_path / ".hydra" / "config.yaml"
    hydra_path = train_path / ".hydra" / "hydra.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    if not hydra_path.exists():
        raise FileNotFoundError(f"Hydra config file not found at {hydra_path}")

    # Load configs using OmegaConf
    config = OmegaConf.load(config_path)
    hydra_config = OmegaConf.load(hydra_path)

    return config, hydra_config


def validate_train_path(train_path):
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found at {train_path}")
    if train_path.is_symlink():
        resolved_train_path = train_path.resolve()
        logger.info(f"Resolved train_path {train_path.name} to {resolved_train_path}")

    return train_path


def setup_output_dir(hydra_config, ckpt_name, debug=False):
    # hydra_config.hydra.runtime.output_dir is the output directory of the training run
    # create a new directory under that called "evaluation" to store the evaluation results
    # and create a new, date-time checkpoint stamped directory under that to store the evaluation results
    # also, add a file handler to the logger to log to a file in the evaluation directory
    output_dir = hydra_config.hydra.runtime.output_dir
    eval_dir = Path(output_dir) / "evaluation"
    if debug:
        eval_dir = eval_dir / "debug"
    eval_dir.mkdir(parents=True, exist_ok=True)
    now_dir = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = eval_dir / now_dir / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"{Path(__file__).stem}.log"
    setup_custom_logging(log_file)

    return output_dir


def setup_wandb(
    *,
    entity=None,
    project=None,
    resume_run: bool,
    from_train,
    train_path,
    output_dir,
    debug=False,
):
    import wandb

    if entity is None:
        entity = from_train["init_kwargs"]["wandb"]["entity"]
    if project is None:
        project = from_train["project_name"]

    if entity is None or project is None:
        raise ValueError("Wandb entity and project must be provided")

    if resume_run:
        tags = from_train["init_kwargs"]["wandb"]["tags"]
        run_id = next(
            filter(
                lambda x: x.stem.startswith("run"),
                (train_path / "tracker" / "wandb").iterdir(),
            )
        ).stem.split("-")[-1]

        wandb_logger = wandb.init(
            entity=entity,
            project=project,
            tags=[*tags, "did_eval_ckpt"],
            resume="must",
            id=run_id,
            dir=str(output_dir / "tracker"),
            mode="disabled" if debug else "online",
        )
    else:
        wandb_logger = wandb.init(
            entity=entity,
            project=project,
            tags=["did_eval_ckpt"],
            dir=str(output_dir / "tracker"),
            mode="disabled" if debug else "online",
        )

    return wandb_logger


def main(args):
    train_path = validate_train_path(args.train_path)

    # Load Hydra configurations
    config, hydra_config = load_hydra_configs(train_path)

    # Use values from config if not provided in args
    seed = args.seed if args.seed is not None else config.get("seed", None)
    debug = args.debug if args.debug is not None else config.get("debug", None)

    if len(args.custom_eval_json) != 0:
        debug = debug or True

    output_dir = setup_output_dir(hydra_config, args.ckpt_name, debug=bool(debug))
    wandb_logger = setup_wandb(
        entity=args.wandb_entity,
        project=args.wandb_project,
        resume_run=args.wandb_resume_run,
        from_train=OmegaConf.to_object(config.accelerator_init_trackers),
        train_path=train_path,
        output_dir=output_dir,
        debug=bool(debug),
    )
    # validate again so that it's logged in the log file too
    validate_train_path(args.train_path)

    accel_path = Path(train_path) / "accelerate"
    checkpoint_dir = get_checkpoint_dir(accel_path, args.ckpt_name)
    checkpoint = checkpoint_dir.name
    checkpoint_epoch = int(checkpoint.split("_")[-1])
    logger.info(f"Using training directory: {train_path}")
    logger.info(f"Using checkpoint: {checkpoint}")

    set_seed_wo_accelerate(seed)
    logger.info(f"Using seed: {seed}")

    if debug is not None and (isinstance(debug, (DictConfig, dict))):
        logger.info("Using debug")
        hydra.utils.instantiate(debug)
    else:
        logger.info(f"Using debug mode: {debug}")

    ema_params = [False]
    if (checkpoint_dir / "ema_model.safetensors").exists():
        ema_params.append(True)

    if not hasattr(config, "custom_evaluator") or config.custom_evaluator is None:
        raise ValueError(
            "No custom_evaluator found in train config, please provide one in order to use automatic checkpoint evaluation"
        )

    custom_evaluator = OmegaConf.to_object(hydra.utils.instantiate(config.custom_evaluator))
    custom_evaluators = (
        custom_evaluator if isinstance(custom_evaluator, list) else [custom_evaluator]
    )
    for ema in ema_params:
        agent = load_agent(accel_path, checkpoint=checkpoint, device=args.device, use_ema=ema)
        for idx, custom_evaluator in enumerate(custom_evaluators):
            proc_name = "custom_eval"
            if ema:
                proc_name = f"{proc_name}-ema"
            proc_name = f"{proc_name}-{idx}"
            if callable(custom_evaluator):
                custom_eval = custom_evaluator(
                    output_dir=f"{output_dir}/{checkpoint}/{proc_name}",
                    **args.custom_eval_json,
                )
            else:
                custom_eval = custom_evaluator
            proc_name = f"{proc_name}-{custom_eval.eval_name}"

            with torch.inference_mode(), torch.no_grad():
                agent.eval()
                eval_infos, rich_media = custom_eval.evaluate(agent)

            rich_logs = handle_rich_media_for_logs(rich_media)
            eval_logs = {f"{k}-mean": v.mean() for k, v in eval_infos.items()}

            custom_eval_logs = {
                f"{proc_name}/{k}": v for (k, v) in ({**eval_logs, **rich_logs}).items()
            }
            custom_eval_logs.update(
                {
                    "eval_epoch": checkpoint_epoch,
                }
            )

            wandb_logger.log(custom_eval_logs)
            logger.info(tabulate(custom_eval_logs.items()))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint")
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to the training directory containing checkpoints",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="last_checkpoint",
        help="Name of the checkpoint file (default: last_checkpoint)",
    )
    parser.add_argument(
        "--debug",
        type=str,
        default=None,
        help="Enable debug mode with specified debug options. By default inferred from train config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. By default inferred from train config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for evaluation (default: cuda)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity to log to. By default inferred from train config.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Wandb project to log to. By default inferred from train config.",
    )
    parser.add_argument(
        "--wandb_resume_run",
        type=bool,
        default=True,
        help="Resume the wandb run inferred from train config.",
    )
    parser.add_argument(
        "--custom_eval_json",
        type=json.loads,
        default="{}",
        help="Custom evaluation configuration in json format",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
