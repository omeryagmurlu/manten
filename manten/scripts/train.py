import hydra
from omegaconf import open_dict

from manten.utils.logging import get_logger
from manten.utils.utils_root import root

logger = get_logger(__name__)


def setup(cfg):  # noqa: PLR0915
    import os
    from pathlib import Path

    import torch
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from omegaconf import OmegaConf

    from manten.utils.train_loops import TrainLoops

    logger.info("Torch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("CUDA_VISIBLE_DEVICES: %s", os.getenv("CUDA_VISIBLE_DEVICES"))
    logger.info("CUDA device count: %d", torch.cuda.device_count())

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False, broadcast_buffers=False
    )
    accelerator = Accelerator(
        **(OmegaConf.to_container(cfg.accelerator, resolve=True)),
        project_dir=output_dir + "/accelerate",
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process and output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir + "/accelerate").mkdir(parents=True, exist_ok=True)
        Path(output_dir + "/tracker").mkdir(parents=True, exist_ok=True)

    # seeding
    base_seed = cfg.seed + accelerator.process_index * 100
    set_seed(base_seed, deterministic=cfg.deterministic)
    if cfg.deterministic:
        # also need to handle this but nvm for now: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        torch.backends.cudnn.benchmark = False

    def worker_init_fn(worker_id):
        set_seed(base_seed + worker_id * 10, deterministic=cfg.deterministic)

    datamodule = hydra.utils.instantiate(cfg.datamodule.datamodule)
    train_dataloader = datamodule.create_train_dataloader(worker_init_fn=worker_init_fn)
    test_dataloader = datamodule.create_test_dataloader(worker_init_fn=worker_init_fn)

    dataset_info = datamodule.get_dataset_info()

    with open_dict(cfg):
        cfg.agent.agent.dataset_info = dataset_info

    agent = hydra.utils.instantiate(cfg.agent.agent)

    optimizer_configurator = hydra.utils.instantiate(cfg.optimizer_configurator, agent=agent)
    optimizer = hydra.utils.instantiate(
        cfg.optimizer, optimizer_configurator.get_grouped_params()
    )

    # accelerate doesn't scale lr: https://huggingface.co/docs/accelerate/en/concept_guides/performance#learning-rates
    if (
        hasattr(cfg.lr_scheduler, "num_warmup_steps")
        and cfg.lr_scheduler.num_warmup_steps is not None
    ):
        cfg.lr_scheduler.num_warmup_steps = (
            cfg.lr_scheduler.num_warmup_steps * accelerator.num_processes
        )
    if (
        hasattr(cfg.lr_scheduler, "num_training_steps")
        and cfg.lr_scheduler.num_training_steps is not None
    ):
        cfg.lr_scheduler.num_training_steps = (
            cfg.lr_scheduler.num_training_steps * accelerator.num_processes
        )
    lr_scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer=optimizer)

    (agent, optimizer, train_dataloader, test_dataloader, lr_scheduler) = accelerator.prepare(
        agent, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    if cfg.training.ema:
        from manten.utils.utils_ema import EMA

        ema = EMA(
            manager=hydra.utils.instantiate(cfg.training.ema, parameters=agent.parameters()),
            agent=hydra.utils.instantiate(cfg.agent.agent),
            accelerator=accelerator,
        )
    else:
        ema = None

    if hasattr(cfg, "custom_evaluator") and cfg.custom_evaluator is not None:
        custom_evaluator = hydra.utils.instantiate(cfg.custom_evaluator)
        custom_evaluator = OmegaConf.to_object(custom_evaluator)
    else:
        custom_evaluator = None

    loops = TrainLoops(
        cfg.training,
        accelerator=accelerator,
        agent=agent,
        train_dl=train_dataloader,
        test_dl=test_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        log_aggregator=hydra.utils.instantiate(cfg.training.log_aggregator),
        custom_evaluator=custom_evaluator,
        ema=ema,
        whole_cfg=cfg,
    )

    loops.begin_sanity_check()

    # accelerator already handles is_main_process for trackers
    init_dict = {**OmegaConf.to_container(cfg.accelerator_init_trackers, resolve=True)}
    if "wandb" in init_dict["init_kwargs"]:
        init_dict["init_kwargs"]["wandb"]["dir"] = output_dir + "/tracker"
        init_dict["init_kwargs"]["wandb"]["config"] = OmegaConf.to_container(
            cfg, resolve=True
        )
        if "tags" in init_dict["init_kwargs"]["wandb"]:  # noqa: SIM102
            if hasattr(cfg.training, "custom_eval_only") and cfg.training.custom_eval_only:
                init_dict["init_kwargs"]["wandb"]["tags"].insert(0, "custom_eval_only")
    accelerator.init_trackers(**init_dict)

    loops.begin_training()

    # potentially online eval?

    accelerator.end_training()


@hydra.main(version_base="1.1", config_path=str(root / "configs"), config_name="train")
def main(cfg):
    """
    Main function to train an agent.
    """
    if hasattr(cfg, "debug") and cfg.debug is not None:
        hydra.utils.instantiate(cfg.debug)

    setup(cfg)


if __name__ == "__main__":
    main()
