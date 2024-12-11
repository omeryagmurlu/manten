import hydra

from manten.utils.logging import get_logger
from manten.utils.utils_root import root

logger = get_logger(__name__)


def setup(cfg):
    import os

    import torch
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from omegaconf import OmegaConf

    from manten.utils.train_loops import TrainLoops
    from manten.utils.utils_file import mkdir

    if cfg.training.resume_from_save is not None:
        # hook here early (normally we load them in the loops) to load the cfg
        agent_cfg = OmegaConf.load(cfg.training.resume_from_save + "/agent_config.yaml")
        # for now only load the normalization stats to keep it simple, this is a hack
        # HACK:  # noqa: FIX004
        cfg.agent._dataset_stats = agent_cfg.position_normalization.dataset_stats  # noqa: SLF001

    set_seed(cfg.seed, deterministic=cfg.deterministic)
    if cfg.deterministic:
        # also need to handle this but nvm for now: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        torch.backends.cudnn.benchmark = False

    logger.info("Torch version: %s", torch.__version__)
    logger.info("CUDA version: %s", torch.version.cuda)
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
        mkdir(output_dir)
        mkdir(output_dir + "/accelerate")
        mkdir(output_dir + "/tracker")

    datamodule = hydra.utils.instantiate(cfg.datamodule.datamodule)
    train_dataloader = datamodule.create_train_dataloader()
    test_dataloader = datamodule.create_test_dataloader()

    if cfg.agent._dataset_stats is None:  # noqa: SLF001
        dataset_stats = datamodule.get_dataset_statistics()
        cfg.agent._dataset_stats = dataset_stats  # noqa: SLF001

    agent = hydra.utils.instantiate(cfg.agent.agent)

    optimizer_configurator = hydra.utils.instantiate(cfg.optimizer_configurator, agent=agent)
    optimizer = hydra.utils.instantiate(
        cfg.optimizer, optimizer_configurator.get_grouped_params()
    )

    # TODO: accelerate doesn't scale lr: https://huggingface.co/docs/accelerate/en/concept_guides/performance#learning-rates
    lr_scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer)

    (agent, optimizer, train_dataloader, test_dataloader, lr_scheduler) = accelerator.prepare(
        agent, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    loops = TrainLoops(
        cfg.training,
        accelerator=accelerator,
        agent=agent,
        train_dl=train_dataloader,
        test_dl=test_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        log_aggregator=hydra.utils.instantiate(cfg.training.log_aggregator),
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
    accelerator.init_trackers(**init_dict)

    loops.begin_training()

    # potentially online eval?

    accelerator.end_training()


@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="train")
def main(cfg):
    """
    Main function to train an agent.
    """
    if hasattr(cfg, "debug") and cfg.debug is not None:
        hydra.utils.instantiate(cfg.debug)

    setup(cfg)


if __name__ == "__main__":
    main()
