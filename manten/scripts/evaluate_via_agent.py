if __name__ == "__main__":
    from logging import getLogger

    import hydra
    import torch
    from accelerate.utils import set_seed
    from omegaconf import OmegaConf
    from tabulate import tabulate

    from manten.utils.utils_checkpointing import load_agent
    from manten.utils.utils_root import root

    logger = getLogger(__name__)

    @hydra.main(
        version_base="1.1",
        config_path=str(root / "configs"),
        config_name="evaluate_via_agent",
    )
    def main(cfg):
        if hasattr(cfg, "debug") and cfg.debug is not None:
            hydra.utils.instantiate(cfg.debug)

        # logger.info("config:\n%s", OmegaConf.to_yaml(cfg)) # already printed by hydra

        set_seed(cfg.seed, deterministic=cfg.deterministic)
        if cfg.deterministic:
            torch.backends.cudnn.benchmark = False

        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        logger.info("output_dir: %s", output_dir)

        load_agent_kwargs = OmegaConf.to_object(cfg.agent_creation.load_agent_kwargs)
        agent = load_agent(
            **load_agent_kwargs,
            device=cfg.device,
        )
        agent.eval()
        agent = hydra.utils.instantiate(cfg.agent_creation.agent_wrapper, agent=agent)

        evaluator = hydra.utils.instantiate(cfg.evaluator, output_dir=output_dir)

        logger.info("evaluating on %s", evaluator.eval_name)
        results, _ = evaluator.evaluate(agent)
        results = {f"{k}-mean": v.mean() for k, v in results.items()}
        logger.info(tabulate(results.items()))

    main()
