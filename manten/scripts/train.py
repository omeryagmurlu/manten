import hydra
import os
from tqdm import tqdm
from omegaconf import OmegaConf

from manten.utils.root import root
from accelerate.utils import set_seed
from manten.utils.logging import get_logger

logger = get_logger(__name__)

SANITY_CHECK_STEPS = 5


@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="train")
def main(cfg):
    """
    Main function to train an agent.
    """
    set_seed(cfg.training.seed)

    accelerator = hydra.utils.instantiate(
        cfg.accelerator, project_dir=os.path.join(cfg.project.output_dir, "logs")
    )

    if accelerator.is_main_process:
        if cfg.project.output_dir is not None:
            os.makedirs(cfg.project.output_dir, exist_ok=True)

    # accelerator already handles is_main_process for trackers
    accelerator.init_trackers(**cfg.accelerator_init_trackers)

    # same for logging
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', None)}")

    agent = hydra.utils.instantiate(cfg.agent)

    optimizer_configurator = hydra.utils.instantiate(
        cfg.optimizer_configurator, agent=agent
    )
    optimizer = hydra.utils.instantiate(
        cfg.optimizer,
        OmegaConf.to_container(optimizer_configurator.get_grouped_params()),
    )

    # TODO: accelerate doesn't scale lr: https://huggingface.co/docs/accelerate/en/concept_guides/performance#learning-rates
    lr_scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer)

    datamodule = hydra.utils.instantiate(cfg.datamodule)

    train_dataloader = datamodule.create_train_dataloader()
    test_dataloader = datamodule.create_test_dataloader()

    agent, optimizer, train_dataloader, test_dataloader, lr_scheduler = (
        accelerator.prepare(
            agent, optimizer, train_dataloader, test_dataloader, lr_scheduler
        )
    )

    # if cfg.training.sanity_check:
    #     logger.info("Running sanity check")
    #     for sanity_step, batch in enumerate(test_dataloader):
    #         if sanity_step == SANITY_CHECK_STEPS:
    #             break
    #         agent.(batch)
    #         if accelerator.is_main_process:
    #             logger.info(agent.metrics())

    logger.info("Starting training")
    global_step = 0
    for epoch in range(cfg.training.num_epochs):
        agent.train()
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            agent.reset()
            # accelerate also handles disabling synchronisation
            with accelerator.accumulate(agent):
                training_metric = agent.train_step(batch)
                loss = training_metric.loss()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(agent.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_local_main_process:
                progress_bar.update(1)
                overview_logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                progress_bar.set_postfix(**overview_logs)

            # accelerate already has @on_main_process deco on trackers, so this here
            # is mostly redundant, it only prevents agent.metrics() from being called
            # on non-main processes at all
            if accelerator.is_main_process:
                logs = training_metric.metrics()
                logs.update(overview_logs)
                accelerator.log(logs, step=global_step)

            global_step += 1

        # # After each epoch you optionally sample some demo images with evaluate() and save the model
        # if accelerator.is_main_process:
        #     pipeline = DDPMPipeline(
        #         unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
        #     )

        #     if (
        #         epoch + 1
        #     ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
        #         evaluate(config, epoch, pipeline)

        #     if (
        #         epoch + 1
        #     ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
        #         pipeline.save_pretrained(config.output_dir)


if __name__ == "__main__":
    if True:
        from manten.utils.debug_utils import monkeypatch_tensor_shape

        monkeypatch_tensor_shape()
    main()
