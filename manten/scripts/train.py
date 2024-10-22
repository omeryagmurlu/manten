import hydra
import os

from tqdm import tqdm

from manten.utils.root import root


@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="train")
def main(cfg):
    accelerator = hydra.utils.instantiate(
        cfg.accelerator, project_dir=os.path.join(cfg.output_dir, "logs")
    )

    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)

        accelerator.init_trackers(**cfg.accelerator_init_trackers)

        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', None)}")

    model = hydra.utils.instantiate(cfg.noise_model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    lr_scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer=optimizer)
    dataloader = hydra.utils.instantiate(cfg.dataloader)

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    agent = hydra.utils.instantiate(cfg.agent, model=model)

    global_step = 0
    for epoch in range(cfg.num_epochs):
        progress_bar = tqdm(
            total=len(dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            train_generator = agent.train_step(batch)

            train_generator.send(None)
            with accelerator.accumulate(model):
                noised_inputs, timesteps = train_generator.send(None)
                noise_pred = model(noised_inputs, timesteps)
                loss = train_generator.send(noise_pred)

                # accelerator will run backward and optimizer step only once every cfg.gradient_accumulation_steps steps
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
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
    main()
