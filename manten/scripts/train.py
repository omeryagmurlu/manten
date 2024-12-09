import os
from dataclasses import dataclass, field

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from tabulate import tabulate

from manten.agents.base_agent import AgentMode
from manten.utils.logging import get_logger
from manten.utils.progbar import progbar
from manten.utils.root import root
from manten.utils.utils_file import mkdir, write_json


def nop(*_, **__):
    return


logger = get_logger(__name__)


def loop_factory(
    *,
    compute_batch_result,
    after_batch=nop,
    progress_bar_impl=progbar,
    progbar_kwargs=None,
):
    progbar_kwargs_top = progbar_kwargs
    if progbar_kwargs_top is None:
        progbar_kwargs_top = {}

    def loop(
        dataloader,
        agent,
        *,
        max_steps=float("inf"),
        log_aggregator=None,
        progbar_kwargs=None,
    ):
        if progbar_kwargs is None:
            progbar_kwargs = {}
        progbar_kwargs = {**progbar_kwargs_top, **progbar_kwargs}

        if log_aggregator:
            log_aggregator.reset()
        agent.eval()
        for step, batch in enumerate(
            progress := progress_bar_impl(
                dataloader,
                total=min(len(dataloader), max_steps),
                **progbar_kwargs,
            )
        ):
            if step == max_steps:
                break
            batch_result = compute_batch_result(
                agent=agent,
                batch=batch,
                log_aggregator=log_aggregator,
                step=step,
                progress=progress,
            )
            after_batch(
                agent=agent,
                step=step,
                batch=batch,
                batch_result=batch_result,
                log_aggregator=log_aggregator,
                progress=progress,
            )
        progress.close()

    return loop


def do_agent_inference_validate(agent, batch, **_):
    with torch.inference_mode():
        metric = agent(AgentMode.VALIDATE, batch)
    return metric


def do_agent_inference_eval(agent, batch, **_):
    with torch.inference_mode():
        trajectory, metric = agent(AgentMode.EVAL, batch, compare_gt=True)
    return trajectory, metric


def update_progress(*, progress, metric, **_):
    progress.set_postfix(**metric.summary_metrics())


def update_progress_n_log(*, progress, metric, log_aggregator, **_):
    progress.set_postfix(**metric.summary_metrics())
    log_aggregator.log(metric)


def sanity_check(dataloader, agent, sanity_check_steps=5):
    validate_sanity_check = loop_factory(
        compute_batch_result=do_agent_inference_validate,
        after_batch=lambda batch_result, **ka: update_progress(metric=batch_result, **ka),
        progbar_kwargs={"desc": "sanity check (validate)"},
    )

    eval_sanity_check = loop_factory(
        compute_batch_result=do_agent_inference_eval,
        after_batch=lambda batch_result, **ka: update_progress(metric=batch_result[1], **ka),
        progbar_kwargs={"desc": "sanity check (eval)"},
    )

    validate_sanity_check(
        dataloader,
        agent,
        max_steps=sanity_check_steps,
    )
    eval_sanity_check(
        dataloader,
        agent,
        max_steps=sanity_check_steps,
    )

    logger.info("Sanity check successful")


@dataclass
class TrainState:
    epoch: int = field(default=0)
    global_step: int = field(default=0)
    best_loss: float = field(default=float("inf"))

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.global_step = state_dict["global_step"]
        self.best_loss = state_dict["best_loss"]


# TODO: This is apparently a very long and complex function, but it'll have to do for now
def train(  # noqa: C901, PLR0915, PLR0912
    cfg,
    accelerator,
    agent,
    train_dl,
    test_dl,
    optimizer,
    lr_scheduler,
    log_aggregator,
):
    validate = loop_factory(
        compute_batch_result=do_agent_inference_validate,
        after_batch=lambda batch_result, **kwargs: update_progress_n_log(
            metric=batch_result, **kwargs
        ),
        progbar_kwargs={"leave": False},
    )
    eeval = loop_factory(
        compute_batch_result=do_agent_inference_eval,
        after_batch=lambda batch_result, **kwargs: update_progress_n_log(
            metric=batch_result[1], **kwargs
        ),
        progbar_kwargs={"leave": False},
    )

    def every_n_epochs(n):
        return bool(n) and ((state.epoch + 1) % n == 0 or (state.epoch + 1) == cfg.num_epochs)

    def every_n_global_steps(n):
        return bool(n) and ((state.global_step + 1) % n == 0 or (step + 1) == len_train_dl)

    def every_n_universe_steps(n):
        return bool(n) and ( ((state.global_step + 1) * num_processes) % n < num_processes or (step + 1) == len_train_dl )

    ism = accelerator.is_main_process
    num_processes = accelerator.num_processes
    len_train_dl = len(train_dl)

    state = TrainState()
    accelerator.register_for_checkpointing(state)

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_save:
        # TODO: maybe later automatic loading of last/best checkpoint
        checkpoint_to_load = cfg.resume_from_save

        accelerator.load_state(checkpoint_to_load)
        logger.info(
            "resuming from checkpoint@epoch:%d via %s", state.epoch, checkpoint_to_load
        )
        state.epoch += 1  # to start from the next epoch instead of redoing the current one

    logger.info("starting training")
    start_epoch = state.epoch
    for state.epoch in range(start_epoch, cfg.num_epochs):
        log_aggregator.reset()
        agent.train()
        for step, batch in enumerate( # train_dl syncs rnd seed across processes
            progress := progbar(train_dl, desc=f"epoch {state.epoch}")
        ):
            with accelerator.accumulate(agent):
                metric = agent(AgentMode.TRAIN, batch)
                loss = metric.loss()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(agent.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            overview_logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "global_step": state.global_step,
            }
            progress.set_postfix(**overview_logs)

            # accelerate already has @on_main_process deco on trackers, so this here
            # is mostly redundant, it only prevents agent.metrics() from being called
            # on non-main processes at all
            if ism and every_n_universe_steps(cfg.log_every_n_steps):
                upl = log_aggregator.log_collate(metric, "train/")
                upl.update(overview_logs)
                upl.update({
                    "step": step,
                    "epoch": state.epoch,
                    "universe_step": state.global_step * num_processes,
                })
                accelerator.log(upl, step=state.global_step)

            state.global_step += 1

        if every_n_epochs(cfg.validate_every_n_epochs):
            accelerator.wait_for_everyone()
            if ism:
                validate(
                    test_dl,
                    agent,
                    max_steps=cfg.validate_ene_max_steps,
                    progbar_kwargs={"desc": f"epoch {state.epoch} (validate)"},
                    log_aggregator=log_aggregator,
                )
                logger.info("validate@epoch:%d", state.epoch)
                accelerator.log(
                    validate_logs := log_aggregator.collate("validate/"), step=state.global_step
                )
                print(tabulate(validate_logs.items()))

        for every_n, dl, max_steps, split in [
            # (cfg.eval_train_every_n_epochs, train_dl, cfg.eval_train_ene_max_steps, "train"),
            (cfg.eval_test_every_n_epochs, test_dl, cfg.eval_test_ene_max_steps, "test"),
        ]:
            if every_n_epochs(every_n):
                accelerator.wait_for_everyone()
                if ism:
                    eeval(
                        dl,
                        agent,
                        max_steps=max_steps,
                        progbar_kwargs={"desc": f"epoch {state.epoch} (eval:{split})"},
                        log_aggregator=log_aggregator,
                    )
                    logger.info("evaluation:%s@epoch:%d", split, state.epoch)

                    eval_logs = log_aggregator.collate(f"eval-{split}/", reset=False)
                    print(tabulate(eval_logs.items()))
                    eval_logs.update(log_aggregator.create_vis_logs("mae_pos"))
                    accelerator.log(eval_logs, step=state.global_step)

                    log_aggregator.reset()

        if every_n_epochs(cfg.save_every_n_epochs):
            accelerator.wait_for_everyone()
            if ism:
                checkpoint_path = f"{accelerator.project_dir}/checkpoint_{state.epoch}"
                accelerator.save_state(checkpoint_path)
                logger.info("checkpoint@epoch:%d saved to %s", state.epoch, checkpoint_path)

                if overview_logs["loss"] < state.best_loss:
                    state.best_loss = overview_logs["loss"]
                    best_checkpoint_path = f"{accelerator.project_dir}/best_checkpoint"
                    accelerator.save_state(best_checkpoint_path)
                    write_json(
                        f"{accelerator.project_dir}/best_checkpoint/overview_logs.json",
                        overview_logs,
                    )
                    logger.info(
                        "best checkpoint@epoch:%d with loss %.4f saved to %s",
                        state.epoch,
                        state.best_loss,
                        best_checkpoint_path,
                    )


@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="train")
def main(cfg):
    """
    Main function to train an agent.
    """
    set_seed(cfg.training.seed, deterministic=cfg.training.deterministic)
    if cfg.training.deterministic:
        # also need to handle this but nvm for now: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        torch.backends.cudnn.benchmark = False

    if hasattr(cfg, "debug") and cfg.debug is not None:
        hydra.utils.instantiate(cfg.debug)
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

    agent = hydra.utils.instantiate(cfg.agent.agent)

    optimizer_configurator = hydra.utils.instantiate(cfg.optimizer_configurator, agent=agent)
    optimizer = hydra.utils.instantiate(
        cfg.optimizer, optimizer_configurator.get_grouped_params()
    )

    # TODO: accelerate doesn't scale lr: https://huggingface.co/docs/accelerate/en/concept_guides/performance#learning-rates
    lr_scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer)

    datamodule = hydra.utils.instantiate(cfg.datamodule.datamodule)
    train_dataloader = datamodule.create_train_dataloader()
    test_dataloader = datamodule.create_test_dataloader()

    (agent, optimizer, train_dataloader, lr_scheduler) = accelerator.prepare(
        agent, optimizer, train_dataloader, lr_scheduler
    )

    log_aggregator = hydra.utils.instantiate(cfg.training.log_aggregator)

    if bool(cfg.training.sanity_check):
        sanity_check(test_dataloader, agent, cfg.training.sanity_check)

    # accelerator already handles is_main_process for trackers
    init_dict = {**OmegaConf.to_container(cfg.accelerator_init_trackers, resolve=True)}
    if "wandb" in init_dict["init_kwargs"]:
        init_dict["init_kwargs"]["wandb"]["dir"] = output_dir + "/tracker"
        init_dict["init_kwargs"]["wandb"]["config"] = OmegaConf.to_container(
            cfg, resolve=True
        )
    accelerator.init_trackers(**init_dict)

    train(
        cfg.training,
        accelerator,
        agent,
        train_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
        log_aggregator,
    )

    # potentially online eval?


if __name__ == "__main__":
    main()
