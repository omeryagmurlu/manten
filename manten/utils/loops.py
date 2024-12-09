from dataclasses import dataclass, field
from typing import Protocol

import torch
from tabulate import tabulate

from manten.agents.base_agent import AgentMode
from manten.utils.logging import get_logger
from manten.utils.progbar import progbar

logger = get_logger(__name__)


@dataclass
class LoopsState:
    epoch: int = field(default=0)
    global_step: int = field(default=0)
    best_mean_train_epoch_loss: float = field(default=float("inf"))

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_mean_train_epoch_loss": self.best_mean_train_epoch_loss,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.global_step = state_dict["global_step"]
        self.best_mean_train_epoch_loss = state_dict["best_mean_train_epoch_loss"]


class LoopsConfig(Protocol):
    num_epochs: int
    sanity_check: int
    log_every_n_steps: int
    save_every_n_epochs: int
    resume_from_save: str | None
    validate_every_n_epochs: int
    eval_test_every_n_epochs: int
    eval_train_every_n_epochs: int
    validate_ene_max_steps: int
    eval_test_ene_max_steps: int
    eval_train_ene_max_steps: int


class Loops:
    def __init__(
        self,
        cfg: LoopsConfig,
        *,
        accelerator,
        agent,
        train_dl,
        test_dl,
        optimizer,
        lr_scheduler,
        log_aggregator,
    ):
        self.cfg = cfg
        self.accelerator = accelerator
        self.agent = agent
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.log_aggregator = log_aggregator

        self.state = LoopsState()
        accelerator.register_for_checkpointing(self.state)

        self.num_processes = accelerator.num_processes
        self.len_train_dl = len(train_dl)

    def begin_training(self):
        # Potentially load in the weights and states from a previous save
        if self.cfg.resume_from_save:
            self.resume_from_save(self.cfg.resume_from_save)

        logger.info("starting training")
        # loop control variable self overwrites as it iterates, but that's what we want
        for self.state.epoch in range(self.state.epoch, self.cfg.num_epochs):  # noqa: B020
            mean_epoch_loss, last_batch_loss = self.train_loop()

            if self.every_n_epochs_wait_fe_on_ism(self.cfg.validate_every_n_epochs):
                self.validation_loop()

            # if self.every_n_epochs_wait_fe_on_ism(self.cfg.eval_train_every_n_epochs):
            #     self.evaluation_loop(
            #         self.train_dl, self.cfg.eval_train_ene_max_steps, "train"
            #     )

            if self.every_n_epochs_wait_fe_on_ism(self.cfg.eval_test_every_n_epochs):
                self.evaluation_loop(self.test_dl, self.cfg.eval_test_ene_max_steps, "test")

            if self.every_n_epochs_wait_fe_on_ism(self.cfg.save_every_n_epochs):
                self.save_checkpoint(mean_epoch_loss)
        logger.info("training finished, trained for %d epochs", self.state.epoch)

    def begin_sanity_check(self):
        if bool(self.cfg.sanity_check):
            logger.info("starting sanity check")
            self.sanity_check_loop()
            logger.info("sanity check successful")

    def train_loop(self) -> tuple[float, float]:
        self.log_aggregator.reset()
        self.agent.train()
        train_epoch_losses = []
        for step, batch in enumerate(  # train_dl syncs rnd seed across processes
            progress := progbar(self.train_dl, desc=f"epoch {self.state.epoch}")
        ):
            metric, loss = self.train_step(batch)

            train_epoch_losses.append(loss := loss.detach().item())
            overview_logs = {
                "loss": loss,
                "lr": self.lr_scheduler.get_last_lr()[0],
                "timing/global_step": self.state.global_step,
            }
            progress.set_postfix(**overview_logs)

            # TODO: for now just log main process metrics for performance reasons
            # accelerate already has @on_main_process deco on trackers, so this here
            # is mostly redundant, it only prevents agent.metrics() from being called
            # on non-main processes at all
            if self.accelerator.is_main_process and self.every_n_universe_steps(
                self.cfg.log_every_n_steps
            ):
                upl = self.log_aggregator.log_collate(metric, "train/")
                upl.update(overview_logs)
                upl.update(
                    {
                        "timing/step": step,
                        "timing/epoch": self.state.epoch,
                        "timing/universe_step": self.state.global_step * self.num_processes,
                    }
                )
                self.accelerator.log(upl, step=self.state.global_step)

            self.state.global_step += 1

        return sum(train_epoch_losses) / len(train_epoch_losses), loss

    def validation_loop(self) -> None:
        self.log_aggregator.reset()
        self.agent.eval()
        for step, batch in enumerate(
            progress := progbar(
                self.test_dl,
                total=min(len(self.test_dl), self.cfg.validate_ene_max_steps),
                desc=f"epoch {self.state.epoch} (validate)",
                leave=False,
            )
        ):
            if step == self.cfg.validate_ene_max_steps:
                progress.close()
                break

            metric = self.validation_step(batch)

            progress.set_postfix(**metric.summary_metrics())
            self.log_aggregator.log(metric)

        logger.info("validate@epoch:%d", self.state.epoch)
        self.accelerator.log(
            validate_logs := self.log_aggregator.collate("validate/"),
            step=self.state.global_step,
        )
        print(tabulate(validate_logs.items()))

    def evaluation_loop(self, dataloader, max_steps, split) -> None:
        self.log_aggregator.reset()
        self.agent.eval()
        for step, batch in enumerate(
            progress := progbar(
                dataloader,
                total=min(len(dataloader), max_steps),
                desc=f"epoch {self.state.epoch} (eval:{split})",
                leave=False,
            )
        ):
            if step == max_steps:
                progress.close()
                break

            metric, trajectory = self.evaluation_step(batch)

            progress.set_postfix(**metric.summary_metrics())
            self.log_aggregator.log(metric)

        logger.info("evaluation:%s@epoch:%d", split, self.state.epoch)

        eval_logs = self.log_aggregator.collate(f"eval-{split}/", reset=False)
        print(tabulate(eval_logs.items()))
        eval_logs.update(self.log_aggregator.create_vis_logs("mae_pos"))
        self.accelerator.log(eval_logs, step=self.state.global_step)
        self.log_aggregator.reset()

    def sanity_check_loop(self):
        self.agent.eval()
        for step, batch in enumerate(
            progress := progbar(
                self.test_dl,
                total=min(len(self.test_dl), self.cfg.sanity_check),
                desc="sanity check (validate)",
            )
        ):
            if step == self.cfg.sanity_check:
                progress.close()
                break

            metric = self.validation_step(batch)
            progress.set_postfix(**metric.summary_metrics())

        self.agent.eval()
        for step, batch in enumerate(
            progress := progbar(
                self.test_dl,
                total=min(len(self.test_dl), self.cfg.sanity_check),
                desc="sanity check (eval)",
            )
        ):
            if step == self.cfg.sanity_check:
                progress.close()
                break

            metric, trajectory = self.evaluation_step(batch)
            progress.set_postfix(**metric.summary_metrics())

    def train_step(self, batch):
        with self.accelerator.accumulate(self.agent):
            metric = self.agent(AgentMode.TRAIN, batch)
            loss = metric.loss()

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.agent.parameters(), 1.0)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return metric, loss

    def validation_step(self, batch):
        with torch.inference_mode():
            metric = self.agent(AgentMode.VALIDATE, batch)
        return metric

    def evaluation_step(self, batch):
        with torch.inference_mode():
            trajectory, metric = self.agent(AgentMode.EVAL, batch, compare_gt=True)
        return metric, trajectory

    def resume_from_save(self, from_str):
        # TODO: maybe later automatic loading of last/best checkpoint
        checkpoint_to_load = from_str

        self.accelerator.load_state(checkpoint_to_load)
        logger.info(
            "resuming from checkpoint@epoch:%d via %s", self.state.epoch, checkpoint_to_load
        )
        self.state.epoch += (
            1  # to start from the next epoch instead of redoing the current one
        )

    def save_checkpoint(self, mean_epoch_loss):
        checkpoint_path = f"{self.accelerator.project_dir}/checkpoint_{self.state.epoch}"
        self.accelerator.save_state(checkpoint_path)
        logger.info("checkpoint@epoch:%d saved to %s", self.state.epoch, checkpoint_path)

        if mean_epoch_loss < self.state.best_mean_train_epoch_loss:
            self.state.best_mean_train_epoch_loss = mean_epoch_loss
            best_checkpoint_path = f"{self.accelerator.project_dir}/best_checkpoint"
            self.accelerator.save_state(best_checkpoint_path)
            logger.info(
                "best checkpoint@epoch:%d with loss %.4f saved to %s",
                self.state.epoch,
                self.state.best_mean_train_epoch_loss,
                best_checkpoint_path,
            )

    def every_n_epochs(self, n):
        return bool(n) and (
            (self.state.epoch + 1) % n == 0
            or (self.state.epoch + 1) == self.cfg.num_epochs  # or last epoch
        )

    def every_n_epochs_wait_fe_on_ism(self, n):
        if not self.every_n_epochs(n):
            return False

        self.accelerator.wait_for_everyone()
        return self.accelerator.is_main_process

    def every_n_global_steps(self, n):
        return bool(n) and (
            (self.state.global_step + 1) % n == 0
            or (self.step + 1) == self.len_train_dl  # or end of epoch
        )

    def every_n_universe_steps(self, n):
        return bool(n) and (
            ((self.state.global_step + 1) * self.num_processes) % n < self.num_processes
            or (self.step + 1) == self.len_train_dl  # or end of epoch
        )
