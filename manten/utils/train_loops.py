from dataclasses import dataclass, field
from typing import Protocol

import torch
from tabulate import tabulate

from manten.utils.logging import get_logger
from manten.utils.progbar import progbar
from manten.utils.utils_config import save_agent_config
from manten.utils.utils_decorators import with_state_dict

logger = get_logger(__name__)


@dataclass
@with_state_dict("epoch", "global_step", "best_mean_train_epoch_loss")
class TrainLoopsState:
    epoch: int = field(default=0)
    global_step: int = field(default=0)
    best_mean_train_epoch_loss: float = field(default=float("inf"))


class TrainLoopsConfig(Protocol):
    num_epochs: int
    max_steps: int
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


class TrainLoops:
    def __init__(
        self,
        cfg: TrainLoopsConfig,
        *,
        accelerator,
        agent,
        train_dl,
        test_dl,
        optimizer,
        lr_scheduler,
        log_aggregator,
        whole_cfg,
    ):
        self.cfg = cfg
        self.accelerator = accelerator
        self.agent = agent
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.log_aggregator = log_aggregator
        self.whole_cfg = whole_cfg  # this is a hack to save the agent config

        self.state = TrainLoopsState()
        accelerator.register_for_checkpointing(self.state)

        self.num_processes = accelerator.num_processes
        self.len_train_dl = len(train_dl)

    def begin_training(self):
        # Potentially load in the weights and states from a previous save
        if self.cfg.resume_from_save:
            self.resume_from_save(self.cfg.resume_from_save)

        logger.info("starting training")
        # loop control variable self overwrites as it iterates, but that's what we want
        to_epoch = self.cfg.num_epochs
        starting_epoch = self.state.epoch
        for self.state.epoch in range(starting_epoch, to_epoch):
            mean_epoch_loss, last_batch_loss = self.train_loop()

            if self.every_n_epochs(
                self.cfg.validate_every_n_epochs, self.cfg.skip_validate_first_n_epochs
            ):
                self.validation_loop()

            if self.every_n_epochs(
                self.cfg.eval_train_every_n_epochs, self.cfg.skip_eval_train_first_n_epochs
            ):
                self.evaluation_loop(
                    self.train_dl, self.cfg.eval_train_ene_max_steps, "train"
                )

            if self.every_n_epochs(
                self.cfg.eval_test_every_n_epochs, self.cfg.skip_eval_test_first_n_epochs
            ):
                self.evaluation_loop(self.test_dl, self.cfg.eval_test_ene_max_steps, "test")

            if self.every_n_epochs(
                self.cfg.save_every_n_epochs, self.cfg.skip_save_first_n_epochs
            ):
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
            progress := progbar(
                self.train_dl,
                desc=f"epoch {self.state.epoch}",
                total=min(len(self.train_dl), self.cfg.max_steps),
            )
        ):
            if step == self.cfg.max_steps:
                progress.close()
                break

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
            if self.accelerator.is_main_process and self.every_n_global_steps(
                self.cfg.log_every_n_steps, curr_train_step=step
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

        self.log_aggregator.all_gather()
        if self.accelerator.is_main_process:
            self.accelerator.log(
                validate_logs := self.log_aggregator.collate("validate/"),
                step=self.state.global_step,
            )
            print(tabulate(validate_logs.items()))
        self.log_aggregator.reset()

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

        self.log_aggregator.all_gather()
        if self.cfg.vis_metric_key is not None and self.accelerator.is_main_process:
            eval_logs = self.log_aggregator.collate(f"eval-{split}/", reset=False)
            print(tabulate(eval_logs.items()))
            eval_logs.update(self.log_aggregator.create_vis_logs(self.cfg.vis_metric_key))
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
            metric = self.agent("train", batch)
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
            metric = self.agent("validate", batch)
        return metric

    def evaluation_step(self, batch):
        with torch.inference_mode():
            metric, trajectory = self.agent("eval", batch, compare_gt=True)
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
        self.accelerator.wait_for_everyone()
        checkpoint_path = f"{self.accelerator.project_dir}/checkpoint_{self.state.epoch}"
        self.accelerator.save_state(checkpoint_path)
        self.save_agent_config(checkpoint_path)
        logger.info("checkpoint@epoch:%d saved to %s", self.state.epoch, checkpoint_path)

        if mean_epoch_loss < self.state.best_mean_train_epoch_loss:
            self.state.best_mean_train_epoch_loss = mean_epoch_loss
            best_checkpoint_path = f"{self.accelerator.project_dir}/best_checkpoint"
            self.accelerator.save_state(best_checkpoint_path)
            self.save_agent_config(best_checkpoint_path)
            logger.info(
                "best checkpoint@epoch:%d with loss %.4f saved to %s",
                self.state.epoch,
                self.state.best_mean_train_epoch_loss,
                best_checkpoint_path,
            )

    def save_agent_config(self, checkpoint_path):
        save_agent_config(checkpoint_path, self.whole_cfg.agent.agent)

    def every_n_epochs(self, n, skip_first=0):
        if self.state.epoch < skip_first:
            return False

        return bool(n) and (
            (self.state.epoch + 1) % n == 0
            or (self.state.epoch + 1) == self.cfg.num_epochs  # or last epoch
        )

    def every_n_global_steps(self, n, *, curr_train_step=0):
        return bool(n) and (
            (self.state.global_step + 1) % n == 0
            or (curr_train_step + 1) == self.len_train_dl  # or end of epoch
        )
