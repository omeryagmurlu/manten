import gc
import shutil
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
from accelerate import Accelerator
from tabulate import tabulate

from manten.utils.debug_utils import TrainTimer
from manten.utils.logging import get_logger
from manten.utils.progbar import progbar
from manten.utils.utils_checkpointing import (
    add_omegaconf_to_safe_globals,
    save_agent_config,
    save_model_to_safetensors,
)
from manten.utils.utils_decorators import with_state_dict
from manten.utils.utils_visualization import handle_rich_media_for_logs

logger = get_logger(__name__)


class EveryNConfig(Protocol):
    every_n_epochs: int | None
    skip_first_epochs: int | None
    every_n_global_steps: int | None
    skip_first_global_steps: int | None


class HasMaxSteps(Protocol):
    max_steps: int


class EveryNConfigWithMaxSteps(EveryNConfig, HasMaxSteps):
    pass


class CustomEvaluator(Protocol):
    eval_name: str
    evaluate: Callable


class TrainLoopsConfig(Protocol):
    sanity_check: int
    resume_from_save: str | None

    num_epochs: int
    num_global_steps: int
    max_steps: int
    log_every_n_steps: int

    save: EveryNConfig
    val: EveryNConfigWithMaxSteps
    eval_test: EveryNConfigWithMaxSteps
    eval_train: EveryNConfigWithMaxSteps
    custom_eval: EveryNConfig

    vis_metric_key: str | None
    log_train_timing: bool


@dataclass
@with_state_dict("epoch", "global_step", "best_mean_train_epoch_loss")
class TrainLoopsState:
    epoch: int = field(default=0)
    global_step: int = field(default=0)
    best_mean_train_epoch_loss: float = field(default=float("inf"))


class TrainLoops:
    def __init__(
        self,
        cfg: TrainLoopsConfig,
        *,
        accelerator: Accelerator,
        agent,
        train_dl,
        test_dl,
        optimizer,
        lr_scheduler,
        log_aggregator,
        custom_evaluator: CustomEvaluator
        | Callable[..., CustomEvaluator]
        | list[CustomEvaluator]
        | list[Callable[..., CustomEvaluator]]
        | None,
        ema,
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
        self.custom_evaluator = custom_evaluator
        self.ema = ema
        self.whole_cfg = whole_cfg  # this is a hack to save the agent config

        self.state = TrainLoopsState()
        accelerator.register_for_checkpointing(self.state)

        self.num_processes = accelerator.num_processes
        self.len_train_loop = min(len(train_dl), cfg.max_steps)

        self.log_train_timing = (
            TrainTimer(accelerator=accelerator) if self.cfg.log_train_timing else None
        )

    def begin_training(self):
        # Potentially load in the weights and states from a previous save
        if self.cfg.resume_from_save:
            self.resume_from_save(self.cfg.resume_from_save)

        if hasattr(self.cfg, "custom_eval_only") and self.cfg.custom_eval_only:
            logger.info("starting custom evaluation only")
            with torch.no_grad():
                self.agent.eval()
                self.custom_evaluation()
                if self.ema:
                    self.custom_evaluation(use_ema=True)
            logger.info("custom evaluation finished, exiting")
            return

        logger.info("starting training")
        to_epoch = self.cfg.num_epochs if self.cfg.num_epochs else float("inf")
        to_global_step = (
            self.cfg.num_global_steps if self.cfg.num_global_steps else float("inf")
        )
        while True:
            if self.state.global_step >= to_global_step or self.state.epoch >= to_epoch:
                break

            mean_epoch_loss, last_batch_loss = self.train_loop()

            if self.every_n_after_train(self.cfg.save):
                self.save_checkpoint(mean_epoch_loss)

            with torch.no_grad():
                self.agent.eval()
                self.trigger_validation()
                if self.ema:
                    self.trigger_validation(use_ema=True)

            self.state.epoch += 1
        logger.info("training finished, trained for %d epochs", self.state.epoch)

    @torch.inference_mode()
    def trigger_validation(self, **kwa):
        if self.every_n_after_train(self.cfg.val):
            self.free_memory()
            self.validation_loop(**kwa)
            self.free_memory()

        if self.every_n_after_train(self.cfg.eval_train):
            self.free_memory()
            self.evaluation_loop(self.train_dl, self.cfg.eval_train.max_steps, "train", **kwa)
            self.free_memory()

        if self.every_n_after_train(self.cfg.eval_test):
            self.free_memory()
            self.evaluation_loop(self.test_dl, self.cfg.eval_test.max_steps, "test", **kwa)
            self.free_memory()

        if self.every_n_after_train(self.cfg.custom_eval):
            self.free_memory()
            self.custom_evaluation(**kwa)
            self.free_memory()

    def begin_sanity_check(self):
        if bool(self.cfg.sanity_check):
            logger.info("starting sanity check")
            self.sanity_check_loop()
            logger.info("sanity check successful")

    def train_loop(self) -> tuple[float, float]:
        self.log_aggregator.reset()
        self.agent.train()
        train_epoch_losses = deque()
        local_log_every_n_steps = max(self.cfg.log_every_n_steps // 10, 1)
        for step, batch in enumerate(  # train_dl syncs rnd seed across processes
            progress := progbar(
                self.train_dl,
                desc=f"epoch {self.state.epoch}",
                total=self.len_train_loop,
                mininterval=1,
            )
        ):
            if step == self.cfg.max_steps:
                progress.close()
                break

            metric, loss = self.train_step(batch)

            train_epoch_losses.append(loss := loss.detach().item())
            if step % local_log_every_n_steps == 0:
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

    @torch.inference_mode()
    def validation_loop(self, *, use_ema=False) -> None:
        agent = self.agent if not use_ema else self.ema.agent
        proc_name = f"validate{'-ema' if use_ema else ''}"

        self.log_aggregator.reset()
        agent.eval()
        for step, batch in enumerate(
            progress := progbar(
                self.test_dl,
                total=min(len(self.test_dl), self.cfg.val.max_steps),
                desc=f"epoch {self.state.epoch} ({proc_name})",
                leave=False,
            )
        ):
            if step == self.cfg.val.max_steps:
                progress.close()
                break

            metric = self.validation_step(batch, agent)

            progress.set_postfix(**metric.summary_metrics())
            self.log_aggregator.log(metric)

        logger.info("%s@epoch:%d", proc_name, self.state.epoch)

        self.log_aggregator.all_gather()
        if self.accelerator.is_main_process:
            self.accelerator.log(
                validate_logs := self.log_aggregator.collate(f"{proc_name}/"),
                step=self.state.global_step,
            )
            print(tabulate(validate_logs.items()))
        self.log_aggregator.reset()

    @torch.inference_mode()
    def evaluation_loop(self, dataloader, max_steps, split, *, use_ema=False) -> None:
        agent = self.agent if not use_ema else self.ema.agent
        proc_name = f"eval-{split}{'-ema' if use_ema else ''}"

        self.log_aggregator.reset()
        agent.eval()
        for step, batch in enumerate(
            progress := progbar(
                dataloader,
                total=min(len(dataloader), max_steps),
                desc=f"epoch {self.state.epoch} ({proc_name})",
                leave=False,
            )
        ):
            if step == max_steps:
                progress.close()
                break

            metric, trajectory = self.evaluation_step(batch, agent)

            progress.set_postfix(**metric.summary_metrics())
            self.log_aggregator.log(metric)

        logger.info("%s@epoch:%d", proc_name, self.state.epoch)

        self.log_aggregator.all_gather()
        if self.cfg.vis_metric_key is not None and self.accelerator.is_main_process:
            eval_logs = self.log_aggregator.collate(f"{proc_name}/", reset=False)
            print(tabulate(eval_logs.items()))
            eval_logs.update(self.log_aggregator.create_vis_logs(self.cfg.vis_metric_key))
            self.accelerator.log(eval_logs, step=self.state.global_step)
        self.log_aggregator.reset()

    @torch.inference_mode()
    def custom_evaluation(self, *, use_ema=False) -> None:
        if not (
            self.custom_evaluator
            and hasattr(self.cfg, "custom_eval")
            and self.cfg.custom_eval
        ):
            return

        if self.accelerator.is_main_process:
            custom_evaluators = (
                self.custom_evaluator
                if isinstance(self.custom_evaluator, list)
                else [self.custom_evaluator]
            )

            for idx, custom_evaluator in enumerate(custom_evaluators):
                proc_name = f"custom_eval{'-ema' if use_ema else ''}-{idx}"

                if callable(custom_evaluator):
                    custom_eval = custom_evaluator(
                        output_dir=f"{self.accelerator.project_dir}/{proc_name}/epoch-{self.state.epoch}",
                    )
                else:
                    custom_eval = custom_evaluator
                proc_name = f"{proc_name}-{custom_eval.eval_name}"

                logger.info("%s@epoch:%d", proc_name, self.state.epoch)
                with torch.no_grad():  # just double checking lol
                    agent = self.agent if not use_ema else self.ema.agent
                    agent.eval()
                    agent = self.accelerator.unwrap_model(agent)
                    agent.eval()
                    eval_infos, rich_media = custom_eval.evaluate(agent)

                rich_logs = handle_rich_media_for_logs(rich_media)
                eval_logs = {f"{k}-mean": v.mean() for k, v in eval_infos.items()}

                custom_eval_logs = {
                    f"{proc_name}/{k}": v for (k, v) in ({**eval_logs, **rich_logs}).items()
                }

                self.accelerator.log(custom_eval_logs, step=self.state.global_step)
                print(tabulate(custom_eval_logs.items()))

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

            metric = self.validation_step(batch, self.agent)
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

            metric, trajectory = self.evaluation_step(batch, self.agent)
            progress.set_postfix(**metric.summary_metrics())

    def train_step(self, batch):
        if self.log_train_timing:
            self.log_train_timing.before_forward()

        with self.accelerator.accumulate(self.agent):
            metric = self.agent("train", batch)
            loss = metric.loss()

            if self.log_train_timing:
                self.log_train_timing.after_forward()

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.agent.parameters(), 1.0)

            if self.log_train_timing:
                self.log_train_timing.after_backward()

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.manager.step(self.agent.parameters())

        if self.log_train_timing:
            self.log_train_timing.before_step_end()

        return metric, loss

    @staticmethod
    def validation_step(batch, agent):
        with torch.no_grad():
            metric = agent("validate", batch)
        return metric

    @staticmethod
    def evaluation_step(batch, agent):
        with torch.no_grad():
            metric, trajectory = agent("eval", batch, compare_gt=True)
        return metric, trajectory

    def resume_from_save(self, from_str):
        add_omegaconf_to_safe_globals()

        checkpoint_to_load = from_str

        self.accelerator.load_state(checkpoint_to_load)

        # TODO: fix this
        if self.ema:
            self.ema.manager.shadow_params = [
                param.to(self.accelerator.device) for param in self.agent.parameters()
            ]

        logger.info(
            "resuming from checkpoint@epoch:%d via %s", self.state.epoch, checkpoint_to_load
        )
        self.state.epoch += (
            1  # to start from the next epoch instead of redoing the current one
        )

    def save_checkpoint(self, mean_epoch_loss):
        self.accelerator.wait_for_everyone()

        resume_path = f"{self.accelerator.project_dir}/resume"
        checkpoint_path = f"{self.accelerator.project_dir}/checkpoint_{self.state.epoch}"
        last_checkpoint_path = f"{self.accelerator.project_dir}/last_checkpoint"
        best_checkpoint_path = f"{self.accelerator.project_dir}/best_checkpoint"

        # save resume
        shutil.rmtree(resume_path, ignore_errors=True)
        logger.info("resume states @epoch:%d saving to %s", self.state.epoch, resume_path)
        self.accelerator.save_state(resume_path)

        # save checkpoint
        if self.accelerator.is_main_process:
            Path(checkpoint_path).mkdir(parents=True)
        save_model_to_safetensors(
            self.accelerator, self.agent, f"{checkpoint_path}/model.safetensors"
        )
        if self.ema:
            save_model_to_safetensors(
                self.accelerator, self.ema.agent, f"{checkpoint_path}/ema_model.safetensors"
            )
        if self.accelerator.is_main_process:
            # keep non-accelerate ops on main process only
            save_agent_config(checkpoint_path, self.whole_cfg.agent.agent)

            Path(last_checkpoint_path).unlink(missing_ok=True)
            Path(last_checkpoint_path).symlink_to(checkpoint_path)
            logger.info("checkpoint@epoch:%d saved to %s", self.state.epoch, checkpoint_path)

            if mean_epoch_loss < self.state.best_mean_train_epoch_loss:
                self.state.best_mean_train_epoch_loss = mean_epoch_loss
                Path(best_checkpoint_path).unlink(missing_ok=True)
                Path(best_checkpoint_path).symlink_to(checkpoint_path)
                logger.info(
                    "best checkpoint@epoch:%d with loss %.4f saved to %s",
                    self.state.epoch,
                    self.state.best_mean_train_epoch_loss,
                    best_checkpoint_path,
                )

    def every_n_after_train(self, every_n_conf: EveryNConfig):
        if not every_n_conf:
            return False

        if "every_n_epochs" in every_n_conf:
            return self.every_n_epochs_after_train(**every_n_conf)
        if "every_n_global_steps" in every_n_conf:
            return self.every_n_global_steps_after_train(**every_n_conf)
        return False

    def every_n_epochs_after_train(self, *, every_n_epochs, skip_first_epochs=0, **_):
        if self.state.epoch < skip_first_epochs:
            return False

        return (
            (self.state.epoch + 1) % every_n_epochs == 0
            or (self.state.epoch + 1) == self.cfg.num_epochs  # or last epoch
        )

    def every_n_global_steps_after_train(
        self, *, every_n_global_steps, skip_first_global_steps=0, **_
    ):
        if self.state.global_step < skip_first_global_steps:
            return False

        step_lower_bound = self.state.global_step - self.len_train_loop
        step_upper_bound = self.state.global_step
        if step_lower_bound < 0:
            raise ValueError(
                "step_lower_bound is negative, which cannot happen as long as we train first"
            )
        lower_quotient = step_lower_bound // every_n_global_steps
        upper_quotient = step_upper_bound // every_n_global_steps

        if lower_quotient != upper_quotient:  # we crossed a multiple of n during training
            return True

        return self.state.epoch + 1 == self.cfg.num_epochs

    def every_n_global_steps(self, n, *, curr_train_step=0):
        return bool(n) and (
            (self.state.global_step + 1) % n == 0
            or (curr_train_step + 1) == self.len_train_loop  # or end of epoch
        )

    def free_memory(self):
        self.accelerator.wait_for_everyone()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        self.accelerator.wait_for_everyone()
