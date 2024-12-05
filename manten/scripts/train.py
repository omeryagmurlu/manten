from dataclasses import dataclass, field
import hydra
import os
from omegaconf import OmegaConf
from tabulate import tabulate
import torch

from accelerate.utils import set_seed
from manten.agents.base_agent import AgentMode
from manten.utils.file_utils import write_json
from manten.utils.progbar import progbar
from manten.utils.log_collator import LogCollator
from manten.utils.root import root
from manten.utils.logging import get_logger

logger = get_logger(__name__)


def _validator_factory(compute_logs):
    def validate(
        dataloader, agent, max_steps=float("inf"), log_col_kwargs={}, progbar_kwargs={}
    ):
        agent.eval()
        logs = LogCollator(**log_col_kwargs)
        for step, batch in enumerate(
            progress := progbar(
                dataloader,
                total=min(len(dataloader), max_steps),
                **progbar_kwargs,
            )
        ):
            if step == max_steps:
                progress.close()
                break
            agent.reset()
            compute_logs(agent, batch, logs, step, progress)
        return logs

    return validate


def eval_factory():
    def compute_logs(agent, batch, logs, step, progress):
        with torch.inference_mode():
            trajectory, metric = agent(AgentMode.EVAL, batch, compare_gt=True)

        progress.set_postfix(**metric.summary_metrics())
        logs.log(metric.metrics())

    return _validator_factory(compute_logs)


def test_factory():
    def compute_logs(agent, batch, logs, step, progress):
        with torch.inference_mode():
            metric = agent(AgentMode.TEST, batch)

        progress.set_postfix(**metric.summary_metrics())
        logs.log(metric.metrics())

    return _validator_factory(compute_logs)


def sanity_check(dataloader, agent, sanity_check_steps=5):
    test_sanity_check = test_factory()
    eval_sanity_check = eval_factory()

    test_sanity_check(
        dataloader,
        agent,
        max_steps=sanity_check_steps,
        progbar_kwargs=dict(desc="sanity check (test)"),
    )
    eval_sanity_check(
        dataloader,
        agent,
        max_steps=sanity_check_steps,
        progbar_kwargs=dict(desc="sanity check (eval)"),
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


def train(cfg, accelerator, agent, train_dl, test_dl, optimizer, lr_scheduler):
    test = test_factory()
    eval = eval_factory()

    def every_n_epochs(n):
        return bool(n) and ((state.epoch + 1) % n == 0 or state.epoch == cfg.num_epochs - 1)

    def every_n_global_steps(n):
        return bool(n) and ((state.global_step + 1) % n == 0 or (step + 1) == len_train_dl)

    ism = accelerator.is_main_process
    len_train_dl = len(train_dl)

    state = TrainState()
    accelerator.register_for_checkpointing(state)

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_save:
        # maybe later automatic loading of best checkpoint
        checkpoint_to_load = cfg.resume_from_save

        accelerator.load_state(checkpoint_to_load)
        logger.info(f"resuming from checkpoint@epoch:{state.epoch} via {checkpoint_to_load}")
        state.epoch += 1  # to start from the next epoch instead of redoing the current one

    logger.info("starting training")
    logs = LogCollator(reductions=["mean"])
    for state.epoch in range(state.epoch, cfg.num_epochs):
        agent.train()
        for step, batch in enumerate(
            progress := progbar(train_dl, desc=f"epoch {state.epoch}")
        ):
            agent.reset()
            # accelerate also handles disabling synchronisation
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
            if ism and every_n_global_steps(cfg.log_every_n_steps):
                upl = logs.log_collate(metric.metrics(), "train/")
                upl.update(overview_logs)
                upl.update({"epoch": state.epoch})
                accelerator.log(upl, step=state.global_step)

            state.global_step += 1

        if ism and every_n_epochs(cfg.test_every_n_epochs):
            test_logs = test(
                test_dl,
                agent,
                max_steps=cfg.test_ene_max_steps,
                progbar_kwargs=dict(desc=f"epoch {state.epoch} (test)", leave=False),
                log_col_kwargs=dict(reductions=logs.reductions),
            )
            logger.info(f"test@epoch:{state.epoch}")
            accelerator.log(test_logs := test_logs.collate("test/"), step=state.global_step)
            print(tabulate(test_logs.items()))

        if ism and every_n_epochs(cfg.eval_every_n_epochs):
            eval_logs = eval(
                test_dl,
                agent,
                max_steps=cfg.eval_ene_max_steps,
                progbar_kwargs=dict(desc=f"epoch {state.epoch} (eval)", leave=False),
                log_col_kwargs=dict(reductions=logs.reductions),
            )
            logger.info(f"evaluation@epoch:{state.epoch}")
            accelerator.log(eval_logs := eval_logs.collate("eval/"), step=state.global_step)
            print(tabulate(eval_logs.items()))

        if ism and every_n_epochs(cfg.save_every_n_epochs):
            accelerator.wait_for_everyone()
            checkpoint_path = f"{accelerator.project_dir}/checkpoint_{state.epoch}"
            accelerator.save_state(checkpoint_path)
            logger.info(f"checkpoint@epoch:{state.epoch} saved to {checkpoint_path}")

            if overview_logs["loss"] < state.best_loss:
                state.best_loss = overview_logs["loss"]
                best_checkpoint_path = f"{accelerator.project_dir}/best_checkpoint"
                accelerator.save_state(best_checkpoint_path)
                write_json(
                    f"{accelerator.project_dir}/best_checkpoint/overview_logs.json",
                    overview_logs,
                )
                logger.info(
                    f"best checkpoint@epoch:{state.epoch} with loss {state.best_loss:.4f} saved to {best_checkpoint_path}"
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

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    accelerator = hydra.utils.instantiate(
        cfg.accelerator, project_dir=output_dir + "/accelerate"
    )

    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(output_dir + "/accelerate", exist_ok=True)
            os.makedirs(output_dir + "/tracker", exist_ok=True)

    # accelerator already handles is_main_process for trackers
    init_dict = {**OmegaConf.to_container(cfg.accelerator_init_trackers)}
    if "wandb" in init_dict["init_kwargs"]:
        init_dict["init_kwargs"]["wandb"]["dir"] = output_dir + "/tracker"
    accelerator.init_trackers(**init_dict)

    # same for logging
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', None)}")

    agent = hydra.utils.instantiate(cfg.agent)

    optimizer_configurator = hydra.utils.instantiate(cfg.optimizer_configurator, agent=agent)
    optimizer = hydra.utils.instantiate(
        cfg.optimizer, OmegaConf.to_container(optimizer_configurator.get_grouped_params())
    )

    # TODO: accelerate doesn't scale lr: https://huggingface.co/docs/accelerate/en/concept_guides/performance#learning-rates
    lr_scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer)

    datamodule = hydra.utils.instantiate(cfg.datamodule)

    train_dataloader = datamodule.create_train_dataloader()
    test_dataloader = datamodule.create_test_dataloader()

    (agent, optimizer, train_dataloader, test_dataloader, lr_scheduler) = accelerator.prepare(
        agent, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    if bool(cfg.training.sanity_check):
        sanity_check(test_dataloader, agent, cfg.training.sanity_check)

    train(
        cfg.training,
        accelerator,
        agent,
        train_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
    )

    # potentially online eval?


if __name__ == "__main__":
    if True:
        from manten.utils.debug_utils import monkeypatch_tensor_shape

        monkeypatch_tensor_shape()
    main()
