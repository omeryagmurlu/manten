from typing import Protocol

import hydra
import torch
from optree import tree_map
from tabulate import tabulate

from manten.utils.log_aggregator import LogAggregator
from manten.utils.logging import get_logger
from manten.utils.progbar import progbar
from manten.utils.utils_checkpointing import load_agent
from manten.utils.utils_root import root

logger = get_logger(__name__)


def plot_2_traj(traj, batch, idx=0):
    import matplotlib.pyplot as plt

    tg = batch["trajectory"].cpu()
    t = traj.cpu()

    plt.subplot(111, projection="3d")
    plt.plot(t[idx, :, 0], t[idx, :, 1], t[idx, :, 2], color="blue")
    plt.plot(tg[idx, :, 0], tg[idx, :, 1], tg[idx, :, 2], color="orange")
    plt.show()
    plt.cla()
    plt.clf()


class TestConfig(Protocol):
    train_folder: str
    checkpoint: str | None
    agent_override: dict | None


def every_n_steps(n, step):
    return bool(n) and ((step + 1) % n == 0)


def single_step(agent, batch):
    with torch.inference_mode():
        trajectory, metric = agent("eval", batch, compare_gt=True)
    return metric, trajectory


def test_agent_with_data(
    agent, test_dl, *, device, max_steps=float("inf"), preview_log_every_n=0
):
    log_aggregator = LogAggregator()
    total = min(len(test_dl), max_steps)
    title = "eval (against test data)"
    agent.eval()
    for step, batch in enumerate(
        progress := progbar(
            test_dl,
            total=total,
            desc=title,
            leave=False,
        )
    ):
        if step == max_steps:
            progress.close()
            break

        batch = tree_map(lambda x: x.to(device), batch)  # noqa: PLW2901
        trajectory, metric = single_step(agent, batch)

        progress.set_postfix(**metric.summary_metrics())
        log_aggregator.log(metric)

        if every_n_steps(preview_log_every_n, step) or step + 1 == total:
            logger.info("%s:%d/%d", title, step + 1, total)
            logs = log_aggregator.collate("eval_against_data/", reset=False)
            print(tabulate(logs.items()))
    log_aggregator.reset()


def test(cfg):
    import torch
    from accelerate.utils import set_seed

    set_seed(cfg.seed, deterministic=cfg.deterministic)
    if cfg.deterministic:
        # also need to handle this but nvm for now: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        torch.backends.cudnn.benchmark = False

    datamodule = hydra.utils.instantiate(cfg.datamodule.datamodule)
    test_dataloader = datamodule.create_test_dataloader()

    # load agent
    agent = load_agent(
        cfg.testing.train_folder,
        cfg.testing.checkpoint,
        cfg.testing.agent_override,
        no_checkpoint_mode="best",
    )

    agent = agent.to(device := "cuda")

    # maybe init trackers?
    logger.info("starting testing against data")
    test_agent_with_data(agent, test_dataloader, device=device)


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="test_against_data"
)
def main(cfg):
    test(cfg)


if __name__ == "__main__":
    main()
