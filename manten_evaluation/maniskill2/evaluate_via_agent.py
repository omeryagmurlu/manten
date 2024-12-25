from logging import getLogger

import hydra
import numpy as np

from manten_evaluation.maniskill2.utils_evaluation import evaluate_via_agent, make_eval_envs

logger = getLogger(__name__)


class ManiskillEvaluation:
    def __init__(
        self,
        *,
        output_dir,
        agent_wrapper,
        env_id,
        num_eval_episodes,
        sim_backend,
        num_envs,
        env_kwargs,
        save_video=False,
        wrappers=None,
        progress_bar=True,
    ):
        if wrappers is None:
            wrappers = []

        self.agent_wrapper = agent_wrapper
        self.output_dir = output_dir
        self.num_eval_episodes = num_eval_episodes
        self.sim_backend = sim_backend
        self.num_envs = num_envs
        self.env_kwargs = env_kwargs
        self.save_video = save_video
        self.wrappers = wrappers
        self.env_id = env_id
        self.progress_bar = progress_bar

    def evaluate(self, agent):
        agent = self.agent_wrapper(agent=agent)

        envs = make_eval_envs(
            env_id=self.env_id,
            num_envs=self.num_envs,
            sim_backend=self.sim_backend,
            env_kwargs=self.env_kwargs,
            video_dir=self.output_dir,
            wrappers=self.wrappers,
            save_video=self.save_video,
        )

        outputs = evaluate_via_agent(
            agent=agent,
            envs=envs,
            num_eval_episodes=self.num_eval_episodes,
            sim_backend=self.sim_backend,
            progress_bar=self.progress_bar,
        )

        envs.close()

        return {k: np.array(v) for k, v in outputs.items()}

    @property
    def eval_name(self):
        return f"eval-maniskill-{self.env_id}"


@hydra.main(config_path="configs", config_name="evaluate_via_agent")
def main(cfg):
    import torch
    from accelerate.utils import set_seed
    from omegaconf import OmegaConf
    from tabulate import tabulate

    from manten.utils.utils_checkpointing import load_agent

    set_seed(cfg.seed, deterministic=cfg.deterministic)
    if cfg.deterministic:
        # also need to handle this but nvm for now: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        torch.backends.cudnn.benchmark = False

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    agent_creation = OmegaConf.to_object(cfg.agent_creation)
    agent = load_agent(
        **agent_creation,
        device=cfg.device,
    )
    agent.eval()

    env_creation = OmegaConf.to_object(cfg.env_creation)
    evaluator = ManiskillEvaluation(
        output_dir=output_dir,
        agent_wrapper=hydra.utils.instantiate(cfg.agent_wrapper, device=cfg.device),
        env_id=env_creation["env_id"],
        num_eval_episodes=cfg.num_eval_episodes,
        sim_backend=env_creation["sim_backend"],
        num_envs=env_creation["num_envs"],
        env_kwargs=env_creation["env_kwargs"],
        save_video=env_creation["save_video"],
        wrappers=env_creation["wrappers"],
    )

    logger.info("Evaluating on %s", evaluator.eval_name)
    results = evaluator.evaluate(agent)
    results = {f"{k}-mean": v.mean() for k, v in results.items()}
    print(tabulate(results.items()))


if __name__ == "__main__":
    main()
