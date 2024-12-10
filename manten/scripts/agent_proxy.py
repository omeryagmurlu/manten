import gc
from logging import getLogger
from multiprocessing.connection import Listener

import hydra
import torch
from accelerate.utils import set_seed

from manten.utils.utils_config import load_agent
from manten.utils.utils_root import root

logger = getLogger(__name__)


@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="agent_proxy")
def main(cfg):
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    if cfg.deterministic:
        # also need to handle this but nvm for now: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        torch.backends.cudnn.benchmark = False

    with Listener(
        tuple(cfg.listener.address), authkey=bytes(cfg.listener.authkey, "utf-8")
    ) as listener:
        print(f"listening for connections on {cfg.listener.address}")
        while True:
            with listener.accept() as conn:
                logger.info("connection accepted, creating agent")
                agent = load_agent(
                    cfg.testing.train_folder,
                    cfg.testing.checkpoint,
                    cfg.testing.agent_override,
                    no_checkpoint_mode="best",
                )

                agent = agent.to(cfg.device)
                while True:
                    try:
                        method, args, kwargs = conn.recv()
                        logger.info(
                            f"received method {method} with args {args} and kwargs {kwargs}"  # noqa: G004
                        )
                        if method == "__shutdown__":
                            break
                        result = getattr(agent, method)(*args, **kwargs)  # Call the method
                        conn.send(result)  # Send back the result
                    except EOFError:
                        break

                logger.info(
                    "connection closed, shutting down agent and waiting for new connection"
                )
                agent.to("cpu")
                del agent
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
