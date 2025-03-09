from http.server import BaseHTTPRequestHandler, HTTPServer
from logging import getLogger

import hydra
import torch
from accelerate.utils import set_seed

from manten.utils.utils_checkpointing import load_agent
from manten.utils.utils_serialization import MantenAgentSerialization

logger = getLogger(__name__)


class AgentHandler(BaseHTTPRequestHandler):
    create_agent = None
    agent = None

    def do_POST(self):  # noqa: N802
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        request = MantenAgentSerialization.deserialize(post_data)

        method = request.get("method")

        try:
            if method == "__shutdown__":
                AgentHandler._destroy_agent()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(MantenAgentSerialization.serialize({"result": "shutdown"}))
                raise KeyboardInterrupt  # To stop the server  # noqa: TRY301

            if method == "__init__":
                AgentHandler.agent = AgentHandler.create_agent()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    MantenAgentSerialization.serialize({"result": "agent created"})
                )
            else:
                # pass these to the agent
                agent = AgentHandler.agent
                args = request.get("args", [])
                kwargs = request.get("kwargs", {})
                if method == "__call__":
                    result = agent(*args, **kwargs)
                else:
                    result = getattr(agent, method)(*args, **kwargs)

                self.send_response(200)
                self.end_headers()
                self.wfile.write(MantenAgentSerialization.serialize({"result": result}))
        except Exception:
            logger.exception("Error handling request")
            AgentHandler._destroy_agent()
            self.send_response(500)
            self.end_headers()
            self.wfile.write(
                MantenAgentSerialization.serialize({"error": "there was a problem"})
            )

    @staticmethod
    def _destroy_agent():
        del AgentHandler.agent
        AgentHandler.agent = None
        AgentHandler.clear_cuda_cache()
        logger.info("agent destroyed")

    @staticmethod
    def clear_cuda_cache():
        if torch.cuda.is_available():
            # Empty CUDA cache
            torch.cuda.empty_cache()
            # Force garbage collection
            import gc

            gc.collect()
            # Log memory stats
            for i in range(torch.cuda.device_count()):
                memory_stats = torch.cuda.memory_stats(i)
                allocated = memory_stats.get("allocated_bytes.all.current", 0) / (1024**3)
                reserved = memory_stats.get("reserved_bytes.all.current", 0) / (1024**3)
                logger.info(
                    f"GPU {i} Memory: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
                )


def start_server(create_agent, host="localhost", port=6000):
    AgentHandler.create_agent = create_agent
    AgentHandler.agent = None
    server = HTTPServer((host, port), AgentHandler)
    logger.info(f"starting server at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down server")
        server.server_close()


@hydra.main(config_path="../../configs", config_name="agent_proxy")
def main(cfg):
    def create_agent():
        logger.info("loading agent")

        set_seed(cfg.seed, deterministic=cfg.deterministic)
        if cfg.deterministic:
            # also need to handle this but nvm for now: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
            torch.backends.cudnn.benchmark = False

        agent = load_agent(
            cfg.testing.train_folder,
            cfg.testing.checkpoint,
            cfg.testing.agent_override,
            no_checkpoint_mode="best",
        )
        agent = agent.to(cfg.device)
        agent.eval()

        if cfg.agent_wrapper is not None:
            agent = hydra.utils.instantiate(cfg.agent_wrapper, agent=agent)

        return agent

    # create_agent()
    start_server(create_agent, cfg.host, cfg.port)


if __name__ == "__main__":
    main()
