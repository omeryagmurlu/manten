from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from logging import getLogger

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate.utils import set_seed
from optree import tree_map

import manten.utils.dda_pytorch3d_transforms as pytorch3d_transforms
from manten.utils.utils_config import load_agent
from manten.utils.utils_serialization import MantenAgentSerialization

logger = getLogger(__name__)


class TDDAEvalUtils:
    @staticmethod
    def prepare_visual_states(obs: dict[str, dict[str, any]]):
        rgb_static = obs["rgb_obs"]["rgb_static"]
        rgb_gripper = obs["rgb_obs"]["rgb_gripper"]
        pcd_gripper = obs["pcd_obs"]["pcd_gripper"]

        # map RGB to [0, 1]
        rgb_static = rgb_static / 255.0
        rgb_gripper = rgb_gripper / 255.0

        # interpolate the gripper RGB and PCD to the same size as the static RGB
        h, w = rgb_static.shape[:2]
        rgb_gripper = (
            F.interpolate(
                torch.as_tensor(rgb_gripper).permute(2, 0, 1).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .numpy()
        )
        pcd_gripper = (
            F.interpolate(
                torch.as_tensor(pcd_gripper).permute(2, 0, 1).unsqueeze(0),
                size=(h, w),
                mode="nearest",
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .numpy()
        )

        obs["rgb_obs"]["rgb_static"] = rgb_static
        obs["rgb_obs"]["rgb_gripper"] = rgb_gripper

        obs["pcd_obs"]["pcd_gripper"] = pcd_gripper

        return obs

    @staticmethod
    def prepare_proprio_states(obs: dict[str, dict[str, any]]):
        proprio = np.concatenate(
            [
                obs["robot_obs"][:3],
                TDDAEvalUtils.convert_rotation(obs["robot_obs"][3:6]),
                (obs["robot_obs"][[-1]] + 1) / 2,
            ],
            axis=-1,
        )

        if "proprio" not in obs:
            obs["proprio"] = np.stack([proprio] * 3, axis=0)
        else:
            obs["proprio"] = np.concatenate([obs["proprio"][1:], proprio[None]], axis=0)

        return obs

    @staticmethod
    def get_text_encoder(text_encoder, text_max_length):
        def load_model(encoder) -> transformers.PreTrainedModel:
            if encoder == "bert":
                model = transformers.BertModel.from_pretrained("bert-base-uncased")
            elif encoder == "clip":
                model = transformers.CLIPTextModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
            else:
                raise ValueError(f"Unexpected encoder {encoder}")
            if not isinstance(model, transformers.PreTrainedModel):
                raise TypeError(f"Unexpected encoder {encoder}")
            return model

        def load_tokenizer(encoder) -> transformers.PreTrainedTokenizer:
            if encoder == "bert":
                tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
            elif encoder == "clip":
                tokenizer = transformers.CLIPTokenizer.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
            else:
                raise ValueError(f"Unexpected encoder {encoder}")
            if not isinstance(tokenizer, transformers.PreTrainedTokenizer):
                raise TypeError(f"Unexpected encoder {encoder}")
            return tokenizer

        tokenizer = load_tokenizer(text_encoder)
        tokenizer.model_max_length = text_max_length

        model = load_model(text_encoder)

        return tokenizer, model

    @staticmethod
    def convert_action(trajectory):
        """Convert [position, rotation, openness] to the same format as Calvin

        Args:
            trajectory: a torch.Tensor or np.ndarray of shape [bs, traj_len, 8]
                - position: absolute [x, y, z] in the world coordinates
                - rotation: absolute quarternion in the world coordinates
                - openness: [0, 1]

        Returns:
            trajectory: a torch.Tensor or np.ndarray of shape [bs, traj_len, 8]
                - position: absolute [x, y, z] in the world coordinates
                - rotation: absolute 'XYZ' Euler angles in the world coordinates
                - openness: [-1, 1]
        """
        assert trajectory.shape[-1] == 8  # noqa: PLR2004
        position, rotation, openness = (
            trajectory[..., :3],
            trajectory[..., 3:7],
            trajectory[..., -1:],
        )
        position = position.data.cpu().numpy()
        _rot = TDDAEvalUtils.convert_quaternion_to_euler(rotation)
        # pytorch3d.transforms does not deal with Gumbel lock, the conversion
        # of some rotation matrix results in nan values.  We usepybullet's
        # implementation in this case.

        # for noqa: I directly copied this line, and I don't want to mix in
        # the pybullet dependency, so I'm happy that this branch will never
        # be entered. Apparently this wasn't a problem in the original 3dda.
        if (_rot != _rot).any():  # noqa: PLR0124
            raise ValueError("Nan values in rotation matrix. Requires pybullet.")
            # # Pybullet has different convention of Quaternion.
            # _rot_shape = list(rotation.shape)[:-1] + [3]
            # _rot = rotation.reshape(-1, 4).data.cpu().numpy()
            # rotation = np.array(
            #     [pybullet.getEulerFromQuaternion([r[-1], r[0], r[1], r[2]]) for r in _rot]
            # ).reshape(_rot_shape)
        else:  # noqa: RET506
            rotation = _rot
        openness = (2 * (openness >= 0.5).long() - 1).data.cpu().numpy()  # noqa: PLR2004

        trajectory = np.concatenate([position, rotation, openness], axis=-1)
        return trajectory

    @staticmethod
    def convert_quaternion_to_euler(quat):
        """Convert Euler angles to Quarternion"""
        quat = torch.as_tensor(quat)
        mat = pytorch3d_transforms.quaternion_to_matrix(quat)
        rot = pytorch3d_transforms.matrix_to_euler_angles(mat, "XYZ")
        rot = rot.data.cpu().numpy()

        return rot

    @staticmethod
    def convert_rotation(rot):
        """Convert Euler angles to Quarternion"""
        rot = torch.as_tensor(rot)
        mat = pytorch3d_transforms.euler_angles_to_matrix(rot, "XYZ")
        quat = pytorch3d_transforms.matrix_to_quaternion(mat)
        quat = quat.numpy()

        return quat

    @staticmethod
    def relative_to_absolute(
        action,
        proprio,
        max_rel_pos=1.0,
        max_rel_orn=1.0,
        magic_scaling_factor_pos=1,
        magic_scaling_factor_orn=1,
    ):
        assert action.shape[-1] == 7  # noqa: PLR2004
        assert proprio.shape[-1] == 7  # noqa: PLR2004

        rel_pos, rel_orn, gripper = np.split(action, [3, 6], axis=-1)
        rel_pos *= max_rel_pos * magic_scaling_factor_pos
        rel_orn *= max_rel_orn * magic_scaling_factor_orn

        pos_proprio, orn_proprio = proprio[..., :3], proprio[..., 3:6]

        target_pos = pos_proprio + rel_pos
        target_orn = orn_proprio + rel_orn
        return np.concatenate([target_pos, target_orn, gripper], axis=-1)


@dataclass
class TDDAAgentWrapperState:
    agent: any
    device: any
    execute_len: int
    interpolation_length: int
    action_dim: int
    text_tokenizer: any
    text_model: any
    calvin_gripper_loc_bounds: any
    current_step: int = 0
    current_trajectory: int | None = None


class TDDAAgentWrapper:
    def __init__(
        self,
        agent,
        text_encoder,
        text_max_length,
        action_dim=7,
        calvin_gripper_loc_bounds=None,
        interpolation_length=20,
        execute_len=20,
    ):
        self._ws = TDDAAgentWrapperState(
            agent=agent,
            device=next(agent.parameters()).device,
            execute_len=execute_len,
            interpolation_length=interpolation_length,
            current_step=0,
            current_trajectory=None,
            action_dim=action_dim,
            calvin_gripper_loc_bounds=calvin_gripper_loc_bounds,
            text_tokenizer=None,
            text_model=None,
        )
        self._ws.text_tokenizer, self._ws.text_model = TDDAEvalUtils.get_text_encoder(
            text_encoder, text_max_length
        )

    def __getattr__(self, attr):
        return getattr(self._ws.agent, attr)

    def reset(self):
        self._ws.current_step = 0
        self._ws.agent.reset()
        self._ws.text_model.eval()
        self._ws.text_model = self._ws.text_model.to(self._ws.device)

    def _encode_instruction(self, lang_annotation):
        """Encode string lang_annotation to latent embeddings.

        Args:
            lang_annotation: a string of lang_annotation
            device: a string of device

        Returns:
            pred: a tensor of latent embeddings of shape (text_max_length, 512)
        """
        device = self._ws.device

        instr = lang_annotation + "."
        tokens = self._ws.text_tokenizer(instr, padding="max_length")["input_ids"]

        tokens = torch.tensor(tokens).to(device)
        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = self._ws.text_model(tokens).last_hidden_state

        return pred

    def step_agent(self, obs, lang_annotation):
        """This return `interpolation_length` number of actions as a trajectory."""
        instruction = self._encode_instruction(lang_annotation)
        obs = TDDAEvalUtils.prepare_visual_states(obs)
        obs = TDDAEvalUtils.prepare_proprio_states(obs)

        trajectory_mask = torch.full([1, self._ws.interpolation_length - 1], fill_value=False)
        fake_trajectory = torch.full(
            [1, self._ws.interpolation_length - 1, self._ws.action_dim], 0
        )
        rgbs = np.stack(
            [obs["rgb_obs"]["rgb_static"], obs["rgb_obs"]["rgb_gripper"]], axis=0
        ).transpose(0, 3, 1, 2)  # [ncam, 3, H, W]
        pcds = np.stack(
            [obs["pcd_obs"]["pcd_static"], obs["pcd_obs"]["pcd_gripper"]], axis=0
        ).transpose(0, 3, 1, 2)  # [ncam, 3, H, W]

        rgbs = torch.as_tensor(rgbs).unsqueeze(0)
        pcds = torch.as_tensor(pcds).unsqueeze(0)

        # Crop the images.  See Line 165-166 in datasets/dataset_calvin.py
        rgbs = rgbs[..., 20:180, 20:180]
        pcds = pcds[..., 20:180, 20:180]

        # history of actions
        gripper = torch.as_tensor(obs["proprio"]).unsqueeze(0)
        gripper = gripper[:, -self._ws.agent.num_history :]

        # if self._ws.input_mode == "3d":
        #     has_3d = torch.tensor([[1]], device=rgbs.device)
        # elif self._ws.input_mode == "2d":
        #     has_3d = torch.tensor([[0]], device=rgbs.device)
        #     pcds = torch.zeros_like(pcds)
        # else:
        #     raise ValueError(f"Unexpected input mode {self._ws.input_mode}")

        batch = {
            "trajectory": fake_trajectory.float(),
            "trajectory_mask": trajectory_mask.float(),
            "rgbs": rgbs.float(),
            "pcds": pcds.float(),
            "instr": instruction.float(),
            "curr_gripper_history": gripper[..., :7].float(),
            # has_3d=has_3d,
        }

        batch = tree_map(lambda x: x.to(self._ws.device), batch)
        _metric, trajectory = self._ws.agent("eval", batch)

        # Convert quaternion to Euler angles
        trajectory = TDDAEvalUtils.convert_action(trajectory)

        if self._ws.agent.relative:
            # Convert quaternion to Euler angles
            gripper = TDDAEvalUtils.convert_action(gripper[:, [-1], :])
            # Convert relative action to absolute action
            trajectory = TDDAEvalUtils.relative_to_absolute(trajectory, gripper)

        # Bound final action by CALVIN statistics
        if self._ws.calvin_gripper_loc_bounds is not None:
            trajectory[:, :, :3] = np.clip(
                trajectory[:, :, :3],
                a_min=self._ws.calvin_gripper_loc_bounds[0].reshape(1, 1, 3),
                a_max=self._ws.calvin_gripper_loc_bounds[1].reshape(1, 1, 3),
            )

        return trajectory

    def step(self, obs, lang_annotation):
        if self._ws.current_step % self._ws.execute_len == 0:
            self._ws.current_trajectory = self.step_agent(obs, lang_annotation)

        curr_traj = self._ws.current_trajectory[
            0, self._ws.current_step % self._ws.execute_len
        ]
        self._ws.current_step += 1
        # fkin calvin interprets the action different depending on whether its (3,3,1) or (7)
        # for no fkin reason / I spent 3 days debugging this, thought I was going insane with my model
        action = [  # I hate my life
            curr_traj[:3],
            curr_traj[3:6],
            curr_traj[[6]],
        ]
        return action


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
                    f"GPU {i} Memory: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"  # noqa: G004
                )


def start_server(create_agent, host="localhost", port=6000):
    AgentHandler.create_agent = create_agent
    AgentHandler.agent = None
    server = HTTPServer((host, port), AgentHandler)
    logger.info(f"starting server at http://{host}:{port}")  # noqa: G004
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

        agent = TDDAAgentWrapper(agent, text_encoder="clip", text_max_length=53)

        return agent

    # create_agent()
    start_server(create_agent, cfg.host, cfg.port)


if __name__ == "__main__":
    main()
