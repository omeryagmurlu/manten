import numpy as np
import torch
import transformers

from manten.utils.dda_utils_with_calvin import relative_to_absolute
from manten_evaluation.calvin.online_evaluation_calvin.evaluate_utils import convert_action


def get_text_encoder(text_encoder, text_max_length):
    def load_model(encoder) -> transformers.PreTrainedModel:
        if encoder == "bert":
            model = transformers.BertModel.from_pretrained("bert-base-uncased")
        elif encoder == "clip":
            model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
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


class TDDACalvinAdapter:
    def __init__(
        self,
        interpolation_length,
        action_dim,
        num_history,
        te_device,
        relative_action,
        calvin_gripper_loc_bounds,
        text_encoder,
        text_max_length,
    ):
        self.interpolation_length = interpolation_length
        self.action_dim = action_dim
        self.num_history = num_history

        self.te_device = te_device
        self.relative_action = relative_action
        self.calvin_gripper_loc_bounds = calvin_gripper_loc_bounds
        self.text_tokenizer, self.text_model = get_text_encoder(text_encoder, text_max_length)

    def reset(self):
        self.text_model.eval()
        self.text_model = self.text_model.to(self.te_device)

    def encode_instruction(self, instruction, device=None):
        """Encode string instruction to latent embeddings.

        Args:
            instruction: a string of instruction
            device: a string of device

        Returns:
            pred: a tensor of latent embeddings of shape (text_max_length, 512)
        """
        device = device if device is not None else self.te_device

        instr = instruction + "."
        tokens = self.text_tokenizer(instr, padding="max_length")["input_ids"]

        tokens = torch.tensor(tokens).to(device)
        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = self.text_model(tokens).last_hidden_state

        return pred

    def pre_step(self, obs, instruction):
        # Organize inputs
        trajectory_mask = torch.full([1, self.interpolation_length - 1], fill_value=False)
        fake_trajectory = torch.full([1, self.interpolation_length - 1, self.action_dim], 0)
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
        gripper = gripper[:, -self.num_history :]

        # if self.input_mode == "3d":
        #     has_3d = torch.tensor([[1]], device=rgbs.device)
        # elif self.input_mode == "2d":
        #     has_3d = torch.tensor([[0]], device=rgbs.device)
        #     pcds = torch.zeros_like(pcds)
        # else:
        #     raise ValueError(f"Unexpected input mode {self.input_mode}")

        batch = {
            "trajectory": fake_trajectory.float(),
            "trajectory_mask": trajectory_mask.float(),
            "rgbs": rgbs.float(),
            "pcds": pcds.float(),
            "instr": instruction.float(),
            "curr_gripper_history": gripper[..., :7].float(),
            # has_3d=has_3d,
        }

        return ("eval", batch), (gripper,)

    def post_step(self, agent_output, pre_step_output):
        _metric, trajectory = agent_output
        (gripper,) = pre_step_output

        # Convert quaternion to Euler angles
        trajectory = convert_action(trajectory)

        if self.relative_action:
            # Convert quaternion to Euler angles
            gripper = convert_action(gripper[:, [-1], :])
            # Convert relative action to absolute action
            trajectory = relative_to_absolute(trajectory, gripper)

        # Bound final action by CALVIN statistics
        if self.calvin_gripper_loc_bounds is not None:
            trajectory[:, :, :3] = np.clip(
                trajectory[:, :, :3],
                a_min=self.calvin_gripper_loc_bounds[0].reshape(1, 1, 3),
                a_max=self.calvin_gripper_loc_bounds[1].reshape(1, 1, 3),
            )
