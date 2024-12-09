import einops
import torch

from manten.agents.base_agent import BaseAgent
from manten.agents.metrics.trajectory_metric import TrajectoryMetric, TrajectoryStats
from manten.utils.dda_utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    matrix_to_quaternion,
    normalise_quat,
    quaternion_to_matrix,
)

# TODO: noqa find a way to handle std input shape this is no way to live
# ruff: noqa: PLR2004


class RotationParametrization:
    def __init__(self, rotation_parametrization, quaternion_format):
        self._rotation_parametrization = rotation_parametrization
        self._quaternion_format = quaternion_format

    def convert_rot(self, signal):
        signal[..., 3:7] = normalise_quat(signal[..., 3:7])
        if self._rotation_parametrization == "6D":
            # The following code expects wxyz quaternion format!
            if self._quaternion_format == "xyzw":
                signal[..., 3:7] = signal[..., (6, 3, 4, 5)]
            rot = quaternion_to_matrix(signal[..., 3:7])
            res = signal[..., 7:] if signal.size(-1) > 7 else None
            if len(rot.shape) == 4:
                B, L, D1, D2 = rot.shape
                rot = rot.reshape(B * L, D1, D2)
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
                rot_6d = rot_6d.reshape(B, L, 6)
            else:
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
            signal = torch.cat([signal[..., :3], rot_6d], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
        return signal

    def unconvert_rot(self, signal):
        if self._rotation_parametrization != "6D":
            signal[:, :, 3:7] = normalise_quat(signal[:, :, 3:7])

        if self._rotation_parametrization == "6D":
            res = signal[..., 9:] if signal.size(-1) > 9 else None
            if len(signal.shape) == 3:
                B, L, _ = signal.shape
                rot = signal[..., 3:9].reshape(B * L, 6)
                mat = compute_rotation_matrix_from_ortho6d(rot)
                quat = matrix_to_quaternion(mat)
                quat = quat.reshape(B, L, 4)
            else:
                rot = signal[..., 3:9]
                mat = compute_rotation_matrix_from_ortho6d(rot)
                quat = matrix_to_quaternion(mat)
            signal = torch.cat([signal[..., :3], quat], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
            # The above code handled wxyz quaternion format!
            if self._quaternion_format == "xyzw":
                signal[..., 3:7] = signal[..., (4, 5, 6, 3)]
        return signal

    @property
    def rotation_dims(self):
        if "6D" in self._rotation_parametrization:
            return 6
        else:
            return 4


class PositionNormalization:
    def __init__(self, gripper_loc_bounds=None):
        self.gripper_loc_bounds = gripper_loc_bounds

    def normalize_pos(self, pos):
        if self.gripper_loc_bounds is None:
            return pos
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        if self.gripper_loc_bounds is None:
            return pos
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min


def convert2rel(pcd, curr_gripper):
    """Convert coordinate system relative to current gripper."""
    center = curr_gripper[:, -1, :3]  # (batch_size, 3)
    bs = center.shape[0]
    pcd = pcd - center.view(bs, 1, 3, 1, 1)
    curr_gripper = curr_gripper.clone()
    curr_gripper[..., :3] = curr_gripper[..., :3] - center.view(bs, 1, 3)
    return pcd, curr_gripper


class ThreeDDAAgent(BaseAgent):
    def __init__(
        self,
        *,
        position_noise_scheduler,
        rotation_noise_scheduler,
        rotation_parametrization,
        position_normalization,
        encoder,
        noise_model,
        num_history=0,
        relative=True,
        use_instruction=True,
        n_inference_steps=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.position_noise_scheduler = position_noise_scheduler
        self.rotation_noise_scheduler = rotation_noise_scheduler
        self.rotation_parametrization = rotation_parametrization
        self.position_normalization = position_normalization
        self.encoder = encoder
        self.noise_model = noise_model

        self.num_history = num_history
        self.relative = relative
        self.use_instruction = use_instruction
        self.n_inference_steps = n_inference_steps

    def train_step(self, batch):
        (gt_trajectory, gt_openness, _, _, conditions) = self.process_batch(batch)

        return self.conditional_diffusion_loss(
            trajectory=torch.cat((gt_trajectory, gt_openness), -1), conditions=conditions
        )

    @torch.no_grad()
    def validate_step(self, batch):
        (gt_trajectory, gt_openness, _, _, conditions) = self.process_batch(batch)

        assert torch.is_grad_enabled() is False

        return self.conditional_diffusion_loss(
            trajectory=torch.cat((gt_trajectory, gt_openness), -1), conditions=conditions
        )

    @torch.no_grad()
    def eval_step(self, batch, *, compare_gt=False):
        if compare_gt:
            if "trajectory" not in batch:
                raise ValueError("trajectory not found in batch")
            traj_len = batch["trajectory"].size(1) - 1
        else:
            traj_len = 20  # # this is hardcoded but eeeh

        (_, _, trajectory_mask, inputs, conditions) = self.process_batch(
            batch, is_evaluation_mode=True
        )

        B, _, D = inputs["curr_gripper"].shape
        # trajectory_shape = (B, trajectory_mask.size(1), D)

        trajectory_shape = (B, traj_len, D)
        trajectory_device = inputs["curr_gripper"].device
        trajectory_dtype = inputs["curr_gripper"].dtype

        pred_complete_traj = self.conditional_sample(
            conditions=conditions,
            shape=trajectory_shape,
            device=trajectory_device,
            dtype=trajectory_dtype,
        )

        pred_complete_traj = self.reverse_input_transformations(pred_complete_traj)
        pred_pos = pred_complete_traj[..., :3]
        pred_quat = pred_complete_traj[..., 3:7]
        pred_openness = pred_complete_traj[..., 7:8]

        if compare_gt:
            gt_traj = batch["trajectory"][:, 1:]
            traj_metric = TrajectoryMetric()
            traj_metric.feed(
                ground=(gt_traj[..., :3], gt_traj[..., 3:7], gt_traj[..., 7:8]),
                prediction=(pred_pos, pred_quat, pred_openness),
            )
        else:
            traj_metric = TrajectoryStats()
            traj_metric.feed(stats=(pred_pos, pred_quat, pred_openness))

        return (pred_complete_traj, traj_metric)

    def conditional_diffusion_loss(self, conditions, trajectory):
        x_0 = trajectory

        # Sample random timesteps t for each trajectory
        timesteps = torch.randint(
            0,
            self.position_noise_scheduler.config.num_train_timesteps,
            (x_0.shape[0],),
            device=x_0.device,
        ).long()

        # epsilon@t ~ N(0, 1)
        # noise that would be removed from trajectory@t to get trajectory@t-1
        # noise that would be added to trajectory@t-1 to get trajectory@t
        epsilon_t = torch.randn(x_0.shape, device=x_0.device)

        x_t_pos = self.position_noise_scheduler.add_noise(
            x_0[..., :3], epsilon_t[..., :3], timesteps
        )
        x_t_rot = self.rotation_noise_scheduler.add_noise(
            x_0[..., 3:9], epsilon_t[..., 3:9], timesteps
        )

        (epsilon_t_pos_pred, epsilon_t_rot_pred, openness_pred) = self.noise_model(
            trajectory=torch.cat((x_t_pos, x_t_rot), -1), timestep=timesteps, **conditions
        )

        self.metric.feed(
            ground=torch.cat((epsilon_t[..., :3], epsilon_t[..., 3:9], x_0[..., 9:10]), -1),
            prediction=torch.cat((epsilon_t_pos_pred, epsilon_t_rot_pred, openness_pred), -1),
        )

        return self.metric

    def conditional_sample(self, conditions, shape, device, dtype):
        sampled_trajectory = torch.randn(size=shape, device=device, dtype=dtype)

        ones_like_timesteps = (
            torch.ones(len(sampled_trajectory)).to(sampled_trajectory.device).long()
        )

        self.position_noise_scheduler.set_timesteps(self.n_inference_steps)
        self.rotation_noise_scheduler.set_timesteps(self.n_inference_steps)

        # Iterative denoising
        timesteps = self.position_noise_scheduler.timesteps
        assert torch.allclose(self.rotation_noise_scheduler.timesteps, timesteps)
        for t in timesteps:
            (epsilon_t_pos_pred, epsilon_t_rot_pred, openness_pred) = self.noise_model(
                trajectory=sampled_trajectory, timestep=t * ones_like_timesteps, **conditions
            )
            # out = out[-1]  # keep only last layer's output # why remove batch???
            pos = self.position_noise_scheduler.step(
                epsilon_t_pos_pred, t, sampled_trajectory[..., :3]
            ).prev_sample
            rot = self.rotation_noise_scheduler.step(
                epsilon_t_rot_pred, t, sampled_trajectory[..., 3:9]
            ).prev_sample
            sampled_trajectory = torch.cat((pos, rot), -1)

        complete_traj = torch.cat((sampled_trajectory, openness_pred), -1)

        return complete_traj

    def process_batch(self, batch, is_evaluation_mode=False):
        (
            gt_trajectory,
            gt_openness,
            trajectory_mask,
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper,
        ) = self.process_input_transformations(
            gt_trajectory=None if is_evaluation_mode else batch["trajectory"][:, 1:],
            trajectory_mask=None if is_evaluation_mode else batch["trajectory_mask"][:, 1:],
            rgb_obs=batch["rgbs"],
            pcd_obs=batch["pcds"],
            instruction=batch["instr"],
            curr_gripper=(  # I hate this I hate life who tf mixes shapes like this
                # batch["curr_gripper"]
                # if self.num_history < 1
                # else
                batch["curr_gripper_history"][:, -self.num_history :]
            ),
        )

        encoded = self.encode_inputs(rgb_obs, pcd_obs, instruction, curr_gripper)

        return (
            gt_trajectory,
            gt_openness,
            trajectory_mask,
            {
                "rgb_obs": rgb_obs,
                "pcd_obs": pcd_obs,
                "instruction": instruction,
                "curr_gripper": curr_gripper,
            },
            {
                **encoded,
                "has_3d": torch.ones((rgb_obs.shape[0],), device=rgb_obs.device).bool(),
            },
        )

    # not ideal but deal with it later
    def process_input_transformations(
        self, gt_trajectory, trajectory_mask, rgb_obs, pcd_obs, instruction, curr_gripper
    ):
        if self.relative:
            pcd_obs, curr_gripper = convert2rel(pcd_obs, curr_gripper)

        if gt_trajectory is not None:
            gt_openness = gt_trajectory[..., 7:]
            gt_trajectory = gt_trajectory[..., :7]
        else:
            gt_openness = None

        curr_gripper = curr_gripper[..., :7]

        if gt_trajectory is not None:
            gt_trajectory = gt_trajectory.clone()
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()

        # Normalize all pos
        if gt_trajectory is not None:
            gt_trajectory[:, :, :3] = self.position_normalization.normalize_pos(
                gt_trajectory[:, :, :3]
            )
        pcd_obs = torch.permute(
            self.position_normalization.normalize_pos(
                torch.permute(pcd_obs, [0, 1, 3, 4, 2])
            ),
            [0, 1, 4, 2, 3],
        )
        curr_gripper[..., :3] = self.position_normalization.normalize_pos(
            curr_gripper[..., :3]
        )

        # Convert rotation parametrization
        if gt_trajectory is not None:
            gt_trajectory = self.rotation_parametrization.convert_rot(gt_trajectory)
        curr_gripper = self.rotation_parametrization.convert_rot(curr_gripper)

        return (
            gt_trajectory,
            gt_openness,
            trajectory_mask,
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper,
        )

    def encode_inputs(self, visible_rgb, visible_pcd, instruction, curr_gripper):
        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, pcd_pyramid = self.encoder.encode_images(visible_rgb, visible_pcd)
        # Keep only low-res scale
        context_feats = einops.rearrange(
            rgb_feats_pyramid[0], "b ncam c h w -> b (ncam h w) c"
        )
        context = pcd_pyramid[0]

        # Encode instruction (B, 53, F)
        instr_feats = None
        if self.use_instruction:
            instr_feats, _ = self.encoder.encode_instruction(instruction)

        # Cross-attention vision to language
        if self.use_instruction:
            # Attention from vision to language
            context_feats = self.encoder.vision_language_attention(context_feats, instr_feats)

        # Encode gripper history (B, nhist, F)
        adaln_gripper_feats, _ = self.encoder.encode_curr_gripper(
            curr_gripper, context_feats, context
        )

        # FPS on visual features (N, B, F) and (B, N, F, 2)
        fps_feats, fps_pos = self.encoder.run_fps(
            context_feats.transpose(0, 1), self.encoder.relative_pe_layer(context)
        )
        return {
            "context_feats": context_feats,
            "context": context,  # contextualized visual features
            "instr_feats": instr_feats,  # language features
            "adaln_gripper_feats": adaln_gripper_feats,  # gripper history features
            "fps_feats": fps_feats,
            "fps_pos": fps_pos,  # sampled visual features
        }

    def reverse_input_transformations(self, complete_traj):
        # Back to quaternion
        complete_traj = self.rotation_parametrization.unconvert_rot(complete_traj)
        # unnormalize position
        complete_traj[:, :, :3] = self.position_normalization.unnormalize_pos(
            complete_traj[:, :, :3]
        )
        # Convert gripper status to probaility
        if complete_traj.shape[-1] > 7:
            complete_traj[..., 7] = complete_traj[..., 7].sigmoid()

        return complete_traj
