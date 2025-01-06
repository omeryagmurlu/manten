import einops
import optree
import torch

from manten.agents.utils.mixins import DatasetActionScalerMixin, DatasetPCDScalerMixin
from manten.agents.utils.templates import BatchPCDObservationActionAgentTemplate
from manten.metrics.traj_action_metric import (
    MSELossPoseBCEGripperMetric,
    PosRotGripperMetric,
    PosRotGripperStats,
)


def convert2rel(pcd, curr_gripper, *others):
    """Convert coordinate system relative to current gripper."""
    center = curr_gripper[:, -1, :3]  # (batch_size, 3)
    bs = center.shape[0]
    mids = len(pcd.shape[1:-3]) * (1,)
    pcd = pcd - center.view(bs, *mids, 3, 1, 1)
    curr_gripper = curr_gripper.clone()
    curr_gripper[..., :3] = curr_gripper[..., :3] - center.view(bs, 1, 3)
    others = [other[..., :3] - center.view(bs, 1, 3) for other in others]
    return (pcd, curr_gripper, *others)


@BatchPCDObservationActionAgentTemplate.make_agent(
    evaluation_metric_cls=PosRotGripperMetric, evaluation_stats_cls=PosRotGripperStats
)
class ThreeDDAAgent(
    BatchPCDObservationActionAgentTemplate, DatasetActionScalerMixin, DatasetPCDScalerMixin
):
    def __init__(
        self,
        *,
        position_noise_scheduler,
        rotation_noise_scheduler,
        encoder,
        noise_model,
        embedding_dim=192,
        relative=True,  # center the pcd around tcp
        n_inference_steps=10,
        tcp_pose_key=None,
        act_horizon=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.position_noise_scheduler = position_noise_scheduler
        self.rotation_noise_scheduler = rotation_noise_scheduler

        self.relative = relative
        self.use_instruction = False  # will come from dataset
        self.n_inference_steps = n_inference_steps
        self.act_horizon = act_horizon

        if tcp_pose_key is None:
            self.tcp_pose_key = self.dataset_info["tcp_pose_key"]
        else:
            self.tcp_pose_key = tcp_pose_key

        remaining_state_keys = set(self.observations_shape["state_obs"].keys())
        remaining_state_keys.remove(self.tcp_pose_key)
        self.encoder_custom_state_shapes = {
            key: self.observations_shape["state_obs"][key][1:] for key in remaining_state_keys
        }

        self.encoder = encoder(
            nhist=self.obs_horizon,
            embedding_dim=embedding_dim,
            custom_state_shapes=self.encoder_custom_state_shapes,
        )
        self.noise_model = noise_model(
            rotation_dim=self.rotation_dim,
            nhist=self.obs_horizon,
            embedding_dim=embedding_dim,
        )

    @property
    def obs_horizon(self):
        return self.dataset_info["obs_horizon"]

    @property
    def pred_horizon(self):
        return self.dataset_info["pred_horizon"]

    @property
    def act_dim(self):
        return self.actions_shape[-1]

    @property
    def rotation_dim(self):
        return self.dataset_info["rotation_dim"]

    def compute_train_gt_and_pred(self, pcd_obs, rgb_obs, pcd_mask, state_obs, actions):
        # rgb_obs (B, obs_horizon, cam, C, H, W)
        norm_actions = self.action_scaler.scale(actions)

        B = actions.shape[0]
        conditions = self.encode_observations(pcd_obs, rgb_obs, pcd_mask, state_obs)

        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=actions.device)

        timesteps = torch.randint(
            0,
            self.position_noise_scheduler.config.num_train_timesteps,
            (B,),
            device=actions.device,
        ).long()

        noisy_position = self.position_noise_scheduler.add_noise(
            norm_actions[..., :3], noise[..., :3], timesteps
        )
        noisy_rotation = self.rotation_noise_scheduler.add_noise(
            norm_actions[..., 3:-1], noise[..., 3:-1], timesteps
        )

        noisy_traj = torch.cat((noisy_position, noisy_rotation), -1)  # don't add openness

        (pos_pred, rot_pred, openness_pred) = self.noise_model(
            trajectory=noisy_traj, timestep=timesteps, **conditions
        )

        pred = torch.cat((pos_pred, rot_pred, openness_pred), -1)

        pred_type = self.position_noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            return noise, pred
        elif pred_type == "sample":
            return norm_actions, pred
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

    def predict_actions(self, pcd_obs, rgb_obs, pcd_mask, state_obs):
        self.position_noise_scheduler.set_timesteps(self.n_inference_steps)
        self.rotation_noise_scheduler.set_timesteps(self.n_inference_steps)

        with torch.no_grad():
            conditions = self.encode_observations(pcd_obs, rgb_obs, pcd_mask, state_obs)
            sample_tensor = next(iter(pcd_obs.values()))
            B = sample_tensor.shape[0]
            device = sample_tensor.device

            traj_wo_openness = torch.randn(
                (B, self.pred_horizon, self.act_dim - 1), device=device
            )

            for t in self.position_noise_scheduler.timesteps:
                (epsilon_t_pos_pred, epsilon_t_rot_pred, openness_pred) = self.noise_model(
                    trajectory=traj_wo_openness,
                    timestep=t * torch.ones(B, device=device, dtype=torch.long),
                    **conditions,
                )

                pos = self.position_noise_scheduler.step(
                    epsilon_t_pos_pred, t, traj_wo_openness[..., :3]
                ).prev_sample
                rot = self.rotation_noise_scheduler.step(
                    epsilon_t_rot_pred, t, traj_wo_openness[..., 3:]
                ).prev_sample
                traj_wo_openness = torch.cat((pos, rot), -1)

            complete_traj = torch.cat((traj_wo_openness, openness_pred), -1)
            complete_traj = self.action_scaler.descale(complete_traj)

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return complete_traj[:, start:end]

    def adapt_actions_from_ds_actions(self, actions):
        return actions[..., : self.act_horizon, :]

    def encode_observations(self, pcd_obs, rgb_obs, pcd_mask, state_obs):
        # n_cam = len(pcd_obs)
        # B, obs_h, C, H, W = next(iter(pcd_obs.values())).shape
        pcd_obs = einops.rearrange(
            list(pcd_obs.values()), "cam b obs_h c h w -> b obs_h cam c h w"
        )
        pcd_mask = einops.rearrange(
            list(pcd_mask.values()), "cam b obs_h c h w -> b obs_h cam c h w"
        )
        rgb_obs = einops.rearrange(
            list(rgb_obs.values()), "cam b obs_h c h w -> b obs_h cam c h w"
        )
        curr_gripper = state_obs[self.tcp_pose_key]
        custom_states = {key: state_obs[key] for key in self.encoder_custom_state_shapes}

        # position (3) normalization
        curr_gripper = self.pcd_scaler.scale(curr_gripper[..., :3])
        pcd_obs = self.pcd_scaler.scale(pcd_obs)
        if "goal_pos" in custom_states:
            custom_states["goal_pos"] = self.pcd_scaler.scale(custom_states["goal_pos"])

        if self.relative:
            if "goal_pos" not in custom_states:
                pcd_obs, curr_gripper = convert2rel(pcd_obs, curr_gripper)
            else:
                pcd_obs, curr_gripper, custom_states["goal_pos"] = convert2rel(
                    pcd_obs, curr_gripper, custom_states["goal_pos"]
                )

        # for now only use last observation for visuals like in the paper
        pcd_obs = pcd_obs[:, -1]
        pcd_mask = pcd_mask[:, -1]
        rgb_obs = rgb_obs[:, -1]

        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, pcd_pyramid, mask_pyramid = self.encoder.encode_images(
            rgb_obs, pcd_obs, pcd_mask
        )
        # Keep only low-res scale
        context_feats = rgb_feats_pyramid[0]
        context = pcd_pyramid[0]
        context_keep_mask = mask_pyramid[0]

        # # Encode instruction (B, 53, F)
        # instr_feats = None
        # # instr will be part of state_obs, first 10 dim tcp_pose
        # # if self.use_instruction:
        # #     instr_feats, _ = self.encoder.encode_instruction(instruction)
        # #     # Attention from vision to language
        # #     context_feats = self.encoder.vision_language_attention(context_feats, instr_feats)
        instr_feats = self.encoder.encode_custom_states(custom_states)

        # Encode gripper history (B, nhist, F)
        adaln_gripper_feats, _ = self.encoder.encode_curr_gripper(
            curr_gripper, context_feats, context
        )

        # FPS on visual features (N, B, F) and (B, N, F, 2)
        fps_feats, fps_pos = self.encoder.run_fps(
            context_feats.transpose(0, 1),
            self.encoder.relative_pe_layer(context),
            keep_mask=context_keep_mask.transpose(0, 1),
        )
        return {
            "context_feats": context_feats,
            "context": context,  # contextualized visual features
            "instr_feats": instr_feats,  # language features
            "adaln_gripper_feats": adaln_gripper_feats,  # gripper history features
            "fps_feats": fps_feats,
            "fps_pos": fps_pos,  # sampled visual features
        }


if __name__ == "__main__":
    from functools import partial

    from diffusers.schedulers import DDPMScheduler

    from manten.agents.utils.normalization import MinMaxScaler
    from manten.data.dataset_maniskill import ManiSkillDataset
    from manten.networks.vendor.three_dda.encoder import Encoder
    from manten.networks.vendor.three_dda.head import DiffusionHead

    dataset = ManiSkillDataset(
        simulated_length=10000000,
        test_ratio=0.05,
        task="PickCube-v1",
        pack_root="/home/i53/student/yagmurlu/code/manten/data/maniskill2/packed_demos",
        obs_horizon=2,
        pred_horizon=16,
        obs_mode="pointcloud",
        # state_modality_keys=["tcp_pose"],
        state_modality_keys=["tcp_pose", "goal_pos"],
        # rgb_modality_keys=["camera1", "gripper1"],
        rgb_modality_keys=["camera1"],
        control_mode="pd_ee_delta_pose",
        use_mmap=True,
        load_count=3,
        rotation_transform="rotation_6d",
        # use_mmap=False,
    )

    dataset_info = dataset.get_dataset_info()

    ddpm = partial(DDPMScheduler, num_train_timesteps=125, beta_schedule="squaredcos_cap_v2")

    #   encoder:
    #     _target_: manten.agents.three_dda.encoder.Encoder
    #     backbone: "clip"
    #     image_size: [256, 256]
    #     embedding_dim: ${..._embedding_dim}
    #     num_sampling_level: 1
    #     nhist: ${..._num_history}
    #     num_vis_ins_attn_layers: 2
    #     fps_subsampling_factor: 3
    #     encoder = Encoder()

    encoder = partial(
        Encoder,
        backbone="clip",
        num_sampling_level=1,
        num_vis_ins_attn_layers=2,
        fps_subsampling_factor=3,
    )

    #   noise_model:
    # _target_: manten.agents.three_dda.head.DiffusionHead
    # embedding_dim: ${..._embedding_dim}
    # use_instruction: True
    # nhist: ${..._num_history}
    # lang_enhanced: True
    # rotation_parametrization: ${..._rotation_parametrization}

    noise_model = partial(
        DiffusionHead,
        # use_instruction=False,
        # lang_enhanced=False,
        use_instruction=True,
        lang_enhanced=True,
    )

    agent = ThreeDDAAgent(
        dataset_info=dataset_info,
        position_noise_scheduler=ddpm(),
        rotation_noise_scheduler=ddpm(),
        encoder=encoder,
        metric=MSELossPoseBCEGripperMetric(),
        noise_model=noise_model,
        action_scaler=MinMaxScaler,
        relative=True,
        n_inference_steps=10,
    )
    agent.to("cuda")  # flash-attn not implemented for cpu

    dl = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    it = iter(dl)
    batch = next(it)
    batch = optree.tree_map(lambda x: x.to("cuda"), batch)

    agent("train", batch)
    agent("eval", batch)

    actions = agent.predict_actions(observations=batch["observations"])
    print(actions.shape)
    print(actions)
