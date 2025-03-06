import logging

import einops
import optree
import torch
from diffusers.schedulers import DDPMScheduler

from manten.agents.utils.mixins import DatasetActionScalerMixin, DatasetPCDScalerMixin
from manten.agents.utils.templates import BatchPCDObservationActionAgentTemplate
from manten.metrics.traj_action_metric import PosRotGripperMetric, PosRotGripperStats
from manten.networks.vendor.diffusion_policy_3d.diffusion.conditional_unet1d import (
    ConditionalUnet1D,
)
from manten.networks.vendor.diffusion_policy_3d.vision.pointnet_extractor import DP3Encoder

logger = logging.getLogger(__name__)


@BatchPCDObservationActionAgentTemplate.make_agent(
    evaluation_metric_cls=PosRotGripperMetric, evaluation_stats_cls=PosRotGripperStats
)
class DP3Agent(
    BatchPCDObservationActionAgentTemplate, DatasetActionScalerMixin, DatasetPCDScalerMixin
):
    @property
    def obs_horizon(self):
        return self.dataset_info["obs_horizon"]

    @property
    def horizon(self):  # pred_horizon
        return self.dataset_info["pred_horizon"]

    @property
    def act_dim(self):
        return self.actions_shape[-1]

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        n_action_steps,  # prolly 8
        num_inference_steps=None,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        encoder_output_dim=256,
        crop_shape=None,
        use_pc_color=False,
        pointnet_type="pointnet",
        pointcloud_encoder_cfg=None,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:  # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        # obs_shape_meta = shape_meta["obs"]
        # obs_dict = dict_apply(obs_shape_meta, lambda x: x["shape"])

        obs_encoder = DP3Encoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim
        if "cross_attention" in self.condition_type:
            global_cond_dim = obs_feature_dim
        else:
            global_cond_dim = obs_feature_dim * self.obs_horizon

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        logger.info(
            f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}"
        )
        logger.info(
            f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}"
        )

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler

        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)

        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.obs_as_global_cond = True
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print_params(self)

    def compute_train_gt_and_pred(self, pcd_obs, rgb_obs, pcd_mask, state_obs, actions):
        pass

    def predict_actions(self, pcd_obs, rgb_obs, pcd_mask, state_obs):
        pass

    def adapt_actions_from_ds_actions(self, actions):
        return actions[..., : self.n_action_steps, :]

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
        context = einops.rearrange(pcd_pyramid[0], "bt ncam c h w -> bt (ncam h w) c")
        context_keep_mask = mask_pyramid[0]

        context_pe = self.encoder.relative_pe_layer(context)

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
            context_pe,
            keep_mask=context_keep_mask.transpose(0, 1),
        )
        return {
            "context_feats": context_feats,
            "context": context_pe,  # contextualized visual features
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
    from manten.metrics.mse_loss_pose_bce_loss_gripper_metric import (
        MSELossPoseBCEWithLogitsLossGripperMetric,
    )
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

    agent = DP3Agent(
        dataset_info=dataset_info,
        position_noise_scheduler=ddpm(),
        rotation_noise_scheduler=ddpm(),
        encoder=encoder,
        metric=MSELossPoseBCEWithLogitsLossGripperMetric(),
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
