import logging
from collections.abc import Callable

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
from manten.networks.vendor.diffusion_policy_3d.vision.pointnet_extractor import (
    PointNetEncoderXYZ,
    PointNetEncoderXYZRGB,
)

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
    def pred_horizon(self):
        return self.dataset_info["pred_horizon"]

    @property
    def act_dim(self):
        return self.actions_shape[-1]

    def __init__(
        self,
        state_encoder: Callable[..., torch.nn.Module],
        noise_scheduler: DDPMScheduler,
        act_horizon,  # prolly 8
        n_inference_steps=None,
        # conditional unet parameters
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        # pcd encoder parameters
        use_pc_color=False,
        pointnet_type="pointnet",
        pointcloud_encoder_cfg=None,
        # rest
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.state_encoder = state_encoder(
            state_shape=optree.tree_map(lambda x: x[1:], self.observations_shape["state_obs"])
        )

        self.use_pc_color = use_pc_color
        if pointnet_type == "pointnet":
            if use_pc_color:
                self.pcd_encoder = PointNetEncoderXYZRGB(
                    in_channels=6,
                    **pointcloud_encoder_cfg,
                )
            else:
                self.pcd_encoder = PointNetEncoderXYZ(
                    in_channels=3,
                    **pointcloud_encoder_cfg,
                )
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        self.condition_type = condition_type
        if "cross_attention" in self.condition_type:
            global_cond_dim = self.encode_obs_out_dim
        else:
            global_cond_dim = self.encode_obs_out_dim * self.obs_horizon

        model = ConditionalUnet1D(
            input_dim=self.act_dim,
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

        self.model = model
        self.noise_scheduler = noise_scheduler

        self.act_horizon = act_horizon

        if n_inference_steps is None:
            n_inference_steps = noise_scheduler.config.num_train_timesteps
        self.n_inference_steps = n_inference_steps

    def compute_train_gt_and_pred(self, pcd_obs, rgb_obs, pcd_mask, state_obs, actions):
        # rgb_obs (B, obs_horizon, cam, C, H, W)
        norm_action_seq = self.action_scaler.scale(actions)

        B = actions.shape[0]
        obs_cond = self.encode_obs(pcd_obs, rgb_obs, pcd_mask, state_obs)

        # sample noise to add to actions
        # only act_dim, no observation inpainting, tho maybe later joint acts?
        noise = torch.randn(
            (B, self.pred_horizon, self.act_dim), device=norm_action_seq.device
        )

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=norm_action_seq.device,
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(norm_action_seq, noise, timesteps)

        # predict the noise residual
        pred = self.model(noisy_action_seq, timesteps, global_cond=obs_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            return noise, pred
        elif pred_type == "sample":
            return norm_action_seq, pred
        elif pred_type == "v_prediction":
            # see: https://github.com/YanjieZe/3D-Diffusion-Policy/blob/c72a1ace81d1217f1e6450ca7a98fcd3b668c009/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py#L344
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = (
                self.noise_scheduler.alpha_t[timesteps],
                self.noise_scheduler.sigma_t[timesteps],
            )
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * norm_action_seq
            return v_t, pred
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

    def predict_actions(self, pcd_obs, rgb_obs, pcd_mask, state_obs):
        self.noise_scheduler.set_timesteps(self.n_inference_steps)

        _tmp = next(iter(rgb_obs.values()))
        B = _tmp.shape[0]
        device = _tmp.device

        with torch.no_grad():
            obs_cond = self.encode_obs(pcd_obs, rgb_obs, pcd_mask, state_obs)

            # initialize action from Gaussian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                pred = self.model(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        noisy_action_seq = self.action_scaler.descale(noisy_action_seq)

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)

    def adapt_actions_from_ds_actions(self, actions):
        return actions[..., : self.act_horizon, :]

    def encode_obs(self, pcd_obs, rgb_obs, pcd_mask, state_obs):
        B = next(iter(rgb_obs.values())).shape[0]

        state_features = self.state_encoder(
            optree.tree_map(lambda x: einops.rearrange(x, "b t ... -> (b t) ..."), state_obs)
        )

        pcd_obs = einops.rearrange(
            list(pcd_obs.values()), "cam b obs_h c h w -> b obs_h (cam h w) c"
        )
        pcd_mask = einops.rearrange(
            list(pcd_mask.values()), "cam b obs_h c h w -> b obs_h (cam h w) c"
        )

        pcd_obs = self.pcd_scaler.scale(pcd_obs)

        if self.use_pc_color:
            rgb_obs = einops.rearrange(
                list(rgb_obs.values()), "cam b obs_h c h w -> b obs_h (cam h w) c"
            )
            pointcloud = torch.cat([pcd_obs, rgb_obs], dim=-1)
        else:
            pointcloud = pcd_obs

        # apply mask:
        pointcloud = pointcloud * pcd_mask  # keep 1s, zero out 0s

        pointcloud = einops.rearrange(pointcloud, "b obs_h n c -> (b obs_h) n c")
        pcd_features = self.pcd_encoder(pointcloud)
        combined_features = torch.cat([pcd_features, state_features], dim=-1)
        if "cross_attention" in self.condition_type:
            # treat as a sequence
            global_cond = einops.rearrange(
                combined_features, "(b t) ... -> b t ...", b=B, t=self.obs_horizon
            )
        else:
            # reshape back to B, Do
            global_cond = einops.rearrange(
                combined_features, "(b t) ... -> b (t ...)", b=B, t=self.obs_horizon
            )

        return global_cond

    @property
    def encode_obs_out_dim(self):
        if hasattr(self, "_encode_obs_out_dim_cached"):
            return self._encode_obs_out_dim_cached

        self.to(self.device)
        with torch.no_grad():
            sample_observation = optree.tree_map(
                lambda x: torch.zeros((1, *x), device=self.device), self.observations_shape
            )

            sample_obs_cond = self.encode_obs(
                sample_observation["pcd_obs"],
                sample_observation["rgb_obs"],
                optree.tree_map(lambda x: ~x.bool(), sample_observation["pcd_mask"]),
                sample_observation["state_obs"],
            )

        self._encode_obs_out_dim_cached = sample_obs_cond.shape[-1]
        return self.encode_obs_out_dim
