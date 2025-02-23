from collections.abc import Callable
from functools import partial

import einops
import optree
import torch
from torch import nn

from manten.agents.utils.mixins import DatasetActionScalerMixin
from manten.agents.utils.templates import BatchRGBObservationActionAgentTemplate
from manten.metrics.traj_action_metric import PosRotGripperMetric, PosRotGripperStats
from manten.networks.vendor.diffusion_policy.diffusion.conditional_unet1d import (
    ConditionalUnet1D,
)
from manten.networks.vendor.diffusion_policy.vision.multi_image_obs_encoder import (
    MultiImageObsEncoder,
)
from manten.utils.utils_pytree import tree_rearrange


def noop_encoder(*_args, **_kwargs):
    return nn.Identity()


@BatchRGBObservationActionAgentTemplate.make_agent(
    evaluation_metric_cls=PosRotGripperMetric, evaluation_stats_cls=PosRotGripperStats
)
class DiffusionUnetImagePolicy(
    BatchRGBObservationActionAgentTemplate, DatasetActionScalerMixin
):
    def __init__(
        self,
        *,
        rgb_encoder: Callable[..., MultiImageObsEncoder],
        state_encoder: Callable[..., torch.nn.Module] = noop_encoder,
        pred_net: Callable[..., ConditionalUnet1D],
        act_horizon,
        noise_scheduler,
        num_diffusion_iters,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.actions_shape[-2] == self.pred_horizon

        self.act_horizon = act_horizon
        self.num_diffusion_iters = num_diffusion_iters

        self.noise_scheduler = noise_scheduler
        self.rgb_encoder = rgb_encoder(
            rgb_shape=optree.tree_map(lambda x: x[1:], self.observations_shape["rgb_obs"])
        )
        self.state_encoder = state_encoder(
            state_shape=optree.tree_map(lambda x: x[1:], self.observations_shape["state_obs"])
        )

        self.pred_net = pred_net(
            input_dim=self.act_dim, global_cond_dim=self.encode_obs_out_dim
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

    # missing: rotation transformation
    #          rgb normalization/transforms
    def compute_train_gt_and_pred(self, rgb_obs, state_obs, actions):
        # rgb_obs (B, obs_horizon, cam, C, H, W)
        norm_action_seq = self.action_scaler.scale(actions)

        B = actions.shape[0]
        obs_cond = self.encode_obs(rgb_obs, state_obs)

        # sample noise to add to actions
        # only act_dim, no observation inpainting, tho maybe later joint acts?
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_cond.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=obs_cond.device,
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(norm_action_seq, noise, timesteps)

        # predict the noise residual
        pred = self.pred_net(noisy_action_seq, timesteps, global_cond=obs_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            return noise, pred
        elif pred_type == "sample":
            return norm_action_seq, pred
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

    def predict_actions(self, rgb_obs, state_obs):
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        with torch.no_grad():
            obs_cond = self.encode_obs(rgb_obs, state_obs)
            B = obs_cond.shape[0]

            # initialize action from Gaussian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_cond.device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                pred = self.pred_net(
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

    def encode_obs(self, rgb_obs, state_obs):
        # rgb_obs is a dict of key(cam_name): (B, obs_horizon, C, H, W)
        B = next(iter(rgb_obs.values())).shape[0]

        rgb_features = self.rgb_encoder(tree_rearrange(rgb_obs, "b t c h w -> (b t) c h w"))
        rgb_cond = einops.rearrange(
            rgb_features, "(b t) ... -> b (t ...)", b=B, t=self.obs_horizon
        )

        state_features = self.state_encoder(
            einops.rearrange(state_obs, "b t ... -> (b t) ...")
        )
        state_cond = einops.rearrange(
            state_features, "(b t) ... -> b (t ...)", b=B, t=self.obs_horizon
        )

        obs_cond = torch.cat([rgb_cond, state_cond], dim=-1)

        return obs_cond

    @property
    def encode_obs_out_dim(self):
        with torch.no_grad():
            sample_observation = optree.tree_map(
                lambda x: torch.zeros((1, *x), device=self.device), self.observations_shape
            )

            sample_obs_cond = self.encode_obs(
                sample_observation["rgb_obs"], sample_observation["state_obs"]
            )

        return sample_obs_cond.shape[-1]


if __name__ == "__main__":
    from functools import partial

    from diffusers.schedulers import DDPMScheduler

    from manten.agents.utils.normalization import MinMaxScaler
    from manten.data.dataset_maniskill import ManiSkillDataset
    from manten.metrics.traj_action_metric import MSELossPoseBCEGripperMetric
    from manten.networks.vendor.diffusion_policy.vision.model_getter import get_resnet

    dataset = ManiSkillDataset(
        simulated_length=10000000,
        test_ratio=0.05,
        task="PickCube-v1",
        pack_root="/home/i53/student/yagmurlu/code/manten/data/maniskill2/packed_demos",
        obs_horizon=2,
        pred_horizon=16,
        obs_mode="rgb",
        state_modality_keys=["goal_pos"],
        rgb_modality_keys=["camera1"],
        control_mode="pd_ee_delta_pose",
        use_mmap=True,
        load_count=35,
        rotation_transform="rotation_6d",
        # use_mmap=False,
    )

    noise_scheduler = DDPMScheduler()
    metric = MSELossPoseBCEGripperMetric()
    action_scaler = MinMaxScaler
    rgb_encoder = partial(
        MultiImageObsEncoder,
        rgb_model=get_resnet(name="resnet18", weights=None),
        resize_shape=None,
        crop_shape=[76, 76],
        random_crop=True,
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=True,
    )
    dataset_info = dataset.get_dataset_info()

    print(dataset_info)

    pred_net = partial(
        ConditionalUnet1D,
        # diffusion_step_embed_dim=256,
        # unet_dims=(256, 512, 1024),
        # n_groups=8,
        # kernel_size=5,
        # cond_predict_scale=True,
    )

    agent = DiffusionUnetImagePolicy(
        metric=metric,
        dataset_info=dataset_info,
        action_scaler=action_scaler,
        rgb_encoder=rgb_encoder,
        pred_net=pred_net,
        act_horizon=8,
        noise_scheduler=noise_scheduler,
        num_diffusion_iters=100,
    )

    dl = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    it = iter(dl)
    batch = next(it)
    agent("train", batch)
    agent("eval", batch)

    actions = agent.predict_actions(observations=batch["observations"])
    print(actions.shape)
    print(actions)
