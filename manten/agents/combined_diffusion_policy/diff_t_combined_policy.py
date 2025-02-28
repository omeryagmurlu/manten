from collections.abc import Callable
from functools import partial

import einops
import optree
import torch

from manten.agents.utils.mixins import DatasetActionScalerMixin, DatasetPCDScalerMixin
from manten.agents.utils.templates import BatchPCDOrRGBObservationActionAgentTemplate
from manten.metrics.traj_action_metric import PosRotGripperMetric, PosRotGripperStats
from manten.networks.vendor.diffusion_policy.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)
from manten.networks.vendor.diffusion_policy.vision.multi_image_obs_encoder import (
    MultiImageObsEncoder,
)
from manten.utils.utils_pytorch import get_ones_shape_like
from manten.utils.utils_pytree import tree_rearrange


def noop_encoder(x, **_kwargs):
    return x


# def cat_all_encoder(x, **_kwargs):
#     return torch.cat(list(x.values()), dim=-1)

def cat_all_encoder(x, excluded_keys=(), **_kwargs):
    return torch.cat(
        [x[key] for key in x if key not in excluded_keys], dim=-1
    )


@BatchPCDOrRGBObservationActionAgentTemplate.make_agent(
    evaluation_metric_cls=PosRotGripperMetric, evaluation_stats_cls=PosRotGripperStats
)
class DiffusionTransformerCombinedPolicy(
    BatchPCDOrRGBObservationActionAgentTemplate,
    DatasetActionScalerMixin,
    DatasetPCDScalerMixin,
):
    def __init__(
        self,
        *,
        rgb_encoder: Callable[..., MultiImageObsEncoder],
        pcd_encoder,
        state_encoder: Callable[..., torch.nn.Module] = cat_all_encoder,
        pred_net: Callable[..., TransformerForDiffusion],
        act_horizon,
        noise_scheduler,
        num_diffusion_iters,
        train_modes=("2d", "3d"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.actions_shape[-2] == self.pred_horizon

        self.act_horizon = act_horizon
        self.train_modes = train_modes
        self.num_diffusion_iters = num_diffusion_iters

        self.noise_scheduler = noise_scheduler
        self.rgb_encoder = rgb_encoder(
            rgb_shape=optree.tree_map(lambda x: x[1:], self.observations_shape["rgb_obs"])
        )
        self.state_encoder = partial(
            state_encoder,
            state_shape=optree.tree_map(
                lambda x: x[1:], self.observations_shape["state_obs"]
            ),
        )

        # if "3d" not in self.train_modes:
        #     self.pcd_encoder = None
        # else:
        #     self.pcd_encoder = pcd_encoder
        self.pcd_encoder = pcd_encoder

        self.pred_net = pred_net(
            input_dim=self.act_dim,
            output_dim=self.act_dim,
            horizon=self.pred_horizon,
            n_obs_steps=self.obs_horizon,
            cond_dim=self.encode_obs_out_dim,
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
    def compute_train_gt_and_pred(
        self, pcd_obs, rgb_obs, pcd_mask, state_obs, actions, keep_mask_3d
    ):
        # rgb_obs (B, obs_horizon, cam, C, H, W)
        norm_action_seq = self.action_scaler.scale(actions)

        B = actions.shape[0]
        obs_cond, obs_cond_2d = self.encode_obs(pcd_obs, rgb_obs, pcd_mask, state_obs)

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
        prediction_by_vis_mode = {}
        for mode in self.train_modes:
            cond = obs_cond if mode == "3d" else obs_cond_2d
            prediction_by_vis_mode[mode] = self.pred_net(
                noisy_action_seq.clone(), timesteps.clone(), cond=cond
            )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            ret_gt = noise
        elif pred_type == "sample":
            ret_gt = norm_action_seq
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        indices_3d = keep_mask_3d.nonzero()
        # an optimization would be moving this to the top, but eh
        gt_modes = {}
        pred_modes = {}
        if "2d" in self.train_modes:
            gt_modes["2d"] = ret_gt
            pred_modes["2d"] = prediction_by_vis_mode["2d"]
        if "3d" in self.train_modes:
            gt_modes["3d"] = ret_gt[indices_3d]
            pred_modes["3d"] = prediction_by_vis_mode["3d"][indices_3d]
        if "2d" in self.train_modes and "3d" in self.train_modes:
            pred_modes["2d_for_3d"] = prediction_by_vis_mode["2d"][indices_3d]

        return gt_modes, pred_modes

    def predict_actions(self, pcd_obs, rgb_obs, pcd_mask, state_obs, keep_mask_3d):
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        with torch.no_grad():
            obs_cond, obs_cond_2d = self.encode_obs(pcd_obs, rgb_obs, pcd_mask, state_obs)
            cond = torch.where(
                ~keep_mask_3d.view(keep_mask_3d.shape[0], *get_ones_shape_like(obs_cond)[1:]),
                obs_cond_2d,
                obs_cond,
            )

            B = cond.shape[0]

            # initialize action from Gaussian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=cond.device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                pred = self.pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    cond=cond,
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
        # rgb_obs is a dict of key(cam_name): (B, obs_horizon, C, H, W)
        B = next(iter(rgb_obs.values())).shape[0]

        rgb_features = self.rgb_encoder(tree_rearrange(rgb_obs, "b t c h w -> (b t) c h w"))
        rgb_cond = einops.rearrange(
            rgb_features, "(b t) ... -> b t ...", b=B, t=self.obs_horizon
        )

        state_features = self.state_encoder(
            optree.tree_map(lambda x: einops.rearrange(x, "b t ... -> (b t) ..."), state_obs)
        )
        state_cond = einops.rearrange(
            state_features, "(b t) ... -> b t ...", b=B, t=self.obs_horizon
        )

        # if self.pcd_encoder is None:
        #     obs_cond_2d = torch.cat([rgb_cond, state_cond], dim=-1)
        #     obs_cond = obs_cond_2d
        #     return obs_cond, obs_cond_2d

        pcd_features = self.pcd_encoder(
            tree_rearrange(pcd_obs, "b t c h w -> (b t) c h w"),
            tree_rearrange(pcd_mask, "b t c h w -> (b t) c h w"),
        )
        pcd_cond = einops.rearrange(
            pcd_features, "(b t) ... -> b t ...", b=B, t=self.obs_horizon
        )

        obs_cond = torch.cat([rgb_cond, pcd_cond, state_cond], dim=-1)
        # obs_cond_2d = torch.cat([rgb_cond, torch.zeros_like(pcd_cond), state_cond], dim=-1)
        obs_cond_2d = obs_cond

        return obs_cond, obs_cond

    @property
    def encode_obs_out_dim(self):
        self.to(self.device)
        with torch.no_grad():
            sample_observation = optree.tree_map(
                lambda x: torch.zeros((1, *x), device=self.device), self.observations_shape
            )

            sample_obs_cond, _ = self.encode_obs(
                sample_observation["pcd_obs"],
                sample_observation["rgb_obs"],
                sample_observation["pcd_mask"],
                sample_observation["state_obs"],
            )

        return sample_obs_cond.shape[-1]


if __name__ == "__main__":
    from functools import partial

    import hydra
    from diffusers.schedulers import DDPMScheduler
    from omegaconf import OmegaConf

    from manten.agents.utils.normalization import MinMaxScaler
    from manten.data.dataset_maniskill import ManiSkillDataset
    from manten.metrics.mse_loss_pose_bce_loss_gripper_metric import (
        MSELossPoseBCEWithLogitsLossGripperMetric,
    )
    from manten.networks.vendor.diffusion_policy.vision.model_getter import get_resnet

    dataset = ManiSkillDataset(
        simulated_length=10000000,
        test_ratio=0.05,
        task="PickCube-v1",
        pack_root="./data/maniskill2/packed_demos",
        obs_horizon=2,
        pred_horizon=16,
        obs_mode="pointcloud",
        state_modality_keys=["goal_pos"],
        rgb_modality_keys=["camera1"],
        control_mode="pd_ee_delta_pose",
        use_mmap=True,
        load_count=35,
        rotation_transform="rotation_6d",
        # use_mmap=False,
    )

    noise_scheduler = DDPMScheduler()
    metric = MSELossPoseBCEWithLogitsLossGripperMetric(1.0, 1.0, 1.0)
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
        TransformerForDiffusion,
        # diffusion_step_embed_dim=256,
        # unet_dims=(256, 512, 1024),
        # n_groups=8,
        # kernel_size=5,
        # cond_predict_scale=True,
    )

    pcd_encoder_config = OmegaConf.load("configs/agent/pcd_encoder/pointnext-b.yaml")
    pcd_encoder = hydra.utils.instantiate(pcd_encoder_config)

    agent = DiffusionTransformerCombinedPolicy(
        metric=metric,
        dataset_info=dataset_info,
        action_scaler=action_scaler,
        rgb_encoder=rgb_encoder,
        pcd_encoder=pcd_encoder,
        # state_encoder=lambda x, **_: x["goal_pos"],
        pred_net=pred_net,
        act_horizon=8,
        noise_scheduler=noise_scheduler,
        num_diffusion_iters=100,
        device="cuda",
    )
    agent.to("cuda")

    dl = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    it = iter(dl)
    batch = next(it)
    batch = optree.tree_map(
        lambda x: x if not isinstance(x, torch.Tensor) else x.to("cuda"), batch
    )
    agent("train", batch)
    agent("eval", batch)

    T = torch.Tensor([True]).bool().to("cuda")
    F = torch.Tensor([False]).bool().to("cuda")

    actions = agent.predict_actions(
        observations=batch["observations"], meta={"3d_mask": T, "2d_mask": T}
    )
    print(actions.shape)
    print(actions)
