from collections.abc import Callable
from functools import partial

import einops
import optree
import torch

from manten.agents.utils.mixins import DatasetActionScalerMixin, DatasetPCDScalerMixin
from manten.agents.utils.templates import BatchPCDOrRGBObservationActionAgentTemplate
from manten.metrics.traj_action_metric import PosRotGripperMetric, PosRotGripperStats
from manten.networks.vendor.diffusion_policy.vision.multi_image_obs_encoder import (
    MultiImageObsEncoder,
)
from manten.utils.utils_pytorch import get_ones_shape_like
from manten.utils.utils_pytree import tree_rearrange


@BatchPCDOrRGBObservationActionAgentTemplate.make_agent(
    evaluation_metric_cls=PosRotGripperMetric, evaluation_stats_cls=PosRotGripperStats
)
class MantenCombinedPolicy(
    BatchPCDOrRGBObservationActionAgentTemplate,
    DatasetActionScalerMixin,
    DatasetPCDScalerMixin,
):
    def __init__(
        self,
        *,
        encoder: Callable,
        pred_net: Callable,
        act_horizon,
        noise_scheduler,
        num_diffusion_iters,
        train_modes=("2d", "3d"),
        include_rgb_cond_in_3d=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.actions_shape[-2] == self.pred_horizon

        self.act_horizon = act_horizon
        self.num_diffusion_iters = num_diffusion_iters

        self.train_modes = train_modes

        self.noise_scheduler = noise_scheduler

        self.encoder = encoder(
            state_shape=self.observations_shape.get("state_obs", None),
            rgb_shape=self.observations_shape.get("rgb_obs", None),
            pcd_shape=self.observations_shape.get("pcd_obs", None),
            pcd_scaler=self.pcd_scaler,
            train_modes=train_modes,
        )

        self.include_rgb_cond_in_3d = include_rgb_cond_in_3d

        encode_obs_out_shape = self.encode_obs_out_shape

        self.pred_net = pred_net(
            act_dim=self.act_dim,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            cond_types=list(encode_obs_out_shape.keys()),
            cond_type_num_tokens=optree.tree_map(lambda x: x[-2], encode_obs_out_shape),
            cond_type_input_dims=optree.tree_map(lambda x: x[-1], encode_obs_out_shape),
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
        obs_cond_3d, obs_cond_2d = self.encode_obs(pcd_obs, rgb_obs, pcd_mask, state_obs)

        # sample noise to add to actions
        # only act_dim, no observation inpainting, tho maybe later joint acts?
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=self.device,
        ).long()

        # forward diffusion
        noisy_action_seq = self.noise_scheduler.add_noise(norm_action_seq, noise, timesteps)

        # predict the noise residual
        prediction_by_vis_mode = {}
        for mode in self.train_modes:
            cond = obs_cond_3d if mode == "3d" else obs_cond_2d
            if mode == "2d":
                prediction_by_vis_mode[mode] = self.pred_net(
                    sample=noisy_action_seq.clone(),
                    timestep=timesteps.clone(),
                    conds=optree.tree_map(lambda x: x.clone(), cond),
                )
            elif mode == "3d":
                n_3d_samples = noisy_action_seq.clone()[keep_mask_3d]
                t_3d_samples = timesteps.clone()[keep_mask_3d]
                c_3d_samples = optree.tree_map(lambda x: x[keep_mask_3d].clone(), cond)
                prediction_by_vis_mode[mode] = self.pred_net(
                    sample=n_3d_samples, timestep=t_3d_samples, conds=c_3d_samples
                )
            else:
                raise ValueError(f"Unsupported train mode {mode}")

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            ret_gt = noise
        elif pred_type == "sample":
            ret_gt = norm_action_seq
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        gt_modes = {}
        pred_modes = {}
        if "2d" in self.train_modes:
            gt_modes["2d"] = ret_gt
            pred_modes["2d"] = prediction_by_vis_mode["2d"]
        if "3d" in self.train_modes:
            gt_modes["3d"] = ret_gt[keep_mask_3d]
            pred_modes["3d"] = prediction_by_vis_mode["3d"]
        if "2d" in self.train_modes and "3d" in self.train_modes:
            # assumption: each 3d data has 2d, but not vice versa
            pred_modes["2d_for_3d"] = prediction_by_vis_mode["2d"][keep_mask_3d]

        return gt_modes, pred_modes

    def predict_actions(self, pcd_obs, rgb_obs, pcd_mask, state_obs, keep_mask_3d):
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        acts = []
        masks = [keep_mask_3d, ~keep_mask_3d]

        for idx, mask in enumerate(masks):
            with torch.no_grad():
                obs_cond_3d, obs_cond_2d = self.encode_obs(
                    pcd_obs, rgb_obs, pcd_mask, state_obs
                )
                obs = obs_cond_3d if idx == 0 else obs_cond_2d
                conds = {k: v[mask] for k, v in obs.items() if v.numel()}

                B = next(iter(conds.values())).shape[0]

                # initialize action from Gaussian noise
                noisy_action_seq = torch.randn(
                    (B, self.pred_horizon, self.act_dim), device=self.device
                )

                for k in self.noise_scheduler.timesteps:
                    if noisy_action_seq.numel() == 0:
                        # if the mask is empty,
                        break
                    if tuple(sorted(conds.keys())) != tuple(sorted(obs.keys())):
                        # missing an observation
                        break

                    # predict noise
                    ts = k.expand(B).to(self.device)

                    pred = self.pred_net(
                        sample=noisy_action_seq,
                        timestep=ts,
                        conds=conds,
                    )

                    # inverse diffusion step (remove noise)
                    noisy_action_seq = self.noise_scheduler.step(
                        model_output=pred,
                        timestep=k,
                        sample=noisy_action_seq,
                    ).prev_sample
            acts.append(noisy_action_seq)

        noisy_action_seq = torch.zeros(
            sum(a.shape[0] for a in acts),
            *acts[0].shape[1:],
            device=acts[0].device,
            dtype=acts[0].dtype,
        )
        for a, mask in zip(acts, masks, strict=True):
            noisy_action_seq[mask] = a

        noisy_action_seq = self.action_scaler.descale(noisy_action_seq)

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)

    def adapt_actions_from_ds_actions(self, actions):
        return actions[..., : self.act_horizon, :]

    def encode_obs(self, pcd_obs, rgb_obs, pcd_mask, state_obs):
        B = next(iter(rgb_obs.values())).shape[0]

        state_cond, global_rgb_cond, local_rgb_cond, pcd_cond = self.encoder(
            pcd_obs=pcd_obs, rgb_obs=rgb_obs, pcd_mask=pcd_mask, state_obs=state_obs
        )

        if not self.include_rgb_cond_in_3d:
            obs_cond_3d = {"state": state_cond, "pcd": pcd_cond}
            obs_cond_2d = {
                "state": state_cond,
                "rgb_global": global_rgb_cond,
                "rgb_local": local_rgb_cond,
            }
        else:
            obs_cond_3d = {
                "state": state_cond,
                "pcd": pcd_cond,
                "rgb_global": global_rgb_cond,
                "rgb_local": local_rgb_cond,
            }
            obs_cond_2d = {
                "state": state_cond,
                # "pcd": torch.zeros_like(pcd_cond), # ugh we don't need this
                "rgb_global": global_rgb_cond,
                "rgb_local": local_rgb_cond,
            }
        return (
            optree.tree_map(lambda x: x.clone(), obs_cond_3d),
            optree.tree_map(lambda x: x.clone(), obs_cond_2d),
        )

    @property
    def encode_obs_out_shape(self):
        if hasattr(self, "_encode_obs_out_shape_cached"):
            return self._encode_obs_out_shape_cached

        self.to(self.device)
        with torch.no_grad():
            sample_observation = optree.tree_map(
                lambda x: torch.zeros((1, *x), device=self.device), self.observations_shape
            )

            o3, o2 = self.encode_obs(
                sample_observation["pcd_obs"],
                sample_observation["rgb_obs"],
                optree.tree_map(lambda x: ~x.bool(), sample_observation["pcd_mask"]),
                sample_observation["state_obs"],
            )

            if "2d" not in self.train_modes:
                selected_obs = o3
            elif "3d" not in self.train_modes:
                selected_obs = o2
            else:
                common_keys = set(o3.keys()) & set(o2.keys())
                for k in common_keys:
                    assert o3[k].shape[-1] == o2[k].shape[-1], "Output dim mismatch"
                selected_obs = {**o3, **o2}

        self._encode_obs_out_shape_cached = optree.tree_map(lambda x: x.shape, selected_obs)
        return self.encode_obs_out_shape


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

    agent = MantenCombinedPolicy(
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
