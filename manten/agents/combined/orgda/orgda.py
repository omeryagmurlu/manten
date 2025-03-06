import einops
import optree
import torch

from manten.agents.utils.mixins import DatasetActionScalerMixin, DatasetPCDScalerMixin
from manten.agents.utils.templates import BatchPCDOrRGBObservationActionAgentTemplate
from manten.metrics.traj_action_metric import (
    PosRotGripperMetric,
    PosRotGripperStats,
)
from manten.utils.utils_pytorch import get_ones_shape_like


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


@BatchPCDOrRGBObservationActionAgentTemplate.make_agent(
    evaluation_metric_cls=PosRotGripperMetric, evaluation_stats_cls=PosRotGripperStats
)
class OrgDA(
    BatchPCDOrRGBObservationActionAgentTemplate,
    DatasetActionScalerMixin,
    DatasetPCDScalerMixin,
):
    def __init__(
        self,
        *,
        position_noise_scheduler,
        rotation_noise_scheduler,
        sigmoid_openness_in_inference,  # needed if bce _with logits_ is used
        relative,  # center the pcd around tcp
        act_horizon,
        _embedding_dim,
        n_inference_steps=10,
        tcp_pose_key=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.position_noise_scheduler = position_noise_scheduler
        self.rotation_noise_scheduler = rotation_noise_scheduler

        self.relative = relative
        self.n_inference_steps = n_inference_steps
        self.act_horizon = act_horizon
        self.sigmoid_openness_in_inference = sigmoid_openness_in_inference

        # This won't work for now, since we'd need to the pcd_scale _before_ calculating actions_stats
        # otherwise actions don't line in -1 1 range
        # this highlights that the proper place to do any kind of pcd scaling is in the
        # dataset / agent wrapper, but maybe later

        # if (
        #     scale_actions_by_pcd is not False
        #     or scale_actions_by_pcd != "delta"
        #     or scale_actions_by_pcd != "absolute"
        # ):
        #     raise ValueError(f"Unsupported scale_actions_by_pcd value {scale_actions_by_pcd}")
        # self.scale_actions_by_pcd = scale_actions_by_pcd

        if tcp_pose_key is None:
            self.tcp_pose_key = self.dataset_info["tcp_pose_key"]
        else:
            self.tcp_pose_key = tcp_pose_key

        remaining_state_keys = set(self.observations_shape["state_obs"].keys())
        remaining_state_keys.remove(self.tcp_pose_key)
        self.encoder_custom_state_shapes = {
            key: self.observations_shape["state_obs"][key][1:] for key in remaining_state_keys
        }

        assert self.act_dim == 3 + self.rotation_dim + 1, "Action dim mismatch"

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

    def compute_train_gt_and_pred(
        self, pcd_obs, rgb_obs, pcd_mask, state_obs, actions, keep_mask_3d
    ):
        # rgb_obs (B, obs_horizon, cam, C, H, W)
        norm_actions = self.action_scaler.scale(actions)

        indices_3d = keep_mask_3d.nonzero()

        B = actions.shape[0]
        conditions_3d, conditions_2d = self.encode_observations(
            pcd_obs, rgb_obs, pcd_mask, state_obs
        )
        conditions = {"2d": conditions_2d, "3d": conditions_3d}

        noise_pos_rot = torch.randn(
            (B, self.pred_horizon, self.act_dim - 1), device=actions.device
        )

        timesteps = torch.randint(
            0,
            self.position_noise_scheduler.config.num_train_timesteps,
            (B,),
            device=actions.device,
        ).long()

        noisy_position = self.position_noise_scheduler.add_noise(
            norm_actions[..., :3], noise_pos_rot[..., :3], timesteps
        )
        noisy_rotation = self.rotation_noise_scheduler.add_noise(
            norm_actions[..., 3:-1], noise_pos_rot[..., 3:None], timesteps
        )

        noisy_rot_pos = torch.cat((noisy_position, noisy_rotation), -1)  # don't add openness

        prediction_by_vis_mode = {}
        for mode in ["2d", "3d"]:
            (pos_pred, rot_pred, openness_pred) = self.noise_model(
                trajectory=noisy_rot_pos.clone(),  # clone because I'm not sure if noise_model modifies in place without clone
                timestep=timesteps.clone(),
                **conditions[mode],
            )

            pred = torch.cat((pos_pred, rot_pred, openness_pred), -1)
            prediction_by_vis_mode[mode] = pred

        pred_type = self.position_noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            ret_gt = torch.cat((noise_pos_rot, norm_actions[..., -1:]), -1)
        elif pred_type == "sample":
            ret_gt = norm_actions
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        indices_3d = keep_mask_3d.nonzero()
        # an optimization would be moving this to the top, but eh

        gt_modes = {
            "2d": ret_gt,
            "3d": ret_gt[indices_3d],
        }
        pred_modes = {
            "2d": prediction_by_vis_mode["2d"],
            "2d_for_3d": prediction_by_vis_mode["2d"][indices_3d],  # for consistency
            "3d": prediction_by_vis_mode["3d"][indices_3d],
        }

        return gt_modes, pred_modes

    def predict_actions(self, pcd_obs, rgb_obs, pcd_mask, state_obs, keep_mask_3d):
        self.position_noise_scheduler.set_timesteps(self.n_inference_steps)
        self.rotation_noise_scheduler.set_timesteps(self.n_inference_steps)

        with torch.no_grad():
            conditions_3d, conditions_2d = self.encode_observations(
                pcd_obs, rgb_obs, pcd_mask, state_obs
            )
            conditions = optree.tree_map(
                lambda c2, c3: torch.where(
                    ~keep_mask_3d.view(keep_mask_3d.shape[0], *get_ones_shape_like(c3)[1:]),
                    c2,
                    c3,
                ),
                conditions_2d,
                conditions_3d,
            )
            sample_tensor = next(iter(pcd_obs.values()))
            B = sample_tensor.shape[0]
            device = sample_tensor.device

            traj_rot_pos = torch.randn(
                (B, self.pred_horizon, self.act_dim - 1), device=device
            )

            for t in self.position_noise_scheduler.timesteps:
                (epsilon_t_pos_pred, epsilon_t_rot_pred, openness_pred) = self.noise_model(
                    trajectory=traj_rot_pos,
                    timestep=t * torch.ones(B, device=device, dtype=torch.long),
                    **conditions,
                )

                pos = self.position_noise_scheduler.step(
                    epsilon_t_pos_pred, t, traj_rot_pos[..., :3]
                ).prev_sample
                rot = self.rotation_noise_scheduler.step(
                    epsilon_t_rot_pred, t, traj_rot_pos[..., 3:]
                ).prev_sample
                traj_rot_pos = torch.cat((pos, rot), -1)

            if self.sigmoid_openness_in_inference:
                openness_pred = torch.sigmoid(openness_pred)
                # also map to -1 1
                openness_pred = 2 * openness_pred - 1

            complete_traj = torch.cat((traj_rot_pos, openness_pred), -1)
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
        context_keep_mask = mask_pyramid[0]
        context_feats = rgb_feats_pyramid[0] * context_keep_mask
        context = einops.rearrange(pcd_pyramid[0], "bt ncam c h w -> bt (ncam h w) c")

        context_feats_2d = rgb_feats_pyramid[0]
        context_2D = einops.rearrange(
            self.centered_2d_meshgrid_like(pcd_pyramid[0]), "b ncam c h w -> b (ncam h w) c"
        )

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
        adaln_gripper_feats_2D, _ = self.encoder.encode_curr_gripper(
            curr_gripper, context_feats_2d, context_2D, use_2d_pe=True
        )

        context_pe = self.encoder.relative_pe_layer(context)
        context_2d_pe = self.encoder.relative_pe_layer_2D(context_2D)

        # FPS on visual features (N, B, F) and (B, N, F, 2)
        fps_feats, fps_pos = self.encoder.run_fps(
            context_feats.transpose(0, 1),
            context_pe,
            keep_mask=context_keep_mask.transpose(0, 1),
        )

        # fps on 2D is dumb, (since you can simply downsample) but let's do it for consistency
        # can substitute with a simple downsample later
        fps_feats_2d, fps_pos_2d = self.encoder.run_fps(
            context_feats_2d.transpose(0, 1),
            context_2d_pe,
        )
        return (
            {
                "context_feats": context_feats,
                "context": context_pe,  # contextualized visual features
                "instr_feats": instr_feats,  # language features
                "adaln_gripper_feats": adaln_gripper_feats,  # gripper history features
                "fps_feats": fps_feats,
                "fps_pos": fps_pos,  # sampled visual features
            },
            {
                "context_feats": context_feats_2d,
                "context": context_2d_pe,  # contextualized visual features
                "instr_feats": instr_feats,  # language features
                "adaln_gripper_feats": adaln_gripper_feats_2D,  # gripper history features
                "fps_feats": fps_feats_2d,
                "fps_pos": fps_pos_2d,  # sampled visual features
            },
        )

    @staticmethod
    def centered_2d_meshgrid_like(tensor):
        B, ncam, C, H, W = tensor.shape
        x = torch.linspace(-1, 1, W, device=tensor.device)
        y = torch.linspace(-1, 1, H, device=tensor.device)
        x, y = torch.meshgrid(x, y, indexing="xy")
        x = x.unsqueeze(0).unsqueeze(0).expand(B, ncam, -1, -1)
        y = y.unsqueeze(0).unsqueeze(0).expand(B, ncam, -1, -1)
        return torch.stack((x, y), -3)


if __name__ == "__main__":
    from functools import partial

    from diffusers.schedulers import DDPMScheduler

    from manten.agents.utils.normalization import MinMaxScaler
    from manten.data.dataset_maniskill import ManiSkillDataset
    from manten.metrics.combined_2d_3d_metric import Combined2D3DMetric
    from manten.metrics.mse_loss_pose_bce_loss_gripper_metric import (
        MSELossPoseBCELossGripperMetric,
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

    metric = Combined2D3DMetric(
        metric_action_2d=MSELossPoseBCEWithLogitsLossGripperMetric(),
        metric_action_3d=MSELossPoseBCEWithLogitsLossGripperMetric(),
        metric_action_consistency=MSELossPoseBCELossGripperMetric(),
    )

    agent = OrgDA(
        dataset_info=dataset_info,
        position_noise_scheduler=ddpm(),
        rotation_noise_scheduler=ddpm(),
        encoder=encoder,
        metric=metric,
        noise_model=noise_model,
        action_scaler=MinMaxScaler,
        relative=True,
        n_inference_steps=10,
        sigmoid_openness_in_inference=True,
        embedding_dim=192,
        act_horizon=8,
    )
    agent.to("cuda")  # flash-attn not implemented for cpu

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
    actions = agent.predict_actions(
        observations=batch["observations"], meta={"3d_mask": F, "2d_mask": T}
    )
    print(actions.shape)
    print(actions)
