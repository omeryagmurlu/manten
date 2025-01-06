import torch

from manten.agents.lowdim_diffusion_policy.conditional_unet1d import ConditionalUnet1D
from manten.agents.utils.mixins import DatasetActionScalerMixin
from manten.agents.utils.templates import BatchStateObservationActionAgentTemplate
from manten.metrics.traj_action_metric import PosRotGripperMetric, PosRotGripperStats


@BatchStateObservationActionAgentTemplate.make_agent(
    evaluation_metric_cls=PosRotGripperMetric, evaluation_stats_cls=PosRotGripperStats
)
class LowdimDiffusionPolicyAgent(
    BatchStateObservationActionAgentTemplate, DatasetActionScalerMixin
):
    def __init__(
        self,
        *,
        act_horizon,
        noise_scheduler,
        num_diffusion_iters,
        diffusion_step_embed_dim=None,
        unet_dims=None,
        n_groups=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.act_horizon = act_horizon
        self.num_diffusion_iters = num_diffusion_iters
        self.noise_scheduler = noise_scheduler

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=(torch.prod(torch.tensor(self.observations_shape))),
            # obs_horizon * obs_dim
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_dims,
            n_groups=n_groups,
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

    def compute_train_gt_and_pred(self, state_obs, actions):
        obs_seq = state_obs
        action_seq = actions

        B = obs_seq.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=obs_seq.device
        ).long()

        action_seq = self.action_scaler.scale(action_seq)

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, global_cond=obs_cond)

        return noise, noise_pred

    def predict_actions(self, state_obs):
        obs_seq = state_obs

        # init scheduler
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq: (B, obs_horizon, obs_dim)
        B = obs_seq.shape[0]
        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

            # initialize action from Gaussian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq.device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
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
