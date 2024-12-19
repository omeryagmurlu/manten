import numpy as np
import torch

from manten.agents.base_agent import BaseAgent
from manten.agents.lowdim_diffusion_policy.conditional_unet1d import ConditionalUnet1D
from manten.metrics.dummy_metric import PosTrajMetric


class LowdimDiffusionPolicyAgent(BaseAgent):
    def __init__(
        self,
        *,
        obs_horizon,
        act_horizon,
        pred_horizon,
        act_dim,
        observation_shape,
        noise_scheduler,
        num_diffusion_iters,
        diffusion_step_embed_dim=None,
        unet_dims=None,
        n_groups=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.num_diffusion_iters = num_diffusion_iters

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=(  # because of sliding window
                obs_horizon * np.prod(observation_shape)
            ),
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_dims,
            n_groups=n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = noise_scheduler

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=obs_seq.device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, global_cond=obs_cond)

        self.metric.feed(ground=noise, prediction=noise_pred)

        return self.metric

    def get_action(self, obs_seq):
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

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)

    def train_step(self, batch):
        obs_seq = batch["observations"]["state_obs"]
        action_seq = batch["actions"]
        return self.compute_loss(obs_seq, action_seq)

    @torch.no_grad()
    def eval_step(self, batch, *_, **__):
        obs_seq = batch["observations"]["state_obs"]
        action = self.get_action(obs_seq)

        metric = PosTrajMetric()
        metric.feed(ground=batch["actions"][..., : self.act_horizon, :], prediction=action)
        return metric, action
