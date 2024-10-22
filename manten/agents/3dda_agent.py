import torch
from manten.agents.base_agent import BaseAgent
import torch.nn.functional as F


class ThreeDDAAgent(BaseAgent):
    def train_step(self, batch):
        clean_images = batch["images"]
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=clean_images.device,
            dtype=torch.int64,
        )

        # Add noise to the clean images according to the noise magnitude at each timestep (forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        yield  # for the accumulator

        noise_pred = yield noisy_images, timesteps
        yield F.mse_loss(noise_pred, noise)
