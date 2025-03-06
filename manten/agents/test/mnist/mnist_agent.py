import diffusers
import torch

from manten.agents.utils.base_agent import BaseAgent
from manten.metrics.mnist_metric import MNISTImageStats


class MNISTAgent(BaseAgent):
    def __init__(
        self,
        noise_scheduler,
        n_inference_steps,
        in_channels=1,  # for the conditional agent to override, otherwise just 1
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.noise_scheduler = noise_scheduler
        self.n_inference_steps = n_inference_steps
        self.noise_model = diffusers.UNet2DModel(
            sample_size=32,
            in_channels=in_channels,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def train_step(self, batch):
        return self.diffusion_loss(clean_images=batch["image"])

    @torch.no_grad()
    def eval_step(self, batch, *_, **__):
        bs = len(batch["image"])

        sampled_image = self.sample(
            shape=(bs, 1, 32, 32),
            device=self.noise_model.device,
            dtype=self.noise_model.dtype,
        )

        image_metric = MNISTImageStats()
        image_metric.feed(sampled_image)

        return (image_metric, sampled_image)

    def diffusion_loss(self, clean_images):
        x_0 = clean_images

        # Sample random timesteps t for each trajectory
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (x_0.shape[0],),
            device=x_0.device,
        ).long()

        # epsilon@t ~ N(0, 1)
        # noise that would be removed from trajectory@t to get trajectory@t-1
        # noise that would be added to trajectory@t-1 to get trajectory@t
        epsilon_t = torch.randn(x_0.shape, device=x_0.device)

        x_t = self.noise_scheduler.add_noise(x_0, epsilon_t, timesteps)

        epsilon_t_pred = self.noise_model(x_t, timesteps)["sample"]

        self.metric.feed(ground=epsilon_t, prediction=epsilon_t_pred)

        return self.metric

    @torch.no_grad()
    def sample(self, shape, device, dtype):
        sampled_image = torch.randn(size=shape, device=device, dtype=dtype)

        ones = torch.ones(len(sampled_image)).to(sampled_image.device).long()

        self.noise_scheduler.set_timesteps(self.n_inference_steps)
        timesteps = self.noise_scheduler.timesteps

        for t in timesteps:
            epsilon_t_pred = self.noise_model(sampled_image, t * ones)["sample"]
            sampled_image = self.noise_scheduler.step(
                epsilon_t_pred, t, sampled_image
            ).prev_sample

        return sampled_image
