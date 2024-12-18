import torch
from diffusers import UNet2DModel
from torch import nn

from manten.agents.base_agent import BaseAgent
from manten.metrics.mnist_metric import MNISTConditionalImageStats


class ClassConditionedUnet(nn.Module):
    def __init__(self, *, num_classes, class_emb_size, sample_size=28):
        super().__init__()

        # The embedding layer will map the class label to a vector of size class_emb_size
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=sample_size,  # the target image resolution
            in_channels=1 + class_emb_size,  # Additional input channels for class cond.
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64),
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",  # a regular ResNet upsampling block
            ),
        )

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x, t, class_labels):
        # Shape of x:
        bs, ch, w, h = x.shape

        # class conditioning in right shape to add as additional input channels
        class_cond = self.class_emb(class_labels)  # Map to embedding dimension
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(
            bs, class_cond.shape[1], w, h
        )
        # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x, class_cond), 1)  # (bs, 5, 28, 28)

        # Feed this to the UNet alongside the timestep and return the prediction
        return self.model(net_input, t).sample  # (bs, 1, 28, 28)

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype


class MNISTConditionalAgent(BaseAgent):
    def __init__(
        self, *, num_classes, noise_scheduler, n_inference_steps, cls_emb_dim=4, **kwargs
    ):
        super().__init__(**kwargs)
        self.noise_scheduler = noise_scheduler
        self.n_inference_steps = n_inference_steps
        self.noise_model = ClassConditionedUnet(
            num_classes=num_classes, class_emb_size=cls_emb_dim
        )

    def train_step(self, batch):
        return self.conditional_diffusion_loss(
            clean_images=batch["image"],
            class_labels=batch["label"],
        )

    @torch.no_grad()
    def eval_step(self, batch, *_, **__):
        bs = len(batch["image"])
        class_label = batch["label"]

        sampled_image = self.conditional_sample(
            class_label,
            shape=(bs, 1, 28, 28),
            device=self.noise_model.device,
            dtype=self.noise_model.dtype,
        )

        image_metric = MNISTConditionalImageStats()
        image_metric.feed((sampled_image, class_label))

        return (image_metric, sampled_image)

    def conditional_diffusion_loss(self, clean_images, class_labels):
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

        epsilon_t_pred = self.noise_model(x_t, timesteps, class_labels)

        self.metric.feed(ground=epsilon_t, prediction=epsilon_t_pred)

        return self.metric

    @torch.no_grad()
    def conditional_sample(self, class_label, shape, device, dtype):
        sampled_image = torch.randn(size=shape, device=device, dtype=dtype)

        ones = torch.ones(len(sampled_image)).to(sampled_image.device).long()

        self.noise_scheduler.set_timesteps(self.n_inference_steps)
        timesteps = self.noise_scheduler.timesteps

        for t in timesteps:
            epsilon_t_pred = self.noise_model(sampled_image, t * ones, class_label)
            sampled_image = self.noise_scheduler.step(
                epsilon_t_pred, t, sampled_image
            ).prev_sample

        return sampled_image
