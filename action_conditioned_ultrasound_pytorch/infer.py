import torch
import torch.nn as nn

from .diffusion import DDIMSampler
from .networks.context import ContextEncoder


class WorldModelInfer(nn.Module):
    def __init__(self, vae, world_model, num_steps=50):
        super().__init__()
        self.context_encoder = ContextEncoder(vae)
        self.world_model = world_model
        self.sampler = DDIMSampler(
            world_model.diffusion,
            num_steps=num_steps
        )

    @torch.no_grad()
    def forward(self, context_images, action, time_offset):
        context_latents = self.context_encoder(context_images)

        future_latent = self.sampler.sample(
            self.world_model.dit,
            context_latents,
            action,
            time_offset
        )

        future_image = self.context_encoder.vae.decode(future_latent)

        return {
            'future_image': future_image,
            'future_latent': future_latent
        }
