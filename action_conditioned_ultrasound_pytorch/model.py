import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    def __init__(self, dit, diffusion):
        super().__init__()
        self.dit = dit
        self.diffusion = diffusion

    def forward(self, future_latent, context_latents, action, time_offset):
        timestep = self.diffusion.sample_timesteps(
            future_latent.shape[0],
            future_latent.device
        )

        noise = torch.randn_like(future_latent)
        noisy_latent = self.diffusion.add_noise(future_latent, noise, timestep)

        pred_noise = self.dit(
            noisy_latent,
            context_latents,
            timestep,
            action,
            time_offset
        )

        loss = F.mse_loss(pred_noise, noise)

        return {
            'loss': loss,
            'pred_noise': pred_noise,
            'noise': noise,
            'timestep': timestep
        }
