
import torch
from torch import nn


class DiffusionSchedule(nn.Module):
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas_cumprod = torch.cumprod(1 - betas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1 - alphas_cumprod).sqrt())

    def sample_timesteps(self, batch_size, device):
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    def add_noise(self, x_start, noise, timestep):
        shape = (-1,) + (1,) * (x_start.ndim - 1)

        alpha = self.sqrt_alphas_cumprod[timestep].reshape(shape).to(x_start.dtype)
        sigma = self.sqrt_one_minus_alphas_cumprod[timestep].reshape(shape).to(x_start.dtype)

        return alpha * x_start + sigma * noise


class DDIMSampler:
    def __init__(self, schedule, num_steps=50):
        assert 1 <= num_steps <= schedule.num_timesteps

        self.schedule = schedule
        self.timesteps = torch.linspace(
            schedule.num_timesteps - 1,
            0,
            num_steps
        ).round().long().tolist()

    @torch.no_grad()
    def sample(self, dit, context_latents, action, time_offset):
        b, _, c, h, w = context_latents.shape

        latent = torch.randn(
            b, c, h, w,
            device=context_latents.device,
            dtype=context_latents.dtype
        )

        alphas = self.schedule.alphas_cumprod

        for i, t in enumerate(self.timesteps):
            timestep = torch.full(
                (b,), t,
                device=latent.device,
                dtype=torch.long
            )

            pred_noise = dit(
                latent,
                context_latents,
                timestep,
                action,
                time_offset
            )

            alpha = alphas[t].to(latent.dtype)

            if i + 1 < len(self.timesteps):
                alpha_prev = alphas[self.timesteps[i + 1]].to(latent.dtype)
            else:
                alpha_prev = torch.ones_like(alpha)

            pred_clean = (
                latent - (1 - alpha).sqrt() * pred_noise
            ) / alpha.sqrt()

            latent = (
                alpha_prev.sqrt() * pred_clean
                + (1 - alpha_prev).sqrt() * pred_noise
            )

        return latent
