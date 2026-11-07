
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
