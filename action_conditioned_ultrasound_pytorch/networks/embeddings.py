import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_dim=256, max_period=10000):
        super().__init__()

        self.freq_dim = freq_dim
        half_dim = freq_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half_dim).float() / half_dim)
        self.register_buffer('freqs', freqs, persistent=False)

        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        t = t.float().unsqueeze(-1) * self.freqs
        emb = torch.cat([t.cos(), t.sin()], dim=-1)
        if self.freq_dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class ActionEmbedding(nn.Module):
    def __init__(self, dim, translation_scale, rotation_scale, num_frequencies=8):
        super().__init__()

        self.register_buffer('scales', torch.tensor(
            [translation_scale] * 3 + [rotation_scale] * 3, dtype=torch.float32
        ), persistent=False)
        self.register_buffer('frequencies', 2. ** torch.arange(num_frequencies).float(), persistent=False)

        self.mlp = nn.Sequential(
            nn.Linear(12 * num_frequencies, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, action):
        action = action / self.scales
        angles = math.pi * action.unsqueeze(-2) / self.frequencies.view(-1, 1)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.mlp(emb.flatten(-2))


class TimeOffsetEmbedding(TimestepEmbedding):
    def __init__(self, dim, freq_dim=256, max_period=10000, time_scale=1000.0):
        super().__init__(dim, freq_dim=freq_dim, max_period=max_period)
        self.time_scale = time_scale

    def forward(self, time_offset):
        return super().forward(time_offset * self.time_scale)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, dim, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)

    def forward(self, x):
        assert x.shape[-2] % self.patch_size == 0 and x.shape[-1] % self.patch_size == 0

        if x.ndim == 5:
            b, t = x.shape[:2]
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            x = self.proj(x)
            return rearrange(x, '(b t) d h w -> b t (h w) d', b=b, t=t)

        x = self.proj(x)
        return rearrange(x, 'b d h w -> b (h w) d')


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        assert dim % 4 == 0

        n = dim // 4
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(n).float() / n
        )
        self.register_buffer('freqs', freqs, persistent=False)

    def forward(self, tokens, height, width):
        assert tokens.shape[-2] == height * width

        y, x = torch.meshgrid(
            torch.arange(height, device=tokens.device),
            torch.arange(width, device=tokens.device),
            indexing='ij'
        )

        y = y.reshape(-1, 1) * self.freqs
        x = x.reshape(-1, 1) * self.freqs

        pos = torch.cat(
            [y.sin(), y.cos(), x.sin(), x.cos()], dim=-1
        )

        return tokens + pos.to(dtype=tokens.dtype)