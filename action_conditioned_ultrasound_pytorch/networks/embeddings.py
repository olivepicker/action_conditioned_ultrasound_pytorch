import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
