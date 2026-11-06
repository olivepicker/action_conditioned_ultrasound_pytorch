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
