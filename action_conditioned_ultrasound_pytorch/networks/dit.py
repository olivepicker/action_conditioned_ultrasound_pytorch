
import torch
import torch.nn as nn
from einops import rearrange

from .embeddings import PatchEmbedding, PositionalEmbedding, TimestepEmbedding, ActionEmbedding, TimeOffsetEmbedding
from .transformer import DiTBlock


class ConditionalDiT(nn.Module):
    def __init__(
        self,
        latent_channels,
        dim,
        depth,
        context_length,
        translation_scale,
        rotation_scale,
        patch_size=2,
        dim_head=64,
        num_heads=8,
        ff_mult=4
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.context_length = context_length

        self.patch_emb = PatchEmbedding(latent_channels, dim, patch_size=patch_size)
        self.pos_emb = PositionalEmbedding(dim)
        self.temporal_emb = nn.Parameter(
            torch.randn(1, context_length + 1, 1, dim) * 0.02
        )

        self.timestep_emb = TimestepEmbedding(dim)
        self.action_emb = ActionEmbedding(dim, translation_scale, rotation_scale)
        self.time_emb = TimeOffsetEmbedding(dim)

        self.blocks = nn.ModuleList([
            DiTBlock(dim, dim_head=dim_head, num_heads=num_heads, ff_mult=ff_mult)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.out = nn.Linear(dim, latent_channels * patch_size ** 2)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, noisy_latent, context_latents, timestep, action, time_offset):
        b, c, h, w = noisy_latent.shape
        gh, gw = h // self.patch_size, w // self.patch_size

        context = self.pos_emb(self.patch_emb(context_latents), gh, gw)
        future = self.pos_emb(self.patch_emb(noisy_latent), gh, gw)

        x = torch.cat([context, future.unsqueeze(1)], dim=1)
        x = x + self.temporal_emb
        x = rearrange(x, 'b t n d -> b (t n) d')

        cond = self.timestep_emb(timestep) + self.action_emb(action) + self.time_emb(time_offset)

        for block in self.blocks:
            x = block(x, cond)

        x = self.out(self.norm(x[:, -gh * gw:]))
        x = rearrange(
            x, 'b (h w) (p q c) -> b c (h p) (w q)',
            h=gh, w=gw, p=self.patch_size, q=self.patch_size, c=self.latent_channels
        )

        return x
