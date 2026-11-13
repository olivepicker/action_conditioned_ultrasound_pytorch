import torch
import torch.nn as nn

from .embeddings import PatchEmbedding, PositionalEmbedding, TimeOffsetEmbedding
from .transformer import TransformerBlock


class GoalConditionedPolicy(nn.Module):
    def __init__(
        self,
        action_scales,
        previous_action_scales,
        latent_channels=4,
        context_length=3,
        dim=256,
        depth=4,
        patch_size=2,
        dim_head=32,
        num_heads=8,
        goal_dropout=0.1
    ):
        super().__init__()
        self.context_length = context_length
        self.patch_size = patch_size
        self.goal_dropout = goal_dropout

        action_scales = torch.as_tensor(action_scales, dtype=torch.float32)
        previous_action_scales = torch.as_tensor(previous_action_scales, dtype=torch.float32)

        if action_scales.shape != (6,) or previous_action_scales.shape != (6,):
            raise ValueError('Action scales must have shape (6,)')

        if (action_scales <= 0).any() or (previous_action_scales <= 0).any():
            raise ValueError('Action scales must be positive')

        self.register_buffer('action_scales', action_scales.clone())
        self.register_buffer('previous_action_scales', previous_action_scales.clone())

        self.patch_emb = PatchEmbedding(latent_channels, dim, patch_size=patch_size)
        self.pos_emb = PositionalEmbedding(dim)
        self.time_emb = TimeOffsetEmbedding(dim)

        self.previous_action_emb = nn.Sequential(
            nn.Linear(6, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        self.context_time_emb = nn.Parameter(
            torch.randn(1, context_length, 1, dim) * 0.02
        )
        self.previous_time_emb = nn.Parameter(
            torch.randn(1, context_length - 1, dim) * 0.02
        )
        self.goal_type_emb = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.null_goal_emb = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, dim_head=dim_head, num_heads=num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.action_head = nn.Linear(dim, 6)

    def forward(
        self,
        context_latents,
        previous_actions,
        goal_latent,
        time_offset,
        use_goal=True
    ):
        b, t, c, h, w = context_latents.shape

        if t != self.context_length:
            raise ValueError('Unexpected context length')

        if previous_actions.shape != (b, t - 1, 6):
            raise ValueError('Unexpected previous action shape')

        if goal_latent.shape != (b, c, h, w):
            raise ValueError('Unexpected goal latent shape')

        gh, gw = h // self.patch_size, w // self.patch_size

        context = self.pos_emb(self.patch_emb(context_latents), gh, gw)
        context = (context + self.context_time_emb).reshape(b, -1, context.shape[-1])

        previous = self.previous_action_emb(
            previous_actions / self.previous_action_scales
        ) + self.previous_time_emb

        goal = self.pos_emb(self.patch_emb(goal_latent), gh, gw) + self.goal_type_emb
        null_goal = self.null_goal_emb.expand_as(goal)

        if not use_goal:
            goal = null_goal
        elif self.training and self.goal_dropout > 0:
            mask = torch.rand(b, 1, 1, device=goal.device) < self.goal_dropout
            goal = torch.where(mask, null_goal, goal)

        cls = self.cls_token.expand(b, -1, -1) + self.time_emb(time_offset).unsqueeze(1)

        x = torch.cat([cls, context, previous, goal], dim=1)

        for block in self.blocks:
            x = block(x)

        normalized_action = self.action_head(self.norm(x[:, 0]))

        return {
            'normalized_action': normalized_action,
            'action': normalized_action * self.action_scales
        }