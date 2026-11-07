import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class AdaLayerNorm(nn.Module):
    def __init__(self, dim, cond_dim=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim or dim, dim * 2)
        )

        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, cond):
        shift, scale = self.modulation(cond).chunk(2, dim=-1)

        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        num_heads=8
    ):
        super().__init__()
        self.num_heads = num_heads

        inner_dim = dim_head * num_heads
        self.q = nn.Linear(dim, inner_dim, bias=False)
        self.kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, context=None, attn_mask=None):
        context = x if context is None else context

        q = self.q(x)
        k, v = self.kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),(q, k, v))

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask[None, None]
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask[:, None]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.out(out)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)

        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult * 2 / 3)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias=False),
            GEGLU(),
            nn.Linear(inner_dim, dim, bias=False)
        )

    def forward(self, x):
        
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, num_heads=8, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, dim_head=dim_head, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult)

    def forward(self, x, context=None, attn_mask=None):
        x = x + self.attn(self.norm1(x), context=context, attn_mask=attn_mask)
        x = x + self.ff(self.norm2(x))

        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        num_heads=8,
        ff_mult=4
    ):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim)
        self.attn = Attention(dim, dim_head=dim_head, num_heads=num_heads)
        self.norm2 = AdaLayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult)

        self.gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2)
        )

        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x, cond):
        gate1, gate2 = self.gate(cond).chunk(2, dim=-1)

        x = x + gate1.unsqueeze(1) * self.attn(self.norm1(x, cond))
        x = x + gate2.unsqueeze(1) * self.ff(self.norm2(x, cond))

        return x
