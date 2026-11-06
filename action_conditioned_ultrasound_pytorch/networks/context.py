import torch
import torch.nn as nn
from einops import rearrange


class ContextEncoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae.eval()
        self.vae.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images):
        b, t = images.shape[:2]
        images = rearrange(images, 'b t c h w -> (b t) c h w')
        latents = self.vae.encode(images, sample=False)
        return rearrange(latents, '(b t) c h w -> b t c h w', b=b, t=t)
