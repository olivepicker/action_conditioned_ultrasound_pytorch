import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample2x(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Downsample2x(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, drop_rate: float = 0.0):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        if out_channels != in_channels:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut_conv = nn.Identity()

        self.block0 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.shortcut_conv(x) + self.block1(self.block0(x))


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.proj.weight, 0)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)

        q = self.q(x).reshape(b, c, h * w).permute(0, 2, 1)
        k = self.k(x).reshape(b, c, h * w)
        v = self.v(x).reshape(b, c, h * w).permute(0, 2, 1)

        with torch.autocast(device_type=x.device.type, enabled=False):
            q = q.float()
            k = k.float()
            attn = (q @ k) * (c ** -0.5)
            attn = attn.softmax(dim=-1)
            out = attn @ v.float()

        out = out.to(x_in.dtype)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        out = self.proj(out)

        return x_in + out


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        hidden_channels: int = 128,
        latent_dim: int = 256,
        using_mid_attn: bool = True,
    ):
        super().__init__()

        self.init_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)

        self.down0 = nn.Sequential(
            ResBlock(hidden_channels * 1, hidden_channels * 1),
            ResBlock(hidden_channels * 1, hidden_channels * 1),
            Downsample2x(hidden_channels * 1),
        )
        self.down1 = nn.Sequential(
            ResBlock(hidden_channels * 1, hidden_channels * 2),
            ResBlock(hidden_channels * 2, hidden_channels * 2),
            Downsample2x(hidden_channels * 2),
        )
        self.down2 = nn.Sequential(
            ResBlock(hidden_channels * 2, hidden_channels * 4),
            ResBlock(hidden_channels * 4, hidden_channels * 4),
            Downsample2x(hidden_channels * 4),
        )
        self.down3 = nn.Sequential(
            ResBlock(hidden_channels * 4, hidden_channels * 8),
            ResBlock(hidden_channels * 8, hidden_channels * 8),
            Downsample2x(hidden_channels * 8),
        )

        self.mid_block1 = ResBlock(hidden_channels * 8, hidden_channels * 8)
        self.mid_attn = AttnBlock(hidden_channels * 8) if using_mid_attn else nn.Identity()
        self.mid_block2 = ResBlock(hidden_channels * 8, hidden_channels * 8)

        self.to_latent = nn.Sequential(
            nn.GroupNorm(32, hidden_channels * 8),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels * 8, latent_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x0 = self.init_conv(x)
        x1 = self.down0(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.mid_block2(self.mid_attn(self.mid_block1(x4)))
        f = self.to_latent(x4)
        return f


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        out_channels: int = 1,
        hidden_channels: int = 128,
        latent_dim: int = 256,
        using_mid_attn: bool = True,
    ):
        super().__init__()

        self.init_conv = nn.Conv2d(latent_dim, hidden_channels * 8, kernel_size=3, stride=1, padding=1)

        self.mid_block1 = ResBlock(hidden_channels * 8, hidden_channels * 8)
        self.mid_attn = AttnBlock(hidden_channels * 8) if using_mid_attn else nn.Identity()
        self.mid_block2 = ResBlock(hidden_channels * 8, hidden_channels * 8)

        self.up0 = nn.Sequential(
            ResBlock(hidden_channels * 8, hidden_channels * 8),
            ResBlock(hidden_channels * 8, hidden_channels * 8),
        )
        self.up1 = nn.Sequential(
            Upsample2x(hidden_channels * 8),
            ResBlock(hidden_channels * 8, hidden_channels * 4),
            ResBlock(hidden_channels * 4, hidden_channels * 4),
        )
        self.up2 = nn.Sequential(
            Upsample2x(hidden_channels * 4),
            ResBlock(hidden_channels * 4, hidden_channels * 2),
            ResBlock(hidden_channels * 2, hidden_channels * 2),
        )
        self.up3 = nn.Sequential(
            Upsample2x(hidden_channels * 2),
            ResBlock(hidden_channels * 2, hidden_channels * 1),
            ResBlock(hidden_channels * 1, hidden_channels * 1),
        )
        self.up4 = nn.Sequential(
            Upsample2x(hidden_channels * 1),
            ResBlock(hidden_channels * 1, hidden_channels * 1),
            ResBlock(hidden_channels * 1, hidden_channels * 1),
        )

        self.to_image = nn.Sequential(
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, f):
        x4 = self.init_conv(f)
        x4 = self.mid_block2(self.mid_attn(self.mid_block1(x4)))
        x3 = self.up0(x4)
        x2 = self.up1(x3)
        x1 = self.up2(x2)
        x0 = self.up3(x1)
        x = self.up4(x0)
        return self.to_image(x)

class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        latent_dim: int = 4,
        kl_weight: float = 1e-4,
        recon_loss_type: str = 'l1',
        using_mid_attn: bool = True,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.recon_loss_type = recon_loss_type

        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim * 2,
            using_mid_attn=using_mid_attn,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            using_mid_attn=using_mid_attn,
        )

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def encode(self, x, sample=False):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return self.reparameterize(mu, logvar) if sample else mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)

        if self.recon_loss_type == 'l1':
            recon_loss = F.l1_loss(x_hat, x)
        else:
            recon_loss = F.mse_loss(x_hat, x)

        kl_loss = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).mean()
        loss = recon_loss + self.kl_weight * kl_loss

        return {
            'x_hat': x_hat,
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'loss': loss,
        }
