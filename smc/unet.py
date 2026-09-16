import torch
import torch.nn as nn
import math


def gn(channels, max_groups=32):
    g = max_groups
    while channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, channels)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        assert dim % 2 == 0, "dim doit être pair"
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        angles = t.float()[:, None] * freqs[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.norm1 = gn(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = gn(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()

        self.skip = (nn.Conv2d(in_channels, out_channels, 1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = gn(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x).view(b, c, h * w).permute(0, 2, 1)
        y, _ = self.attn(y, y, y)
        return x + y.permute(0, 2, 1).view(b, c, h, w)


class UNet(nn.Module):
    def __init__(self, in_channels=3, n_feat=64):
        super().__init__()
        time_dim = n_feat * 4

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(n_feat),
            nn.Linear(n_feat, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, n_feat, 3, padding=1)

        # Encoder            32x32
        self.down_block1 = ResidualBlock(n_feat, n_feat, time_dim)
        self.down_conv1 = nn.Conv2d(n_feat, n_feat, 3, stride=2, padding=1)   # -> 16x16
        self.down_block2 = ResidualBlock(n_feat, 2 * n_feat, time_dim)
        self.down_attn = SelfAttention(2 * n_feat)
        self.down_conv2 = nn.Conv2d(2 * n_feat, 2 * n_feat, 3, stride=2, padding=1)  # -> 8x8

        # Bottleneck          8x8
        self.mid_block1 = ResidualBlock(2 * n_feat, 2 * n_feat, time_dim)
        self.mid_attn = SelfAttention(2 * n_feat)
        self.mid_block2 = ResidualBlock(2 * n_feat, 2 * n_feat, time_dim)

        # Decoder
        self.up_conv2 = nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, stride=2, padding=1)
        self.up_block2 = ResidualBlock(4 * n_feat, n_feat, time_dim)
        self.up_attn = SelfAttention(n_feat)
        self.up_conv1 = nn.ConvTranspose2d(n_feat, n_feat, 4, stride=2, padding=1)
        self.up_block1 = ResidualBlock(2 * n_feat, n_feat, time_dim)

        self.out_norm = gn(n_feat)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(n_feat, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)      # départ à epsilon=0, stabilise le début
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        h = self.init_conv(x)
        h1 = self.down_block1(h, t_emb)
        h = self.down_conv1(h1)
        h = self.down_block2(h, t_emb)
        h2 = self.down_attn(h)
        h = self.down_conv2(h2)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        h = self.up_conv2(h)
        h = self.up_block2(torch.cat([h, h2], dim=1), t_emb)
        h = self.up_attn(h)
        h = self.up_conv1(h)
        h = self.up_block1(torch.cat([h, h1], dim=1), t_emb)

        return self.out_conv(self.out_act(self.out_norm(h)))