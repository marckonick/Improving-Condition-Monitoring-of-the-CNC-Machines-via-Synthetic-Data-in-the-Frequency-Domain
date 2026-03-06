import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    timesteps: (B,)
    returns:   (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half - 1))
    args = timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb



class ConvBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8):
        super().__init__()
        g = groups if (out_ch % groups == 0) else 1

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(g, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g, out_ch)

        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        cond = self.cond_proj(cond_vec).unsqueeze(-1).unsqueeze(-1)

        h = self.conv1(x)
        h = self.norm1(h)
        h = h + cond
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = h + cond
        h = self.act(h)
        return h



class UNet2D(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        time_emb_dim: int = 128,
        num_ops: int = 11,
        op_emb_dim: int = 32,
    ):
        super().__init__()

        self.num_ops = num_ops  # number of *real* operation classes

        # ---- Embeddings (time + operation) ----
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, 4 * time_emb_dim),
            nn.SiLU(),
            nn.Linear(4 * time_emb_dim, time_emb_dim),
        )

        # +1 for the null / unconditional class
        self.op_emb = nn.Embedding(num_ops + 1, op_emb_dim)
        # Initialize the null embedding to zeros so it starts as "no info"
        nn.init.zeros_(self.op_emb.weight[num_ops])

        cond_dim = time_emb_dim + op_emb_dim
        C = base_channels

        # ---- Encoder ----
        self.down1 = ConvBlock2D(in_channels, C, cond_dim)
        self.down2 = ConvBlock2D(C, 2 * C, cond_dim)
        self.down3 = ConvBlock2D(2 * C, 4 * C, cond_dim)
        self.down4 = ConvBlock2D(4 * C, 8 * C, cond_dim)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---- Bottleneck ----
        self.bottleneck = ConvBlock2D(8 * C, 8 * C, cond_dim)

        # ---- Decoder ----
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(8 * C, 4 * C, 3, padding=1))
        self.dec4 = ConvBlock2D(4 * C + 8 * C, 4 * C, cond_dim)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(4 * C, 2 * C, 3, padding=1))
        self.dec3 = ConvBlock2D(2 * C + 4 * C, 2 * C, cond_dim)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(2 * C, C, 3, padding=1))
        self.dec2 = ConvBlock2D(C + 2 * C, C, cond_dim)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(C, C, 3, padding=1))
        self.dec1 = ConvBlock2D(C + C, C, cond_dim)
        self.final_conv = nn.Conv2d(C, in_channels, kernel_size=3, padding=1)

    @property
    def null_class_idx(self) -> int:
        """Index used for the unconditional (null) embedding."""
        return self.num_ops

    def forward(self, x: torch.Tensor, t: torch.Tensor, op_id: torch.Tensor) -> torch.Tensor:
        """
        x:     (B, 3, 64, 128)
        t:     (B,)
        op_id: (B,)  — can contain values in [0, num_ops] where num_ops = null class
        returns: v_pred (B, 3, 64, 128)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x to be 4D (B,C,H,W). Got shape {tuple(x.shape)}")
        B, C, H, W = x.shape
        if C != self.final_conv.out_channels:
            raise ValueError(f"Expected x channels={self.final_conv.out_channels}, got {C}")
        if (H, W) != (64, 128):
            raise ValueError(f"Expected spatial size (64,128), got {(H,W)}")

        # --- time embedding ---
        t_emb = timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        # --- op embedding (null class is index self.num_ops) ---
        op_vec = self.op_emb(op_id)

        # --- conditioning vector ---
        cond_vec = torch.cat([t_emb, op_vec], dim=1)

        # ---- Encoder ----
        d1 = self.down1(x, cond_vec)
        p1 = self.pool(d1)

        d2 = self.down2(p1, cond_vec)
        p2 = self.pool(d2)

        d3 = self.down3(p2, cond_vec)
        p3 = self.pool(d3)

        d4 = self.down4(p3, cond_vec)
        p4 = self.pool(d4)

        # ---- Bottleneck ----
        b = self.bottleneck(p4, cond_vec)

        # ---- Decoder ----
        u4 = self.up4(b)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.dec4(u4, cond_vec)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.dec3(u3, cond_vec)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.dec2(u2, cond_vec)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.dec1(u1, cond_vec)

        out = self.final_conv(u1)
        return out

    def number_of_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        print("Number of network parameters:", n)
        return n
