from __future__ import annotations

import torch
from torch import nn


class TokenMixer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj_in = nn.Conv2d(dim, dim, kernel_size=1)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj_in(x)
        y = self.dw_conv(y)
        y = self.proj_out(y)
        return x + y


class HybridMambaVisionBackbone(nn.Module):
    """Lightweight hybrid block as a placeholder for official MambaVision-B."""

    def __init__(self, in_ch: int = 4, base_dim: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.local_block = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.global_block = TokenMixer(base_dim)
        self.head = nn.Conv2d(base_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = h + self.local_block(h)
        h = self.global_block(h)
        return self.head(h)
