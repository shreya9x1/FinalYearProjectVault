from __future__ import annotations

import torch
from torch import nn


class RAWMambaEnhancer(nn.Module):
    """Simple zero-reference enhancement front-end inspired by Retinex decomposition."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        illum = self.net(x)
        return torch.clamp(x * (1.0 + illum), 0.0, 1.0)
