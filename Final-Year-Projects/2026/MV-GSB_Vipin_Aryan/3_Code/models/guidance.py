from __future__ import annotations

import torch
from torch import nn


class FrozenDINOv2Oracle(nn.Module):
    """Interface-compatible semantic oracle stub.

    Replace with real DINOv2-ViT-L/14 features for production.
    """

    def __init__(
        self,
        out_dim: int = 256,
        use_real_backbone: bool = False,
        model_name: str = "dinov2_vitl14",
    ):
        super().__init__()
        self._using_real = False

        if use_real_backbone:
            try:
                # Requires internet/model cache on first load.
                self.encoder = torch.hub.load("facebookresearch/dinov2", model_name)
                self._using_real = True
            except Exception:
                self.encoder = self._build_stub(out_dim=out_dim)
        else:
            self.encoder = self._build_stub(out_dim=out_dim)

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    @staticmethod
    def _build_stub(out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, out_dim),
        )

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if self._using_real:
            # DINOv2 returns global embeddings as [B, D].
            return self.encoder(rgb)
        return self.encoder(rgb)


class FrozenRFDETRTaskOracle(nn.Module):
    """Interface-compatible task oracle stub.

    Replace with real RF-DETR backbone feature taps for production.
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, out_dim),
        )
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, ir: torch.Tensor) -> torch.Tensor:
        return self.encoder(ir)
