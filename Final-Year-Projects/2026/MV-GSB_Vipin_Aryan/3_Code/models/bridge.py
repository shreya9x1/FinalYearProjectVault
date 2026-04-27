from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from models.enhancement import RAWMambaEnhancer
from models.guidance import FrozenDINOv2Oracle, FrozenRFDETRTaskOracle
from models.mv_generator import HybridMambaVisionBackbone
from utils.schedules import linear_alpha, parabolic_sigma


class MVGSBModel(nn.Module):
    def __init__(
        self,
        sigma_max: float = 0.25,
        use_real_dino: bool = False,
        dino_model_name: str = "dinov2_vitl14",
        time_eps: float = 1e-4,
    ):
        super().__init__()
        self.enhancer = RAWMambaEnhancer(channels=3)
        self.backbone = HybridMambaVisionBackbone(in_ch=4)
        self.semantic_oracle = FrozenDINOv2Oracle(
            use_real_backbone=use_real_dino,
            model_name=dino_model_name,
        )
        self.task_oracle = FrozenRFDETRTaskOracle()
        self.sigma_max = sigma_max
        self.time_eps = time_eps

    @staticmethod
    def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
        w = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x * w).sum(dim=1, keepdim=True)

    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        t: torch.Tensor,
        semantic_scale: float,
        task_scale: float,
        dino_feat_rgb: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | None]:
        t = torch.clamp(t, min=self.time_eps, max=1.0 - self.time_eps)

        enhanced_rgb = self.enhancer(rgb)
        rgb_gray = self._rgb_to_gray(enhanced_rgb)

        alpha = linear_alpha(t).view(-1, 1, 1, 1)
        sigma = parabolic_sigma(t, self.sigma_max).view(-1, 1, 1, 1)

        eps = torch.randn_like(ir)
        z_t = alpha * ir + (1.0 - alpha) * rgb_gray + sigma * eps

        # Promote scalar time embeddings to full-resolution maps for channel concat.
        alpha_map = alpha.expand(-1, -1, z_t.size(2), z_t.size(3))
        sigma_map = sigma.expand(-1, -1, z_t.size(2), z_t.size(3))
        bridge_in = torch.cat([z_t, rgb_gray, alpha_map, sigma_map], dim=1)
        pred_noise = self.backbone(bridge_in)

        # Rearranged interpolation to estimate target-domain sample.
        denom = torch.clamp(alpha, min=1e-4)
        pred_ir = (z_t - (1.0 - alpha) * rgb_gray - sigma * pred_noise) / denom
        pred_ir = torch.clamp(pred_ir, 0.0, 1.0)

        semantic_feat_rgb: torch.Tensor | None = None
        semantic_feat_ir: torch.Tensor | None = None
        task_feat_gt: torch.Tensor | None = None
        task_feat_pred: torch.Tensor | None = None

        # In stage-1 both scales are zero, so skipping these oracle paths saves time.
        if semantic_scale > 0.0:
            if dino_feat_rgb is None:
                semantic_feat_rgb = self.semantic_oracle(enhanced_rgb)
            else:
                semantic_feat_rgb = dino_feat_rgb
            semantic_feat_ir = self.semantic_oracle(pred_ir.repeat(1, 3, 1, 1))

        if task_scale > 0.0:
            task_feat_gt = self.task_oracle(ir)
            task_feat_pred = self.task_oracle(pred_ir)

        return {
            "eps": eps,
            "pred_noise": pred_noise,
            "enhanced_rgb": enhanced_rgb,
            "z_t": z_t,
            "gt_ir": ir,
            "alpha": alpha,
            "sigma": sigma,
            "pred_ir": pred_ir,
            "semantic_feat_rgb": semantic_feat_rgb,
            "semantic_feat_ir": semantic_feat_ir,
            "task_feat_gt": task_feat_gt,
            "task_feat_pred": task_feat_pred,
            "semantic_scale": torch.tensor(float(semantic_scale), device=rgb.device),
            "task_scale": torch.tensor(float(task_scale), device=rgb.device),
        }
