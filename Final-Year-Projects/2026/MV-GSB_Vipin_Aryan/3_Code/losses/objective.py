from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from losses.structural import multiscale_log_loss


def entropy_kl_term(pred_noise: torch.Tensor) -> torch.Tensor:
    # KL to N(0,1) under moment matching.
    mu = pred_noise.mean()
    var = pred_noise.var(unbiased=False)
    eps = 1e-8
    var = torch.clamp(var, min=eps, max=1e6)
    kl = 0.5 * (mu.pow(2) + var - torch.log(var + eps) - 1.0)
    return torch.nan_to_num(kl, nan=0.0, posinf=1e4, neginf=0.0)


def composite_loss(
    outputs: Dict[str, torch.Tensor | None],
    noise_weight: float,
    edge_weight: float,
    entropy_weight: float,
) -> Dict[str, torch.Tensor]:
    noise_loss = F.mse_loss(outputs["pred_noise"], outputs["eps"])
    edge_loss = multiscale_log_loss(outputs["pred_ir"], outputs["gt_ir"])

    if outputs["semantic_feat_ir"] is not None and outputs["semantic_feat_rgb"] is not None:
        semantic_loss = F.mse_loss(outputs["semantic_feat_ir"], outputs["semantic_feat_rgb"])
    else:
        semantic_loss = noise_loss.new_zeros(())

    if outputs["task_feat_pred"] is not None and outputs["task_feat_gt"] is not None:
        task_loss = F.mse_loss(outputs["task_feat_pred"], outputs["task_feat_gt"])
    else:
        task_loss = noise_loss.new_zeros(())
    entropy_loss = entropy_kl_term(outputs["pred_noise"])

    total = (
        noise_weight * noise_loss
        + edge_weight * edge_loss
        + outputs["semantic_scale"] * semantic_loss
        + outputs["task_scale"] * task_loss
        + entropy_weight * entropy_loss
    )

    return {
        "total": total,
        "noise": noise_loss.detach(),
        "edge": edge_loss.detach(),
        "semantic": semantic_loss.detach(),
        "task": task_loss.detach(),
        "entropy": entropy_loss.detach(),
    }
