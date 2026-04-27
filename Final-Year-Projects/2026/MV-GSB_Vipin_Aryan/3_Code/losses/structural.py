from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_kernel(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    return kernel_2d


def _log_kernel(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    norm = (xx ** 2 + yy ** 2) / (2 * sigma ** 2)
    kernel = (norm - 1) * torch.exp(-norm)
    kernel = kernel - kernel.mean()
    return kernel


def multiscale_log_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    scales = [(5, 0.8), (7, 1.2), (9, 1.6)]

    for size, sigma in scales:
        logk = _log_kernel(size, sigma, pred.device, pred.dtype).view(1, 1, size, size)
        pred_edge = F.conv2d(pred, logk, padding=size // 2)
        target_edge = F.conv2d(target, logk, padding=size // 2)
        loss = loss + F.l1_loss(pred_edge, target_edge)

    return loss / len(scales)
