# pointfly.py  --- PyTorch implementation

import torch
import torch.nn as nn
import torch.nn.functional as F


def dense(x, C_out, name, is_training=True):
    N, P, K, C_in = x.shape if x.dim() == 4 else (x.shape[0], x.shape[1], 1, x.shape[-1])
    layer = nn.Linear(C_in, C_out).to(x.device)
    return layer(x)


def conv2d(x, C_out, name, is_training, kernel_size):
    conv = nn.Conv2d(
        in_channels=x.shape[-1],
        out_channels=C_out,
        kernel_size=kernel_size,
        bias=True
    ).to(x.device)

    # reorder to N,C,H,W
    x = x.permute(0, 3, 1, 2)
    y = conv(x)
    return y.permute(0, 2, 3, 1)


def depthwise_conv2d(x, K, name, is_training, kernel_size, activation=F.relu):
    C = x.shape[-1]
    conv = nn.Conv2d(
        in_channels=C,
        out_channels=C,
        kernel_size=kernel_size,
        groups=C,
        bias=True
    ).to(x.device)

    x = x.permute(0, 3, 1, 2)
    y = conv(x)
    y = y.permute(0, 2, 3, 1)
    return activation(y) if activation else y


def separable_conv2d(x, C_out, name, is_training, kernel_size, depth_multiplier=1):
    C_in = x.shape[-1]
    # depthwise
    depthwise = nn.Conv2d(
        in_channels=C_in,
        out_channels=C_in * depth_multiplier,
        kernel_size=kernel_size,
        groups=C_in
    ).to(x.device)

    # pointwise
    pointwise = nn.Conv2d(
        in_channels=C_in * depth_multiplier,
        out_channels=C_out,
        kernel_size=1
    ).to(x.device)

    x = x.permute(0, 3, 1, 2)
    y = depthwise(x)
    y = pointwise(y)
    return y.permute(0, 2, 3, 1)


def knn_indices_general(qrs, pts, K, return_dist=False):
    """
    qrs: (N,P,3)
    pts: (N,N_pts,3)
    """
    N, P, _ = qrs.shape
    N2 = pts.shape[1]

    diff = qrs.unsqueeze(2) - pts.unsqueeze(1)  # (N,P,N2,3)
    dist = torch.sum(diff ** 2, dim=-1)         # (N,P,N2)
    knn = torch.topk(dist, K, largest=False)[1] # indices
    return (dist, knn) if return_dist else (None, knn)


def sort_points(pts, indices, method):
    # You can later extend this. For now, return original.
    return indices


def inverse_density_sampling(pts, K, P):
    N, P0, _ = pts.shape
    return torch.randint(0, P0, (N, P, 1), device=pts.device)
