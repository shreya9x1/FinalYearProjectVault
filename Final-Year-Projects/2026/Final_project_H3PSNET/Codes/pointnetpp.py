# pointnetpp.py
# PointNet++ (classification) - simplified implementation for ModelNet-style datasets

import torch
import torch.nn as nn
import torch.nn.functional as F

def farthest_point_sample(x, npoint):
    """
    x: (B, N, 3)
    returns: (B, npoint) indices
    """
    B, N, _ = x.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=x.device)
    distance = torch.ones(B, N, device=x.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=x.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = x[torch.arange(B, device=x.device), farthest].view(B, 1, 3)
        dist = torch.sum((x - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    points: (B, N, C)
    idx:    (B, S) or (B, S, K)
    returns: (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]
    if idx.dim() == 2:
        # (B, S)
        S = idx.shape[1]
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1).repeat(1, S)
        return points[batch_indices, idx, :]
    elif idx.dim() == 3:
        # (B, S, K)
        B, S, K = idx.shape
        idx_flat = idx.reshape(B, -1)
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1).repeat(1, S * K)
        selected = points[batch_indices, idx_flat, :].view(B, S, K, -1)
        return selected
    else:
        raise ValueError("Unsupported idx dimensions")

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    xyz: (B, N, 3)
    new_xyz: (B, S, 3)
    returns: group_idx (B, S, nsample)
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    # squared distances (B, N, S)
    sqrdists = torch.sum((xyz.unsqueeze(2) - new_xyz.unsqueeze(1)) ** 2, dim=3)
    # get nsample smallest distances' indices for each new point
    group_idx = torch.argsort(sqrdists, dim=1)[:, :S, :nsample]  # shape (B, S, nsample)
    # note: argsort yields shape (B,N,S) then slicing
    return group_idx

class PointNetSetAbstraction(nn.Module):
    """
    Single-level Set Abstraction (sampling + grouping + pointnet)
    Input:
        xyz : (B, C_in, N)  where C_in is either 3 (coords) or feature channels
    Returns:
        features: (B, C_out, npoint)
        new_xyz:  (B, 3, npoint)  (coordinates of centroids)  -- useful but optional
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        layers = []
        last_channel = in_channel
        # Conv2d expects input (B, C, S, K) -> we will permute grouped points to that shape
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz):
        """
        xyz: (B, C_in, N)
        """
        B, C, N = xyz.shape
        # convert to (B, N, C) for sampling code
        xyz_trans = xyz.transpose(1, 2).contiguous()  # (B, N, C)

        # sampling
        fps_idx = farthest_point_sample(xyz_trans, self.npoint)  # (B, npoint)
        new_xyz = index_points(xyz_trans, fps_idx)  # (B, npoint, 3)

        # grouping
        group_idx = query_ball_point(self.radius, self.nsample, xyz_trans, new_xyz)  # (B, npoint, nsample)
        grouped_xyz = index_points(xyz_trans, group_idx)  # (B, npoint, nsample, 3)

        # subtract centroid to get local coordinates
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)  # (B, npoint, nsample, 3)

        # if input features exist beyond xyz channels, we expect they were provided as channels in xyz.
        # grouped_xyz currently shape (B, npoint, nsample, 3). To feed into MLP that expects 'in_channel'
        # channels, we permute to (B, in_channel, npoint, nsample). For first SA block, in_channel will be 3.
        grouped_xyz = grouped_xyz.permute(0, 3, 1, 2).contiguous()  # (B, 3, npoint, nsample)
        features = self.mlp(grouped_xyz)  # (B, out, npoint, nsample)
        # max pool over nsample dimension
        features = torch.max(features, -1)[0]  # (B, out, npoint)
        # return features and new_xyz as (B, 3, npoint)
        return features, new_xyz.transpose(1, 2).contiguous()

class PointNetPP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # SA layers: in_channel is number of input channels (3 for coords; later stages use previous MLP output)
        # For this simplified implementation we use coordinates-only SA blocks
        # (each SA will operate on XYZ only). This keeps shapes consistent for farthest point sampling.
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=3, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=1, radius=999, nsample=128, in_channel=3, mlp=[256, 512, 1024])

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        x: (B, 3, N)
        """
        # Run SA blocks using coordinates only (x is (B,3,N))
        l1, _ = self.sa1(x)   # l1: (B, 128, 512)
        l2, _ = self.sa2(x)   # l2: (B, 256, 128)  (uses coords again in this simplified variant)
        l3, _ = self.sa3(x)   # l3: (B, 1024, 1)

        x = l3.squeeze(-1)  # (B, 1024)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def pointnetpp_loss(pred, target):
    return nn.NLLLoss()(pred, target)
