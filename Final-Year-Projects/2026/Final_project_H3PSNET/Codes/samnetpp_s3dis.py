import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Utility Functions (Keep exactly as they were) ---
def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3) # Fixed unsqueeze for older torch versions
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

# --- Layers (Unchanged) ---
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 1, xyz.shape[2]).to(xyz.device)
            grouped_idx = torch.arange(xyz.shape[1], dtype=torch.long).to(xyz.device).view(1, 1, -1).repeat(xyz.shape[0], 1, 1)
        else:
            new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))
            grouped_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)

        grouped_xyz = index_points(xyz, grouped_idx)
        grouped_xyz -= new_xyz.view(new_xyz.shape[0], new_xyz.shape[1], 1, 3) # Fixed unsqueeze

        if points is not None:
            grouped_points = index_points(points, grouped_idx)
            # Concatenate Relative XYZ (3) + Features (C)
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz

        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm # [B, N, 3]
            
            # FIX: Use unsqueeze instead of view to safely add the last dimension
            weight = weight.unsqueeze(-1) # [B, N, 3, 1]
            
            interpolated_points = torch.sum(index_points(points2, idx) * weight, dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
# --- FIXED Model Class ---
class SAMNetPP(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(SAMNetPP, self).__init__()
        
        # S3DIS inputs are usually 6 channels (XYZ + RGB) or 9 (XYZ + RGB + UVW)
        # We will split inputs: First 3 are XYZ, rest are Points
        input_channels = 6 # Assuming XYZRGB
        num_feature_channels = input_channels - 3 # = 3 (RGB)

        self.normal_channel = normal_channel
        
        # SA1: 
        # Inputs: Relative XYZ (3) + Feature Channels (3) = 6
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3 + num_feature_channels, mlp=[64, 64, 128], group_all=False)
        
        # SA2: 
        # Inputs: Relative XYZ (3) + SA1 Output (128) = 131
        # *FIXED*: Changed in_channel from 134 to 131
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        
        # SA3:
        # Inputs: Relative XYZ (3) + SA2 Output (256) = 259
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        # Decoders
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        
        # FP1 Input Channels:
        # We concatenate: OneHot(16) + Original XYZ(3) + Original RGB(3) + FP2 Output(128)
        # Total = 16 + 3 + 3 + 128 = 150
        fp1_in_channels = 128 + 16 + 3 + num_feature_channels
        self.fp1 = PointNetFeaturePropagation(in_channel=fp1_in_channels, mlp=[128, 128, 128])
        
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        """
        xyz: [B, 6, N] -> Expected to be XYZ + RGB
        cls_label: [B, 16] -> One hot vector
        """
        B, C, N = xyz.shape
        
        # --- STRICT SPLIT ---
        # l0_xyz: Strictly Geometry (3 channels) for grouping
        l0_xyz = xyz[:, :3, :] 
        # l0_points: Strictly Features (RGB, 3 channels) for processing
        l0_points = xyz[:, 3:, :] 

        # Encoder
        # SA1: takes XYZ(3) and RGB(3)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        
        # SA2: takes SA1_XYZ(3) and SA1_Features(128)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        # SA3: takes SA2_XYZ(3) and SA2_Features(256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        # FP1: Concatenate Skip Connections
        cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
        
        # Concatenate: OneHot + Original XYZ + Original RGB
        cat_vec = torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1)
            
        l0_points = self.fp1(l0_xyz, l1_xyz, cat_vec, l1_points)
        
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x