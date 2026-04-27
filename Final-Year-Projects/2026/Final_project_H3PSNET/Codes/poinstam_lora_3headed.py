import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False 
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


# --- PointNet++ Helper Functions ---

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
    return points[batch_indices, idx, :]

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
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

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        # FPS/Sampling
        new_xyz = xyz[:, :self.npoint, :] 
        dists = square_distance(new_xyz, xyz)
        if self.nsample is None:
            # Group-all (used for the final SA layer)
            idx = torch.arange(N, device=xyz.device, dtype=torch.long).view(1, 1, N).repeat(B, self.npoint, 1)
        else:
            idx = dists.argsort(dim=-1)[:, :, :self.nsample]
        
        grouped_xyz = index_points(xyz, idx) 
        grouped_xyz -= new_xyz.unsqueeze(2)
        
        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz
            
        new_points = new_points.permute(0, 3, 1, 2)
        for i, conv in enumerate(self.mlp_convs):
            new_points = F.relu(self.mlp_bns[i](conv(new_points)))
            
        new_points = torch.max(new_points, -1)[0].transpose(1, 2)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2)

        new_points = torch.cat([points1, interpolated_points], dim=-1) if points1 is not None else interpolated_points
        new_points = new_points.transpose(1, 2)
        for i, conv in enumerate(self.mlp_convs):
            new_points = F.relu(self.mlp_bns[i](conv(new_points)))
        return new_points.transpose(1, 2)

# --- WRAPPER FOR BACKBONE ---

class PointNet2Backbone(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3+3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=1, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024])

        self.fp3 = PointNetFeaturePropagation(in_channel=1024+256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256+128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+3, mlp=[128, 128, embed_dim])

    def forward(self, xyz):
        # Handle cases where input might be [B, 3, N]
        if xyz.shape[1] == 3 and xyz.shape[2] != 3:
            xyz = xyz.transpose(1, 2).contiguous()
            
        l0_xyz = xyz
        l0_points = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # SA3 produces a single global feature with 1024 channels (required by fp3 in_channel=1024+256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # Provide xyz as low-level features so fp1 in_channel=128+3 is satisfied.
        return self.fp1(l0_xyz, l1_xyz, l0_xyz, l1_points) # [B, N, embed_dim]

# --- MODIFIED PointSAM CLASS ---

class PointSAM_3Headed(nn.Module):
    def __init__(self, embed_dim=128, lora_rank=8):
        super().__init__()
        # FIX: Define the encoder as a sub-module so model.encoder(points) works
        self.encoder = PointNet2Backbone(embed_dim)
        
        self.prompt_encoder = nn.Linear(3, embed_dim)
        
        # HEAD 1: Mask Decoder
        self.mask_decoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # HEAD 2: Risk Head
        self.risk_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # HEAD 3: Prototype Buffer
        self.register_buffer("target_prototypes", None)

    def forward(self, points, pos_coords):
        # Call the encoder submodule
        point_feats = self.encoder(points) # [B, N, D]
        
        # HEAD 1: NPC Refinement logic
        pos_embed = self.prompt_encoder(pos_coords).unsqueeze(2) 
        initial_logits = self.mask_decoder(point_feats.unsqueeze(1) + pos_embed).squeeze(-1)
        
        neg_coords = self.get_npc_refined_mask(initial_logits, pos_coords)
        neg_embed = self.prompt_encoder(neg_coords).unsqueeze(2)
        
        refined_logits = self.mask_decoder(point_feats.unsqueeze(1) + pos_embed - neg_embed).squeeze(-1)

        # HEAD 2: Risk Prediction
        pred_risk = self.risk_head(point_feats).squeeze(-1) 
        
        return refined_logits, point_feats, pred_risk

    def get_npc_refined_mask(self, initial_logits, pos_coords, tau_iou=0.1):
        B, I, N = initial_logits.shape
        masks = (torch.sigmoid(initial_logits) > 0.5).float()
        intersection = torch.bmm(masks, masks.transpose(1, 2))
        m_sum = masks.sum(dim=2).unsqueeze(2)
        union = m_sum + m_sum.transpose(1, 2) - intersection
        iou_matrix = intersection / (union + 1e-6)
        for b in range(B): iou_matrix[b].fill_diagonal_(0)
        neg_coords = torch.zeros_like(pos_coords)
        for b in range(B):
            for i in range(I):
                candidates = (iou_matrix[b, i] >= tau_iou).nonzero(as_tuple=True)[0]
                if len(candidates) > 0:
                    idx = candidates[torch.randint(0, len(candidates), (1,))].item()
                    neg_coords[b, i] = pos_coords[b, idx]
        return neg_coords

    def compute_pbr_loss(self, current_feats):
        if self.target_prototypes is None: return torch.tensor(0.0, device=current_feats.device)
        pred_protos = current_feats.mean(dim=1)
        pred_protos = F.normalize(pred_protos, dim=-1, eps=1e-6)
        target_protos = F.normalize(self.target_prototypes, dim=-1, eps=1e-6)
        dist_mat = 1.0 - (pred_protos @ target_protos.t())
        dist_mat = torch.nan_to_num(dist_mat, nan=1.0, posinf=1.0, neginf=1.0)
        dist_np = dist_mat.detach().cpu().numpy()
        row, col = linear_sum_assignment(dist_np)
        return dist_mat[row, col].mean()   
class MLP(nn.Module):
    """Simple multi-layer perceptron with optional batch norm."""
    def __init__(self, layer_sizes, use_bn: bool = False):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
                if use_bn:
                    layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    

def run_offline_generation(dataloader, model, max_total_points: int = 20000, points_per_object: int = 20, distance_threshold: float = 0.8):
    all_feats = []
    model.eval()
    device = next(model.parameters()).device
    
    print(f"Collecting features for clustering (target total: {max_total_points})...")
    
    with torch.no_grad():
        for points, labels in dataloader:
            points = points.to(device)
            
            # Fix the shape mismatch [B, 3, N] -> [B, N, 3]
            if points.shape[1] == 3:
                points = points.transpose(1, 2)
                
            feats = model.encoder(points) # [B, N, D]
            
            for b in range(feats.size(0)):
                # Sample a very small subset of points from each object
                idx = torch.randperm(feats.size(1))[:points_per_object]
                all_feats.append(feats[b, idx].cpu().numpy())
                
                # Stop collecting if we reached our limit
                if len(all_feats) * points_per_object >= max_total_points:
                    break
            else: continue
            break
                
    all_feats = np.concatenate(all_feats, axis=0)
    all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure we don't exceed memory limits (20k points is ~1.5GB RAM for clustering)
    if len(all_feats) > max_total_points:
        indices = np.random.choice(len(all_feats), max_total_points, replace=False)
        all_feats = all_feats[indices]

    print(f"Clustering {len(all_feats)} features... (This may take a few minutes)")
    # Using distance_threshold to find the 'natural' number of part clusters
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold) 
    cluster_labels = clustering.fit_predict(all_feats)

    prototypes = []
    unique_labels = np.unique(cluster_labels)
    for u in unique_labels:
        prototypes.append(all_feats[cluster_labels == u].mean(axis=0)) 

    print(f"Generated {len(prototypes)} target prototypes.")
    protos = np.array(prototypes)
    protos = np.nan_to_num(protos, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tensor(protos).float().to(device)