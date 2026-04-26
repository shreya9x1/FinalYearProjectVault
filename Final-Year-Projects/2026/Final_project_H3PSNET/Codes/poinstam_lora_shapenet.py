import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment #Hungarian Matching

class LoRALinear(nn.Module):
    # Change _init_ to __init__ (double underscores)
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        # Change super()._init_() to super().__init__()
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False 
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Result = xW + x(AB) * scaling
        return self.linear(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
    
class LoRAConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, rank=4, alpha=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv.weight.requires_grad = False #freeze to original weights
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_channels))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank))
        self.scaling = alpha / rank
        
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Result = xW + x(AB) * scaling
        return self.conv(x) + (x.transpose(1, 2) @ self.lora_A.t() @ self.lora_B.t()).transpose(1, 2) * self.scaling
    
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^2 + dst^2 - 2*src*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

class PointSAM(nn.Module):
    def __init__(self, embed_dim=128, num_classes=1, lora_rank=4): # Default to 1 for binary masks
        super().__init__()
        self.encoder = nn.Sequential(
            LoRALinear(3, 64, rank=lora_rank),
            nn.ReLU(),
            LoRALinear(64, embed_dim, rank=lora_rank)
        )
        
        self.prompt_encoder = nn.Linear(3, embed_dim)
        self.mask_decoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self.register_buffer("target_prototypes", None)

    def forward(self, points, pos_coords):
        # 1. Capture dynamic shapes from inputs
        B, N, _ = points.shape
        _, I, _ = pos_coords.shape
        
        point_feats = self.encoder(points) # [B, N, D]
        
        # Initial Pass
        # point_feats: [B, 1, N, D], pos_embed: [B, I, 1, D]
        pos_embed = self.prompt_encoder(pos_coords).unsqueeze(2)
        initial_logits = self.mask_decoder(point_feats.unsqueeze(1) + pos_embed)
        initial_logits = initial_logits.squeeze(-1) # Result: [B, I, N]
        
        # NPC Refinement
        neg_coords = self.get_npc_refined_mask(initial_logits, pos_coords)
        neg_embed = self.prompt_encoder(neg_coords).unsqueeze(2)
        
        # Final Refined Mask 
        refined_logits = self.mask_decoder(point_feats.unsqueeze(1) + pos_embed - neg_embed)
        refined_logits = refined_logits.squeeze(-1) # Result: [B, I, N]
        
        return refined_logits, point_feats

    def get_npc_refined_mask(self, initial_logits, pos_coords, tau_iou=0.1):
        B, I, N = initial_logits.shape
        initial_masks = (torch.sigmoid(initial_logits) > 0.5).float()
        
        intersection = torch.bmm(initial_masks, initial_masks.transpose(1, 2))
        m_sum = initial_masks.sum(dim=2).unsqueeze(2)
        union = m_sum + m_sum.transpose(1, 2) - intersection
        iou_matrix = intersection / (union + 1e-6)
        
        for b in range(B): 
            iou_matrix[b].fill_diagonal_(0) 
        
        neg_coords = torch.zeros_like(pos_coords)
        for b in range(B):
            for i in range(I):
                candidates = (iou_matrix[b, i] >= tau_iou).nonzero(as_tuple=True)[0]
                if len(candidates) > 0:
                    # Use .item() to ensure we get a standard index
                    idx = candidates[torch.randint(0, len(candidates), (1,))].item()
                    neg_coords[b, i] = pos_coords[b, idx]
        return neg_coords

    def set_target_prototypes(self, prototypes):
        """Helper to update the buffer from the training loop"""
        if prototypes is not None:
            self.target_prototypes = prototypes

    def compute_pbr_loss(self, current_feats):
        if self.target_prototypes is None:
            return torch.tensor(0.0, device=current_feats.device)
            
        # pred_prototypes shape: [B, D]
        pred_prototypes = current_feats.mean(dim=1) 
        
        # dist_matrix shape: [B, NumPrototypes]
        dist_matrix = 1 - F.cosine_similarity(
            pred_prototypes.unsqueeze(1), 
            self.target_prototypes.unsqueeze(0), dim=-1
        )
        
        row_ind, col_ind = linear_sum_assignment(dist_matrix.detach().cpu().numpy())
        loss_match = dist_matrix[torch.tensor(row_ind), torch.tensor(col_ind)].mean() 

        return loss_match

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
    

def run_offline_generation(dataloader, model):
    all_feats = []
    model.eval()
    device = next(model.parameters()).device
    
    # TARGET: We only need about 15,000 - 20,000 points TOTAL to find prototypes
    max_total_points = 20000 
    points_per_object = 20 # Small number of points per shape
    
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
    
    # Ensure we don't exceed memory limits (20k points is ~1.5GB RAM for clustering)
    if len(all_feats) > max_total_points:
        indices = np.random.choice(len(all_feats), max_total_points, replace=False)
        all_feats = all_feats[indices]

    print(f"Clustering {len(all_feats)} features... (This may take a few minutes)")
    # Using distance_threshold to find the 'natural' number of part clusters
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.8) 
    cluster_labels = clustering.fit_predict(all_feats)

    prototypes = []
    unique_labels = np.unique(cluster_labels)
    for u in unique_labels:
        prototypes.append(all_feats[cluster_labels == u].mean(axis=0)) 

    print(f"Generated {len(prototypes)} target prototypes.")
    return torch.tensor(np.array(prototypes)).float().to(device)