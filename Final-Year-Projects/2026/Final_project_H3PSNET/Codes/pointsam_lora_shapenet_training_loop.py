import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from poinstam_lora_shapenet import PointSAM, run_offline_generation
from shapenet_loader import ShapeNetPartDataset 


def _ensure_points_bn3(points: torch.Tensor) -> torch.Tensor:
    """Ensure points are shaped [B, N, 3]. Dataset yields [B, 3, N]."""
    if points.dim() != 3:
        raise ValueError(f"Expected points to be 3D [B,*,*], got shape={tuple(points.shape)}")
    # Common case from ShapeNetPartDataset: [B, 3, N]
    if points.shape[1] == 3 and points.shape[2] != 3:
        return points.transpose(1, 2).contiguous()
    return points


def get_prompts_from_labels(points, labels, num_prompts_per_class=1):
    """
    For pointly-supervised learning, we pick one point per part 
    category present in the shape to act as the positive prompt.
    """
    points = _ensure_points_bn3(points)
    B, N, _ = points.shape
    device = points.device
    
    max_prompts = 16
    batch_prompts = torch.zeros(B, max_prompts, 3, device=device, dtype=points.dtype)
    # Stores which class each prompt corresponds to; -1 indicates padded/unused.
    prompt_classes = torch.full((B, max_prompts), -1, device=device, dtype=torch.long)
    
    for b in range(B):
        # Keep validation safe for integer labels (torch.isnan not valid on long).
        label_min = int(labels[b].min().item())
        label_max = int(labels[b].max().item())
        if label_min < 0 or label_max >= 50:
            unique = labels[b].detach().cpu().unique().numpy()
            raise ValueError(
                f"Invalid label detected in batch {b}: min={label_min}, max={label_max}, unique={unique}"
            )
        unique_classes = torch.unique(labels[b])
        selected_coords = []
        selected_classes = []
        for cls in unique_classes:
            # Find indices where this part class exists
            indices = (labels[b] == cls).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                # Randomly pick one point as the prompt for this part
                idx = indices[torch.randint(0, len(indices), (1,))]
                selected_coords.append(points[b, idx])  # [1, 3]
            selected_classes.append(cls.view(1))

        coords_tensor = (
            torch.cat(selected_coords, dim=0)
            if selected_coords
            else torch.zeros(0, 3, device=device, dtype=points.dtype)
        )  # [K, 3]

        num_found = min(coords_tensor.shape[0], max_prompts)
        if num_found > 0:
            batch_prompts[b, :num_found] = coords_tensor[:num_found]
            prompt_classes[b, :num_found] = torch.cat(selected_classes, dim=0)[:num_found]

    return batch_prompts, prompt_classes


def train():
  
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 8
    LR = 1e-4
    EPOCHS = 100
    DATA_PATH = r"C:\Users\CSE_SDPL\Desktop\ModelNet40-PointNet\ShapenetPart\PartAnnotation"
    
   
    train_ds = ShapeNetPartDataset(DATA_PATH, split='train', num_points=2048)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    test_ds = ShapeNetPartDataset(DATA_PATH, split='test', num_points=2048)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # PointSAM predicts a binary mask per prompt; prompts correspond to part classes.
    model = PointSAM(embed_dim=128, num_classes=1, lora_rank=8).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # We will compute BCE manually with masking over valid prompts.

    print("--- Generating Target Prototypes Offline ---")
    
    target_prototypes = run_offline_generation(train_loader, model)
    model.set_target_prototypes(target_prototypes.to(DEVICE))
    print(f"Prototypes initialized: {target_prototypes.shape}")

    # 5. Training Loop
    best_miou = 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for points, labels in pbar:
            points, labels = points.to(DEVICE), labels.to(DEVICE)

            # Dataset yields [B, 3, N]; model/prompt code expects [B, N, 3]
            points = _ensure_points_bn3(points)
            
            # Generate Positive Prompts from GT
            pos_coords, prompt_classes = get_prompts_from_labels(points, labels)
            
            optimizer.zero_grad()
            
            # Forward Pass
            # refined_logits: [B, I, N], point_feats: [B, N, D]
            refined_logits, point_feats = model(points, pos_coords)

            # Build per-prompt binary targets: target_masks[b, i, n] = (label[b, n] == prompt_class[b, i])
            valid_prompts = prompt_classes != -1  # [B, I]
            prompt_classes_exp = prompt_classes.unsqueeze(-1).expand(-1, -1, labels.shape[1])
            target_masks = (labels.unsqueeze(1) == prompt_classes_exp).float()  # [B, I, N]

            bce = F.binary_cross_entropy_with_logits(refined_logits, target_masks, reduction='none')
            bce = bce * valid_prompts.unsqueeze(-1).float()
            denom = valid_prompts.sum().clamp_min(1).float() * labels.shape[1]
            seg_loss = bce.sum() / denom

            pbr_loss = model.compute_pbr_loss(point_feats)
            
            # Total Loss (Section III-C)
            total_loss = seg_loss + 0.1 * pbr_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            pbar.set_postfix({"Loss": total_loss.item()})

       
        miou = evaluate(model, test_loader, DEVICE)
        print(f"Epoch {epoch+1} Complete. Mean Loss: {epoch_loss/len(train_loader):.4f} | mIoU: {miou:.4f}")
        
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), "pointsam_best_shapenet.pth")

def evaluate(model, loader, device):
    model.eval()
    ious = []
    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            points = _ensure_points_bn3(points)
            pos_coords, prompt_classes = get_prompts_from_labels(points, labels)
            
            logits, _ = model(points, pos_coords)

            # Decode: for each point choose the prompt with max logit, then map to its class id.
            valid_prompts = prompt_classes != -1  # [B, I]
            masked_logits = logits.clone()
            masked_logits = masked_logits.masked_fill(~valid_prompts.unsqueeze(-1), -1e9)
            best_prompt = masked_logits.argmax(dim=1)  # [B, N]

            prompt_classes_exp = prompt_classes.unsqueeze(-1).expand_as(masked_logits)  # [B, I, N]
            preds = prompt_classes_exp.gather(1, best_prompt.unsqueeze(1)).squeeze(1)  # [B, N]
            preds = torch.clamp(preds, min=0)
            
            # Simple IoU calculation
            for b in range(points.shape[0]):
                intersect = torch.logical_and(preds[b] == labels[b], labels[b] > 0).sum()
                union = torch.logical_or(preds[b] == labels[b], labels[b] > 0).sum()
                ious.append((intersect / (union + 1e-6)).item())
                
    return np.mean(ious)

if __name__ == "__main__":
    train()