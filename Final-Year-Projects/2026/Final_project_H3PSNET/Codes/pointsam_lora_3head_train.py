import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import your custom modules
from poinstam_lora_3headed import PointSAM_3Headed, run_offline_generation
from shapenet_loader import ShapeNetPartDataset 
from relief_utils import compute_relief_risk_score, apply_inverted_mu_relief, IOUMetric

# --- Utility Functions ---
def _ensure_points_bn3(points):
    """Ensures input is [B, N, 3]"""
    if points.shape[1] == 3 and points.shape[2] != 3:
        return points.transpose(1, 2).contiguous()
    return points

def get_prompts_from_labels(points, labels):
    """Pick one positive prompt point per present part category."""
    points = _ensure_points_bn3(points)
    B, N, _ = points.shape
    device = points.device
    
    # Identify unique classes per object in batch
    batch_prompts = []
    batch_classes = []
    
    for b in range(B):
        unique_classes = torch.unique(labels[b])
        unique_classes = unique_classes[unique_classes != -100] # Ignore void
        
        p_coords = torch.zeros((16, 3), device=device)
        p_classes = torch.full((16,), -1, dtype=torch.long, device=device)
        
        for i, cls in enumerate(unique_classes[:16]):
            indices = (labels[b] == cls).nonzero(as_tuple=True)[0]
            selected_idx = indices[torch.randint(0, len(indices), (1,))]
            p_coords[i] = points[b, selected_idx]
            p_classes[i] = cls
            
        batch_prompts.append(p_coords)
        batch_classes.append(p_classes)
        
    return torch.stack(batch_prompts), torch.stack(batch_classes)


def _looks_like_shapenet_part_root(root_dir: str) -> bool:
    if not root_dir or not os.path.isdir(root_dir):
        return False

    # The expected structure is:
    # root_dir/<category_id>/points/*.pts
    # root_dir/<category_id>/points_label/<part_name>/*.seg
    for entry in os.listdir(root_dir):
        cat_path = os.path.join(root_dir, entry)
        if not os.path.isdir(cat_path):
            continue
        if os.path.isdir(os.path.join(cat_path, 'points')) and os.path.isdir(os.path.join(cat_path, 'points_label')):
            return True
    return False


def resolve_shapenet_part_root_dir(root_dir: str | None) -> str:
    if root_dir and _looks_like_shapenet_part_root(root_dir):
        return root_dir

    env_candidates = [
        os.environ.get('SHAPENET_PART_ROOT'),
        os.environ.get('SHAPENET_ROOT'),
        os.environ.get('SHAPENET_DATA'),
    ]
    for candidate in env_candidates:
        if candidate and _looks_like_shapenet_part_root(candidate):
            return candidate

    script_dir = os.path.dirname(os.path.abspath(__file__))
    rel_candidates = [
        os.path.join(script_dir, 'ShapenetPart', 'PartAnnotation'),
        os.path.join(script_dir, 'ShapeNetPart', 'PartAnnotation'),
        os.path.join(script_dir, 'data', 'ShapenetPart', 'PartAnnotation'),
        os.path.join(script_dir, 'data', 'ShapeNetPart', 'PartAnnotation'),
    ]
    for candidate in rel_candidates:
        if _looks_like_shapenet_part_root(candidate):
            return candidate

    # Fall back to the provided root_dir even if it doesn't validate so the
    # downstream error message can show the exact user input.
    return root_dir or rel_candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(description='Train PointSAM LoRA (3-headed) on ShapeNetPart')
    parser.add_argument(
        '--root_dir',
        '--data_root',
        dest='root_dir',
        default=None,
        help=(
            'Path to ShapeNetPart/PartAnnotation (folder that contains category IDs like 02691156). '
            'If omitted, the script tries ./ShapenetPart/PartAnnotation relative to this file.'
        ),
    )
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--proto_points', type=int, default=20000, help='Total points used for offline prototype clustering')
    parser.add_argument('--proto_points_per_object', type=int, default=20, help='Points sampled per object for offline clustering')
    parser.add_argument('--proto_distance_threshold', type=float, default=0.8, help='Agglomerative clustering distance threshold')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    return parser.parse_args()

# --- Training Logic ---
def train(
    root_dir: str | None = None,
    num_points: int = 2048,
    batch_size: int = 8,
    total_epochs: int = 1000,
    proto_points: int = 20000,
    proto_points_per_object: int = 20,
    proto_distance_threshold: float = 0.8,
    save_dir: str = 'checkpoints',
):
    # 1. Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 50 
    NORMAL_EPOCHS = 20
    MINING_EPOCHS = 10
    CYCLE_LENGTH = NORMAL_EPOCHS + MINING_EPOCHS
    TOTAL_EPOCHS = total_epochs
    
    # 2. Data & Model
    resolved_root = resolve_shapenet_part_root_dir(root_dir)
    if not _looks_like_shapenet_part_root(resolved_root):
        raise FileNotFoundError(
            "Could not find a valid ShapeNetPart root_dir.\n"
            f"Tried: {resolved_root!r}\n\n"
            "Expected a folder that contains category ID subfolders (e.g. 02691156) each with 'points' and 'points_label'.\n"
            "If your data lives elsewhere, re-run with: --root_dir <path-to-.../PartAnnotation>"
        )

    train_ds = ShapeNetPartDataset(root_dir=resolved_root, split='train', num_points=num_points)
    test_ds = ShapeNetPartDataset(root_dir=resolved_root, split='test', num_points=num_points)
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(
            f"Loaded empty dataset(s) from root_dir={resolved_root!r}. "
            "Double-check the dataset structure under that folder."
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    model = PointSAM_3Headed(embed_dim=128, lora_rank=8).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # Initialize Head 3: Global Prototypes
    print("Generating Initial Prototypes...")
    target_protos = run_offline_generation(
        train_loader,
        model,
        max_total_points=proto_points,
        points_per_object=proto_points_per_object,
        distance_threshold=proto_distance_threshold,
    )
    model.target_prototypes = target_protos.to(DEVICE)
    
    best_miou = 0.0
    os.makedirs(save_dir, exist_ok=True)
    
    # 3. Training Loop
    for epoch in range(1, TOTAL_EPOCHS + 1):
        # Determine Phase
        cycle_pos = (epoch - 1) % CYCLE_LENGTH
        is_mining_phase = cycle_pos >= NORMAL_EPOCHS
        phase_name = "BLENDED MINING" if is_mining_phase else "BASE TRAINING"
        
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [{phase_name}]")
        
        for points, labels in pbar:
            points, labels = points.to(DEVICE), labels.to(DEVICE)
            points = _ensure_points_bn3(points)
            
            # --- STEP A: Risk Analysis ---
            with torch.no_grad():
                # Compute Ground Truth difficulty scores
                gt_risk = compute_relief_risk_score(points, labels, k=5, alpha=1.2)
            
            # --- STEP B: Blended Mining Filtering ---
            effective_labels = labels.clone()
            if is_mining_phase:
                # Set 'Easy' points to -100 (ignored by loss)
                effective_labels = apply_inverted_mu_relief(points, labels, mu=1.5)
            
            # --- STEP C: Forward Pass (3 Heads) ---
            pos_coords, prompt_classes = get_prompts_from_labels(points, labels)
            refined_logits, feats, pred_risk = model(points, pos_coords)
            
            # Head 1: Mask Segmentation Loss
            # Generate binary targets for each prompt [B, I, N]
            target_masks = (labels.unsqueeze(1) == prompt_classes.unsqueeze(-1)).float()
            
            # Calculate loss weighted by risk (harder points = higher gradient)
            raw_bce = F.binary_cross_entropy_with_logits(refined_logits, target_masks, reduction='none')
            weighted_bce = raw_bce * (1.0 + gt_risk.unsqueeze(1))
            
            # Apply the blended mining mask (ignore index)
            valid_mask = (effective_labels != -100).float().unsqueeze(1)
            seg_loss = (weighted_bce * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            
            # Head 2: Risk Prediction Loss (Predicting Mu-Relief Score)
            risk_loss = F.mse_loss(pred_risk, gt_risk)
            
            # Head 3: Prototype Alignment (PBR)
            pbr_loss = model.compute_pbr_loss(feats)
            
            # Total Multi-Task Loss
            total_loss = seg_loss + (0.5 * risk_loss) + (0.1 * pbr_loss)
            
            # Optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            pbar.set_postfix(loss=total_loss.item())

        # 4. Evaluation
        miou = evaluate(model, test_loader, DEVICE, NUM_CLASSES)
        print(f"Epoch {epoch} Result -> mIoU: {miou:.4f}")
        
        if miou > best_miou:
            best_miou = miou
            ckpt_path = os.path.join(save_dir, "pointsam_3head_blended_best.pth")
            torch.save(
                {
                    'epoch': epoch,
                    'best_miou': best_miou,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                ckpt_path,
            )
            # Optional: also save a filename that includes the score for easy tracking
            scored_path = os.path.join(save_dir, f"pointsam_3head_blended_best_miou_{best_miou:.4f}.pth")
            torch.save(model.state_dict(), scored_path)

def evaluate(model, loader, device, num_classes):
    model.eval()
    metric = IOUMetric(num_classes=num_classes)
    
    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            points = _ensure_points_bn3(points)
            pos_coords, prompt_classes = get_prompts_from_labels(points, labels)
            
            logits, _, _ = model(points, pos_coords) # [B, I, N]
            
            # Map prompt-based predictions back to point labels
            # Choose prompt with highest confidence for each point
            valid_prompts = (prompt_classes != -1).unsqueeze(-1)
            logits = logits.masked_fill(~valid_prompts, -1e9)
            best_prompt_idx = torch.argmax(logits, dim=1) # [B, N]
            
            # Gather corresponding class IDs
            preds = torch.gather(prompt_classes, 1, best_prompt_idx)
            metric.update(preds, labels)
            
    miou, _ = metric.compute()
    return miou

if __name__ == "__main__":
    args = parse_args()
    train(
        root_dir=args.root_dir,
        num_points=args.num_points,
        batch_size=args.batch_size,
        total_epochs=args.epochs,
        proto_points=args.proto_points,
        proto_points_per_object=args.proto_points_per_object,
        proto_distance_threshold=args.proto_distance_threshold,
        save_dir=args.save_dir,
    )