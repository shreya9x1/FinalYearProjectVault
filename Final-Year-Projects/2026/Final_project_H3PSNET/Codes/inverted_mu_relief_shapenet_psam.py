import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# --- IMPORTS ---
# Ensure you have your ShapeNet dataloader here.
# If your file is named differently (e.g., data_reader.py), change this line.
from shapenet_loader2 import ShapeNetPartDataset
# Import the model
from pointsam import PointSAM_Segmentation

# --- CONFIGURATION ---
BATCH_SIZE = 8           # ShapeNet objects are smaller, we can increase batch size
NUM_EPOCHS = 50          # Short, aggressive fine-tuning run
LEARNING_RATE = 0.0001   # Low LR for fine-tuning
NUM_POINTS = 2048        # Standard ShapeNet resolution
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATHS ---
DATA_ROOT = r"C:\Users\CSE_SDPL\Desktop\ModelNet40-PointNet\ShapenetPart\PartAnnotation"

PRETRAINED_PATH = r"C:\Users\CSE_SDPL\Desktop\ModelNet40-PointNet\pointsam_shapenet_final.pth"

# --- AGGRESSIVE RELIEF WEIGHTS (ShapeNet) ---
# ShapeNet has 50 classes. Small parts (legs, handles, engines) need higher weights.
# We create a generic weight tensor where:
# - Common parts (indices 0-4 usually) get weight 1.0
# - Rare parts get weight 3.0 to 5.0
# You can tune this if you know specific classes are failing.
GENERIC_PART_WEIGHTS = [2.0] * 50 

def calculate_metrics(pred_np, target_np, num_classes):
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    if len(pred_flat) == 0: return 0, 0, 0
    
    # Filter out valid classes only (ShapeNet Part often has irregular class indices)
    # We map predictions to the actual classes present in the batch/dataset
    
    cm = confusion_matrix(target_flat, pred_flat, labels=list(range(num_classes)))
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = intersection / union
    iou = np.nan_to_num(iou)
    miou = np.mean(iou)
    
    oa = accuracy_score(target_flat, pred_flat)
    return oa, 0.0, miou # Returning 0 for mAcc to keep signature simple

def inverted_relief_loss(pred, target, num_classes):
    """
    Custom Loss:
    1. Standard NLL Loss
    2. "Confidence Penalty": Focus heavily on points the model is unsure about.
    """
    # pred is LogSoftmax [B, C, N]
    # target is [B, N]
    
    # 1. Standard Loss (No static weights for now, relying on dynamic mining)
    nll_loss = F.nll_loss(pred, target, reduction='none')
    
    # 2. Confidence Mining (The "Inverted" Part)
    with torch.no_grad():
        probs = torch.exp(pred) # [B, C, N]
        
        # Gather probability of the TRUE class
        target_unsqueeze = target.unsqueeze(1)
        correct_class_prob = torch.gather(probs, 1, target_unsqueeze).squeeze(1) # [B, N]
        
        # Inverted Weight Logic:
        # If Prob is 0.9 (Easy), Weight = ~0.01 (Ignore)
        # If Prob is 0.2 (Hard), Weight = ~0.64 (Focus)
        # Multiplier 5.0 ensures hard points dominate the gradient.
        mining_factor = (1.0 - correct_class_prob).pow(2) * 5.0
        
        # Clamp to keep training stable
        mining_factor = torch.clamp(mining_factor, min=1.0, max=8.0)

    # Final Loss
    weighted_loss = nll_loss * mining_factor
    return weighted_loss.mean()

def main():
    print(f"--- STARTING SHAPENET RELIEF FINETUNING ---")
    print(f"Loading weights from: {PRETRAINED_PATH}")
    
    # 1. Dataset
    # Assuming 'train' and 'test' splits are handled by the class constructor or arguments
    train_dataset = ShapeNetPartDataset(root_dir=DATA_ROOT, num_points=NUM_POINTS, split='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    
    test_dataset = ShapeNetPartDataset(root_dir=DATA_ROOT, num_points=NUM_POINTS, split='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # ShapeNet Part has 50 total part classes across 16 object categories
    num_part_classes = 2
    
    # 2. Model
    model = PointSAM_Segmentation(num_classes=num_part_classes).to(DEVICE)
    
    if os.path.exists(PRETRAINED_PATH):
        model.load_state_dict(torch.load(PRETRAINED_PATH), strict=False)
        print("✓ Weights Loaded Successfully")
    else:
        print(f"!! ERROR: Could not find weights at {PRETRAINED_PATH}")
        return
    
    # 3. Optimizer (Low LR)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    best_miou = 0.0

    # 4. Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Relief Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch in pbar:
            # Handle different dataloader return formats
            if len(batch) == 3:
                points, label, target = batch
            else:
                points, target = batch
                label = torch.zeros(points.shape[0], 1).long() # Dummy label if missing
            
            points, label, target = points.to(DEVICE), label.to(DEVICE), target.to(DEVICE)
            
            # Create One-Hot Category Vector [B, 16] if your model needs it
            # ShapeNet normally requires the object category (Airplane vs Chair)
            # If your SAMNetPP expects 'dummy_cat', we create it from 'label'
            dummy_cat = torch.zeros(points.shape[0], 16).to(DEVICE)
            dummy_cat.scatter_(1, label.view(-1, 1), 1) # Populate based on object label
            
            optimizer.zero_grad()
            
            pred = model(points) # [B, N, C]
            pred = pred.transpose(1, 2)     # [B, C, N]
            
            # --- CUSTOM LOSS CALL ---
            loss = inverted_relief_loss(pred, target, num_part_classes)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'Relief Loss': loss.item()})
            
        # Eval
        print("Evaluating Relief Impact...")
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                if len(batch) == 3:
                    points, label, target = batch
                else:
                    points, target = batch
                    label = torch.zeros(points.shape[0], 1).long()

                points, label, target = points.to(DEVICE), label.to(DEVICE), target.to(DEVICE)
                
                dummy_cat = torch.zeros(points.shape[0], 16).to(DEVICE)
                dummy_cat.scatter_(1, label.view(-1, 1), 1)
                
                pred = model(points)
                pred_choice = pred.data.max(2)[1]
                
                all_preds.append(pred_choice.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        oa, _, miou = calculate_metrics(all_preds, all_targets, num_part_classes)
        
        print(f"Epoch {epoch+1} Relief Stats:")
        print(f"  OA  : {oa:.4f}")
        print(f"  mIoU: {miou:.4f}")
        
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), 'best_psam_shapenet_relief.pth')
            print(f"  ✓ Saved RELIEF model (mIoU: {best_miou:.4f})")

if __name__ == "__main__":
    main()