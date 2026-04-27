import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# Import your dataloader (Make sure this is the version that returns GLOBAL labels 0-49)
from shapenet_loader import ShapeNetPartDataset
# Import the model
from samnetpp import SAMNetPP

# --- Configuration ---
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
NUM_POINTS = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = r"C:\Users\CSE_SDPL\Desktop\ModelNet40-PointNet\ShapenetPart\PartAnnotation"

# Standard ShapeNet Category to Part ID mapping
CAT_TO_PARTS = {
    'Airplane': [0, 1, 2, 3], 'Bag': [4, 5], 'Cap': [6, 7], 'Car': [8, 9, 10, 11],
    'Chair': [12, 13, 14, 15], 'Earphone': [16, 17, 18], 'Guitar': [19, 20, 21],
    'Knife': [22, 23], 'Lamp': [24, 25, 26, 27], 'Laptop': [28, 29],
    'Motorbike': [30, 31, 32, 33, 34, 35], 'Mug': [36, 37],
    'Pistol': [38, 39, 40], 'Rocket': [41, 42, 43], 'Skateboard': [44, 45, 46],
    'Table': [47, 48, 49]
}

def calculate_metrics(pred_np, target_np, num_classes=50):
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    
    # Overall Accuracy
    oa = accuracy_score(target_flat, pred_flat)
    
    # Confusion Matrix
    cm = confusion_matrix(target_flat, pred_flat, labels=list(range(num_classes)))
    
    # Per-Part Accuracy and IoU
    with np.errstate(divide='ignore', invalid='ignore'):
        part_acc = np.diag(cm) / cm.sum(axis=1)
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        part_iou = intersection / union
        
    part_acc = np.nan_to_num(part_acc)
    part_iou = np.nan_to_num(part_iou)

    # --- Grouping into 16 Categories ---
    cat_metrics = {}
    for cat_name, part_ids in CAT_TO_PARTS.items():
        # Only include category if it actually appears in the target data
        if any(np.sum(target_flat == pid) > 0 for pid in part_ids):
            ious = [part_iou[pid] for pid in part_ids]
            accs = [part_acc[pid] for pid in part_ids]
            cat_metrics[cat_name] = {
                'iou': np.mean(ious),
                'acc': np.mean(accs)
            }
            
    return oa, cat_metrics

def main():
    print(f"Using device: {DEVICE}")

    # 1. Prepare Datasets
    train_dataset = ShapeNetPartDataset(root_dir=DATA_ROOT, num_points=NUM_POINTS, split='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    
    test_dataset = ShapeNetPartDataset(root_dir=DATA_ROOT, num_points=NUM_POINTS, split='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    num_part_classes = 50 # Standard Global ShapeNet Part count
    print(f"Total Global Part Classes: {num_part_classes}")

    # 2. Model Setup
    model = SAMNetPP(num_classes=num_part_classes, normal_channel=False).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    best_miou = 0.0

    for epoch in range(NUM_EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for points, target in pbar:
            points, target = points.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            
            # Using dummy category for architecture; dataloader must provide global labels
            dummy_cat = torch.zeros(points.shape[0], 16).to(DEVICE) 
            
            pred = model(points, dummy_cat) 
            pred_flat = pred.contiguous().view(-1, num_part_classes)
            target_flat = target.view(-1)
            
            loss = F.nll_loss(pred_flat, target_flat)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        
        # --- Evaluation ---
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for points, target in tqdm(test_loader, desc="Evaluating"):
                points, target = points.to(DEVICE), target.to(DEVICE)
                dummy_cat = torch.zeros(points.shape[0], 16).to(DEVICE)
                
                pred = model(points, dummy_cat)
                pred_choice = pred.data.max(2)[1] 
                
                all_preds.append(pred_choice.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate Category Metrics
        oa, cat_metrics = calculate_metrics(all_preds, all_targets, num_part_classes)
        
        # --- Print Results Table ---
        print("\n" + "="*65)
        print(f" EPOCH {epoch+1} SUMMARY")
        print("-" * 65)
        print(f"{'Category':<20} | {'mAcc':<15} | {'mIoU':<15}")
        print("-" * 65)
        
        all_mious = []
        for cat_name, scores in cat_metrics.items():
            print(f"{cat_name:<20} | {scores['acc']:<15.4f} | {scores['iou']:<15.4f}")
            all_mious.append(scores['iou'])
        
        mean_miou = np.mean(all_mious) if all_mious else 0
        
        print("-" * 65)
        print(f"OVERALL -> OA: {oa:.4f} | Class mIoU: {mean_miou:.4f}")
        print("="*65 + "\n")
        
        if mean_miou > best_miou:
            best_miou = mean_miou
            torch.save(model.state_dict(), 'best_samnetpp_shapenet.pth')
            print(f" ✓ Saved best model with mIoU: {best_miou:.4f}")
            
    print("Training Complete.")

if __name__ == "__main__":
    main()