import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from shapenet_loader import ShapeNetPartDataset
from pointsam import PointSAM_Segmentation

# ============================================================
# Metrics (Updated for 50 Classes)
# ============================================================
def calculate_metrics(preds, labels, num_classes):
    preds = preds.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    oa = np.mean(preds == labels)
    
    iou_per_class = []
    for cls in range(num_classes):
        p = (preds == cls)
        l = (labels == cls)
        if np.sum(l) > 0: # Only count classes present in the data
            intersection = np.sum(p & l)
            union = np.sum(p | l)
            iou_per_class.append(intersection / union if union > 0 else 1.0)
    
    miou = np.mean(iou_per_class) if iou_per_class else 0
    return oa, miou, iou_per_class

# ============================================================
# Training / Evaluation Loops (Fixed Unpacking)
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    # FIXED: Unpacking 2 values instead of 3
    for pts, labels in tqdm(loader, desc="Training"):
        pts, labels = pts.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(pts) # No obj_label passed here
        logits = logits.reshape(-1, logits.shape[-1])
        loss = criterion(logits, labels.reshape(-1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device, num_classes):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for pts, labels in tqdm(loader, desc="Evaluating"):
            pts, labels = pts.to(device), labels.to(device)
            logits = model(pts)
            preds = torch.argmax(logits, dim=-1)
            preds_all.append(preds)
            labels_all.append(labels)

    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)
    return calculate_metrics(preds_all, labels_all, num_classes)

# ============================================================
# Main Script
# ============================================================
if __name__ == "__main__":
    DATA_PATH = r"C:\Users\CSE_SDPL\Desktop\ModelNet40-PointNet\ShapenetPart\PartAnnotation"
    NUM_CLASSES = 50 # ShapeNet Part global total
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ShapeNetPartDataset(DATA_PATH, num_points=2048, split='train')
    test_ds  = ShapeNetPartDataset(DATA_PATH, num_points=2048, split='test')
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = PointSAM_Segmentation(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting Training: 50 Part Classes Detected")

    for epoch in range(1, 11):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        oa, miou, iou_list = evaluate(model, test_loader, DEVICE, NUM_CLASSES)

        print(f"\n--- Epoch {epoch} ---")
        print(f"Loss: {loss:.4f} | OA: {oa:.4f} | mIoU: {miou:.4f}")
        
        # Optional: Print individual IoU if needed
        # for i, val in enumerate(iou_list): print(f"Part {i}: {val:.4f}")

        torch.save(model.state_dict(), "pointsam_shapenet_best.pth")