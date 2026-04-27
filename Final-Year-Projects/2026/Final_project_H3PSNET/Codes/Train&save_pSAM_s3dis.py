import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_reader_s3dis import S3DISDataset

from pointsam_s3dis import PointSAM_Segmentation


# ============================================================
# Metrics
# ============================================================
def calculate_metrics(preds, labels, num_classes):
    preds = preds.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()

    oa = np.mean(preds == labels)

    mIoU_list, mAcc_list = [], []
    for cls in range(num_classes):
        p = preds == cls
        l = labels == cls
        intersection = np.sum(p & l)
        union = np.sum(p | l)
        correct = intersection
        total = np.sum(l)

        if union > 0:
            mIoU_list.append(intersection / union)
        if total > 0:
            mAcc_list.append(correct / total)

    mIoU = np.mean(mIoU_list)
    mAcc = np.mean(mAcc_list)
    return oa, mAcc, mIoU


# ============================================================
# Training
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for pts, labels in tqdm(loader, desc="Training"):
        pts, labels = pts.to(device), labels.to(device)
        optimizer.zero_grad()

        # Ensure input is [B, 3, N] and only XYZ is used
        if pts.shape[1] != 3 and pts.shape[2] == 3:
            pts = pts.permute(0, 2, 1)
        if pts.shape[1] > 3:
            pts = pts[:, :3, :]
        logits = model(pts)        # [B, N, C]
        logits = logits.reshape(-1, logits.shape[-1])
        loss = criterion(logits, labels.reshape(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# Evaluation
# ============================================================
def evaluate(model, loader, device, num_classes):
    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for pts, labels in tqdm(loader, desc="Evaluating"):
            pts, labels = pts.to(device), labels.to(device)

            # Ensure input is [B, 3, N] and only XYZ is used
            if pts.shape[1] != 3 and pts.shape[2] == 3:
                pts = pts.permute(0, 2, 1)
            if pts.shape[1] > 3:
                pts = pts[:, :3, :]
            logits = model(pts)  # [B, N, C]
            preds = torch.argmax(logits, dim=-1)

            preds_all.append(preds)
            labels_all.append(labels)

    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)

    oa, mAcc, mIoU = calculate_metrics(preds_all, labels_all, num_classes)
    return oa, mAcc, mIoU


# ============================================================
# Main Script
# ============================================================
if __name__ == "__main__":
    DATA_PATH = r"C:\Users\CSE_SDPL\Desktop\ModelNet40-PointNet\preprocessed_Area_5"
    NUM_POINTS = 2048
    BATCH_SIZE = 4
    EPOCHS = 200
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    train_ds = S3DISDataset(DATA_PATH, num_points=NUM_POINTS)
    test_ds  = S3DISDataset(DATA_PATH, num_points=NUM_POINTS)

    NUM_CLASSES = train_ds.num_classes

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Load model
    model = PointSAM_Segmentation(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"Training PointSAM on S3dis area5")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Train samples: {len(train_ds)} Test samples: {len(test_ds)}")

    best_mIoU = 0

    for epoch in range(1, EPOCHS+1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        oa, mAcc, mIoU = evaluate(model, test_loader, device, NUM_CLASSES)

        print(f"Epoch {epoch}/{EPOCHS} | Loss {loss:.4f} | OA {oa:.4f} | mAcc {mAcc:.4f} | mIoU {mIoU:.4f}")

        if mIoU > best_mIoU:
            best_mIoU = mIoU
            torch.save(model.state_dict(), "pointsam_shapenet_best.pth")
            print("✓ Saved best model")

    torch.save(model.state_dict(), "pointsam_shapenet_final.pth")
    print("Training complete. Final model saved.")
