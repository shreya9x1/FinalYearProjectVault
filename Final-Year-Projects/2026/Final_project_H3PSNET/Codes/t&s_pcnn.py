import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pointcnn import PointCNN
from data_reader_s3dis import S3DISDataset
# ============================================================
# pointcnn.py — PointCNN Implementation
# ============================================================
def calculate_metrics(preds, labels, num_classes):
    """Compute OA, mAcc, and mIoU."""
    preds = preds.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    oa = np.mean(preds == labels)

    ious, accs = [], []
    for cls in range(num_classes):
        pred_mask = preds == cls
        label_mask = labels == cls
        intersection = np.sum(pred_mask & label_mask)
        union = np.sum(pred_mask | label_mask)
        correct = np.sum(pred_mask & label_mask)
        total = np.sum(label_mask)
        if union > 0:
            ious.append(intersection / union)
        if total > 0:
            accs.append(correct / total)
    miou = np.mean(ious) if ious else 0
    macc = np.mean(accs) if accs else 0
    return oa, macc, miou

# ============================================================
# Training and Evaluation Functions
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for points, labels in tqdm(loader, desc="Training"):
        points, labels = points.to(device), labels.to(device)
        optimizer.zero_grad()
        # PointCNN
        outputs, _, _ = model(points)
        outputs = outputs.transpose(1, 2).contiguous()  # [B, N, C]
        loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for points, labels in tqdm(loader, desc="Evaluating"):
            points, labels = points.to(device), labels.to(device)
            outputs, _, _ = model(points)  # (B, C, N) segmentation logits
            outputs = outputs.transpose(1, 2).contiguous()  # [B, N, C]
            preds = torch.argmax(outputs, dim=2)
            all_preds.append(preds)
            all_labels.append(labels)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    oa, macc, miou = calculate_metrics(all_preds, all_labels, num_classes)
    return oa, macc, miou



# ============================================================
# Main Training Script
# ============================================================
if __name__ == "__main__":
    DATA_PATH = r"C:\Users\CSE_SDPL\Desktop\ModelNet40-PointNet\preprocessed_Area_5"
    NUM_POINTS = 4096
    BATCH_SIZE = 4
    EPOCHS = 50
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and loader (using preprocessed S3DIS Area_5 with labeled files)
    dataset = S3DISDataset(DATA_PATH, num_points=NUM_POINTS)
    
    # Determine NUM_CLASSES from dataset
    all_unique_labels = set()
    for i in range(min(50, len(dataset))):  # sample 50 files
        _, lbls = dataset[i]
        all_unique_labels.update(lbls.numpy().tolist())
    NUM_CLASSES = len(all_unique_labels)
    print(f"Detected {NUM_CLASSES} unique classes in dataset")

    # Sanity check: label range
    print("🔍 Checking label range in dataset...")
    all_labels = []
    for i in range(min(10, len(dataset))):  # sample 10 files for speed
        _, lbls = dataset[i]
        # lbls may be a scalar tensor (classification label) or a tensor of per-point labels
        if torch.is_tensor(lbls):
            if lbls.dim() == 0:
                all_labels.append(int(lbls.item()))
            else:
                all_labels.extend(lbls.numpy().tolist())
        else:
            # fallback for non-tensor
            try:
                all_labels.extend(list(lbls))
            except Exception:
                all_labels.append(int(lbls))
    unique_labels = sorted(set(all_labels))
    print(f"Unique label IDs in sample: {unique_labels}")
    if min(unique_labels) < 0 or max(unique_labels) >= NUM_CLASSES:
        raise ValueError(
            f"❌ Label range mismatch! Found labels {unique_labels}, expected 0–{NUM_CLASSES-1}"
        )

    # drop_last=True avoids small final batches (BatchNorm requires >1 in training)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    # Model setup
    model = PointCNN(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Training loop (run all EPOCHS)
    for epoch in range(EPOCHS):
        try:
            loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss:.4f}")
        except RuntimeError as e:
            print(f"⚠️ Runtime error in epoch {epoch+1}: {e}")
            print("Tip: try setting CUDA_LAUNCH_BLOCKING=1 for detailed trace.")
            break

    # Evaluation
    oa, macc, miou = evaluate(model, dataloader, device, NUM_CLASSES)
    print("\n📊 Final Benchmark Results - PointCNN on S3DIS Area_5")
    print("=======================================")
    print(f"Overall Accuracy (OA): {oa:.4f}")
    print(f"Mean Class Accuracy (mAcc): {macc:.4f}")
    print(f"Mean IoU (mIoU): {miou:.4f}")

    # Save model
    torch.save(model.state_dict(), "pointcnn_s3dis_area5.pth")
    print("✅ Model saved as pointcnn_s3dis_area5.pth")
