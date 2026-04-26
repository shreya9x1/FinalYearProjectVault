# ============================================================
# train&saveS3DIS.py — Train PointNet on S3DIS (Area_5)
# ============================================================
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model_S3DIS import PointNetSegmentation, pointnet_seg_loss
from data_reader_s3dis import S3DISDataset

# ============================================================
# Dataset Loader for S3DIS (Area_5)
# ============================================================
class S3DISDataset(Dataset):
    def __init__(self, root_dir, num_points=4096):
        self.files = []
        self.num_points = num_points
        total_files = 0
        valid_files = 0
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".txt"):
                    total_files += 1
                    fp = os.path.join(root, f)
                    # Quick validation: try reading a few rows and ensure there are >=7 columns
                    try:
                        sample = np.loadtxt(fp, max_rows=5)
                        if sample.ndim == 1:
                            cols = sample.shape[0]
                        else:
                            cols = sample.shape[1]
                        if cols >= 7:
                            self.files.append(fp)
                            valid_files += 1
                    except Exception:
                        # skip malformed files
                        continue
        print(f"Found {valid_files} valid files out of {total_files} total files in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        try:
            data = np.loadtxt(fp)
        except Exception as e:
            raise RuntimeError(f"Failed loading {fp}: {e}")

        if data.ndim == 1:
            # single-row file
            if data.shape[0] < 7:
                raise RuntimeError(f"File {fp} does not contain >=7 columns")
            data = data.reshape(1, -1)

        # Keep first 6 columns as features (XYZ + RGB) and last column as label
        xyz = data[:, 0:6]
        labels = data[:, -1].astype(int)

        # If labels appear 1-based, shift to 0-based. We detect typical S3DIS (1..13)
        if labels.max() > 12:
            labels = labels - 1

        # Random sampling to fixed num_points
        if len(xyz) >= self.num_points:
            choice = np.random.choice(len(xyz), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(xyz), self.num_points, replace=True)
        xyz = xyz[choice, :]
        labels = labels[choice]

        xyz = torch.from_numpy(xyz.T).float()  # [C, N]
        labels = torch.from_numpy(labels).long()
        return xyz, labels

# ============================================================
# Metric Utilities
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
        outputs, m3x3, m64x64 = model(points)
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
            outputs, m3x3, m64x64 = model(points)
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
    NUM_CLASSES = 13   # S3DIS has 13 semantic classes
    NUM_POINTS = 4096
    BATCH_SIZE = 4
    EPOCHS = 100
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and loader
    dataset = S3DISDataset(DATA_PATH, num_points=NUM_POINTS)

    # Sanity check: label range
    print("🔍 Checking label range in dataset...")
    all_labels = []
    for i in range(min(10, len(dataset))):  # sample 10 files for speed
        _, lbls = dataset[i]
        all_labels.extend(lbls.numpy().tolist())
    unique_labels = sorted(set(all_labels))
    print(f"Unique label IDs in sample: {unique_labels}")
    if min(unique_labels) < 0 or max(unique_labels) >= NUM_CLASSES:
        raise ValueError(
            f"❌ Label range mismatch! Found labels {unique_labels}, expected 0–{NUM_CLASSES-1}"
        )

    # drop_last=True avoids small final batches (BatchNorm requires >1 in training)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    # Model setup
    model = PointNetSegmentation().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Training loop
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
    print("\n📊 Final Benchmark Results on Area_5")
    print("=======================================")
    print(f"Overall Accuracy (OA): {oa:.4f}")
    print(f"Mean Class Accuracy (mAcc): {macc:.4f}")
    print(f"Mean IoU (mIoU): {miou:.4f}")

    # Save model
    torch.save(model.state_dict(), "pointnet_s3dis_area5.pth")
    print("✅ Model saved as pointnet_s3dis_area5.pth")
