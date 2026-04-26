import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from shapenet_loader import ShapeNetPartDataset


# ============================================================
# PointCNN Configuration and Wrapper
# ============================================================
class PointCNNSetting:
    """Configuration for PointCNN model."""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        # Reduced K (neighbors) and P (points) to handle smaller point clouds
        self.xconv_params = [
            {'K': 8, 'D': 1, 'P': -1, 'C': 32, 'links': []},
            {'K': 8, 'D': 1, 'P': 256, 'C': 64, 'links': [0]},
        ]
        self.with_X_transformation = True
        self.sorting_method = None
        self.with_global = True


class PointCNNWrapper(nn.Module):
    """Wrapper for PointCNN to provide segmentation output."""
    def __init__(self, num_classes=50):
        super().__init__()
        from pointcnn import PointCNN
        
        setting = PointCNNSetting(num_classes)
        self.backbone = PointCNN(setting)
        self.num_classes = num_classes
        
        # Classification head
        self.fc1 = nn.Linear(64 + 32, 128)  # Last layer features + global
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, points):
        """
        Args:
            points: [B, 3, N] point cloud
        
        Returns:
            logits: [B, N, num_classes] per-point class logits
            pts: [B, N, 3] point positions
            fts: [B, N, C] features
        """
        pts, fts = self.backbone(points, None, is_training=True)
        # pts: [B, P, 3], fts: [B, P, C]
        
        # Apply classification head to features
        B, P, C = fts.shape
        fts_flat = fts.reshape(B * P, C)
        logits_flat = self.fc2(self.drop1(torch.relu(self.fc1(fts_flat))))
        logits = logits_flat.reshape(B, P, self.num_classes)
        
        # Return [B, C, P] format for compatibility
        return logits.transpose(1, 2), pts, fts

# ============================================================
# Metrics Calculation
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
        
        # PointCNN forward pass
        outputs, _, _ = model(points)  # [B, C, N]
        outputs = outputs.transpose(1, 2).contiguous()  # [B, N, C]
        
        # Compute loss
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))
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
            
            # PointCNN forward pass
            outputs, _, _ = model(points)  # [B, C, N]
            outputs = outputs.transpose(1, 2).contiguous()  # [B, N, C]
            
            # Get predictions
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
    DATA_PATH = r"C:\Users\CSE_SDPL\Desktop\ModelNet40-PointNet\ShapenetPart\PartAnnotation"
    NUM_POINTS = 2048
    BATCH_SIZE = 4
    EPOCHS = 50
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = ShapeNetPartDataset(DATA_PATH, num_points=NUM_POINTS)
    NUM_CLASSES = dataset.num_classes
    
    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    # Initialize model
    model = PointCNNWrapper(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"🚀 Training PointCNN on ShapeNet Part")
    print(f"   Num classes: {NUM_CLASSES}")
    print(f"   Num samples: {len(dataset)}")
    print(f"   Num points: {NUM_POINTS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")

    # Training loop
    best_miou = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        oa, macc, miou = evaluate(model, eval_loader, device, NUM_CLASSES)
        
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss:.4f} | OA: {oa:.4f} | mAcc: {macc:.4f} | mIoU: {miou:.4f}")
        
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), "pointcnn_shapenet_part_best.pth")
            print(f"  ✓ Saved best model (mIoU: {miou:.4f})")

    # Final evaluation and save
    print("\n🎯 Final Evaluation:")
    oa, macc, miou = evaluate(model, eval_loader, device, NUM_CLASSES)
    print(f"OA: {oa:.4f}")
    print(f"mAcc: {macc:.4f}")
    print(f"mIoU: {miou:.4f}")
    
    torch.save(model.state_dict(), "pointcnn_shapenet_part_final.pth")
    print("✓ Model saved as pointcnn_shapenet_part_final.pth")
