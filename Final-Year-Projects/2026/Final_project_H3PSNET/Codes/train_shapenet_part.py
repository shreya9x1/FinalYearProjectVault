import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from shapenet_loader import ShapeNetPartDataset

# ============================================================
# Simple PointNet-based Segmentation Model
# ============================================================
class SimpleSegmentationNet(nn.Module):
    """Simple segmentation network for point clouds."""
    def __init__(self, num_classes=50):
        super().__init__()
        self.num_classes = num_classes
        
        # Shared MLP layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        # Max pool for global features
        self.global_mlp1 = nn.Conv1d(256, 512, 1)
        self.global_mlp2 = nn.Conv1d(512, 1024, 1)
        
        # Concatenate and segment
        self.seg_conv1 = nn.Conv1d(256 + 1024, 512, 1)
        self.seg_conv2 = nn.Conv1d(512, 256, 1)
        self.seg_conv3 = nn.Conv1d(256, 128, 1)
        self.seg_conv4 = nn.Conv1d(128, num_classes, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn_g1 = nn.BatchNorm1d(512)
        self.bn_g2 = nn.BatchNorm1d(1024)
        self.bn_s1 = nn.BatchNorm1d(512)
        self.bn_s2 = nn.BatchNorm1d(256)
        self.bn_s3 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, N] point cloud
        Returns:
            logits: [B, num_classes, N]
        """
        # Shared feature extraction
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        local_feat = torch.relu(self.bn3(self.conv3(x)))
        
        # Global feature
        global_feat = torch.max(local_feat, dim=2, keepdim=True)[0]
        global_feat = torch.relu(self.bn_g1(self.global_mlp1(global_feat)))
        global_feat = torch.relu(self.bn_g2(self.global_mlp2(global_feat)))
        
        # Expand global feature and concatenate
        B, _, N = local_feat.shape
        global_feat_expanded = global_feat.expand(B, -1, N)
        
        # Segmentation head
        feat = torch.cat([local_feat, global_feat_expanded], dim=1)
        feat = torch.relu(self.bn_s1(self.seg_conv1(feat)))
        feat = self.dropout(feat)
        feat = torch.relu(self.bn_s2(self.seg_conv2(feat)))
        feat = self.dropout(feat)
        feat = torch.relu(self.bn_s3(self.seg_conv3(feat)))
        logits = self.seg_conv4(feat)
        
        return logits, local_feat, global_feat_expanded

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
        
        # Forward pass
        logits, _, _ = model(points)  # [B, C, N]
        
        # Reshape for loss computation
        logits_reshaped = logits.transpose(1, 2).reshape(-1, logits.shape[1])  # [B*N, C]
        labels_reshaped = labels.reshape(-1)  # [B*N]
        
        loss = criterion(logits_reshaped, labels_reshaped)
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
            
            # Forward pass
            logits, _, _ = model(points)  # [B, C, N]
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)  # [B, N]
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
    BATCH_SIZE = 8
    EPOCHS = 50
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = ShapeNetPartDataset(DATA_PATH, num_points=NUM_POINTS)
    NUM_CLASSES = dataset.num_classes
    
    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=0)

    # Initialize model
    model = SimpleSegmentationNet(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"🚀 Training Simple Segmentation Network on ShapeNet Part")
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
            torch.save(model.state_dict(), "shapenet_part_segmentation_best.pth")
            print(f"  ✓ Saved best model (mIoU: {miou:.4f})")

    # Final evaluation and save
    print("\n🎯 Final Evaluation:")
    oa, macc, miou = evaluate(model, eval_loader, device, NUM_CLASSES)
    print(f"OA: {oa:.4f}")
    print(f"mAcc: {macc:.4f}")
    print(f"mIoU: {miou:.4f}")
    
    torch.save(model.state_dict(), "shapenet_part_segmentation_final.pth")
    print("✓ Model saved as shapenet_part_segmentation_final.pth")
