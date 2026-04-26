import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# Import your dataloader
from data_reader_s3dis import S3DISDataset
# Import the model
from samnetpp_s3dis import SAMNetPP

# --- Configuration ---
BATCH_SIZE = 4
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
NUM_POINTS = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = r"C:\Users\CSE_SDPL\Desktop\ModelNet40-PointNet\preprocessed_Area_5"
def calculate_metrics(pred_np, target_np, num_classes):
    """
    Calculate OA, mAcc, mIoU using numpy/sklearn
    """
    # Flatten batches
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    
    # OA
    oa = accuracy_score(target_flat, pred_flat)
    
    # Confusion Matrix for per-class metrics
    cm = confusion_matrix(target_flat, pred_flat, labels=list(range(num_classes)))
    
    # mAcc (Mean Class Accuracy)
    # diagonal / sum of rows
    with np.errstate(divide='ignore', invalid='ignore'):
        class_acc = np.diag(cm) / cm.sum(axis=1)
    class_acc = np.nan_to_num(class_acc) # Replace NaNs (if a class is missing) with 0
    macc = np.mean(class_acc)
    
    # mIoU
    # IoU = TP / (TP + FP + FN)
    # TP = diag, FP = sum(col) - diag, FN = sum(row) - diag
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = intersection / union
    iou = np.nan_to_num(iou)
    miou = np.mean(iou)
    
    return oa, macc, miou

def get_one_hot(targets, nb_classes):
    res = torch.eye(nb_classes)[targets.reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def main():
    print(f"Using device: {DEVICE}")

    # 1. Prepare Datasets
    # Note: Using 'train' and 'test' splits if your dataloader supports it, 
    # otherwise relying on the random split logic in your dataloader.
    print("Loading Training Data...")
    train_dataset = S3DISDataset(root_dir=DATA_ROOT, num_points=NUM_POINTS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    
    print("Loading Testing Data...")
    test_dataset = S3DISDataset(root_dir=DATA_ROOT, num_points=NUM_POINTS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    num_part_classes = train_dataset.num_classes
    print(f"Total Part Classes: {num_part_classes}")

    # 2. Model Setup
    model = SAMNetPP(num_classes=num_part_classes, normal_channel=False).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    best_miou = 0.0

    # 3. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        # ShapeNet Part requires Object Category Label. 
        # Your provided dataloader returns (points, labels). 
        # It does NOT return the object category (0-15).
        # HACK: For this specific code to run without modifying your dataloader heavily,
        # we will generate a dummy object label or infer it if possible.
        # Ideally, your dataloader should return (points, label, object_cat_id).
        # Since it doesn't, we will pass a zero-tensor or simple mock for the category input
        # purely to make the architecture run.
        # *Recommendation*: Update your dataloader to return the object category index.
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for points, target in pbar:
            points, target = points.to(DEVICE), target.to(DEVICE)
            # points: [B, 3, N], target: [B, N]
            
            optimizer.zero_grad()
            
            # --- Dummy Category Label Handling ---
            # Creating a fake category input [B, 16] to satisfy architecture dimensions
            # In a real scenario, this helps the model distinguish a 'leg' of a chair from a 'leg' of a table.
            dummy_cat = torch.zeros(points.shape[0], 16).to(DEVICE) 
            
            # Forward
            pred = model(points, dummy_cat) # [B, N, num_classes]
            
            # Reshape for CrossEntropy: [B*N, C] vs [B*N]
            pred_flat = pred.contiguous().view(-1, num_part_classes)
            target_flat = target.view(-1, 1).squeeze()
            
            loss = F.nll_loss(pred_flat, target_flat)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
            
        scheduler.step()
        
        # 4. Evaluation Loop
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for points, target in tqdm(test_loader, desc="Evaluating"):
                points, target = points.to(DEVICE), target.to(DEVICE)
                dummy_cat = torch.zeros(points.shape[0], 16).to(DEVICE)
                
                pred = model(points, dummy_cat)
                pred_choice = pred.data.max(2)[1] # [B, N]
                
                all_preds.append(pred_choice.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        oa, macc, miou = calculate_metrics(all_preds, all_targets, num_part_classes)
        
        print(f"Epoch {epoch+1} Results:")
        print(f"  Loss: {train_loss / len(train_loader):.4f}")
        print(f"  OA  : {oa:.4f}")
        print(f"  mAcc: {macc:.4f}")
        print(f"  mIoU: {miou:.4f}")
        
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), 'best_samnetpp_s3dis_old.pth')
            print("  ✓ Saved best model")
            
    print("Training Complete.")

if __name__ == "__main__":
    main()