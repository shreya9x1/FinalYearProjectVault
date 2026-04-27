# train_and_save.py
import os
import random
import json
import time
from collections import defaultdict
import multiprocessing

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

from data_reader import PointCloudDataset
from model import PointNet, pointnetloss

# --------------------------
# Config (you can tweak these)
# --------------------------
DATA_PATH = r"C:\Users\CSE_SDPL\Downloads\archive\ModelNet40"
OUT_DIR = "training_outputs"
BATCH_SIZE = 32
NUM_EPOCHS = 15
LR = 8e-4
SEED = 42
NUM_WORKERS = 4

# --------------------------
# Helper functions
# --------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_dataloaders(data_path, batch_size, num_workers, seed=42):
    full_dataset = PointCloudDataset(root_dir=data_path, valid=False, get_testset=False)

    # *** THE 5 TARGET CLASSES ***
    target_classnames = [
        "airplane", "bed", "bottle", "chair", "door",
        #"guitar", "lamp", "sofa", "table", "vase"
    ]

    # confirm classes exist
    for c in target_classnames:
        if c not in full_dataset.classes:
            raise RuntimeError(f"Class '{c}' not found in dataset.")

    target_class_ids = [full_dataset.classes[c] for c in target_classnames]

    # collect sample indices belonging to any of the classes
    selected_indices = [i for i, lbl in enumerate(full_dataset.labels) if lbl in target_class_ids]
    if len(selected_indices) == 0:
        raise RuntimeError("No samples found for selected classes")

    # deterministic split
    random.seed(seed)
    random.shuffle(selected_indices)
    N = len(selected_indices)
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)

    train_idx = selected_indices[:n_train]
    val_idx = selected_indices[n_train:n_train + n_val]
    test_idx = selected_indices[n_train + n_val:]

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    try:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=max(0, num_workers // 2), pin_memory=True)
    except Exception:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)

    return full_dataset, train_loader, val_loader, test_dataset

# --------------------------
# Training routines
# --------------------------
def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch in train_loader:
        pc = batch['pointcloud'].transpose(2, 1).to(device).float()
        lbls = batch['category'].to(device).long()

        optimizer.zero_grad()
        outputs, m3x3, m64x64 = model(pc)
        loss = pointnetloss(outputs, lbls, m3x3, m64x64)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * pc.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == lbls).sum().item()
        total_samples += pc.size(0)

    return total_loss / total_samples, total_correct / total_samples

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            pc = batch['pointcloud'].transpose(2, 1).to(device).float()
            lbls = batch['category'].to(device).long()
            outputs, m3x3, m64x64 = model(pc)
            loss = pointnetloss(outputs, lbls, m3x3, m64x64)

            total_loss += loss.item() * pc.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == lbls).sum().item()
            total_samples += pc.size(0)
    return total_loss / total_samples, total_correct / total_samples

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    full_dataset, train_loader, val_loader, test_dataset = make_dataloaders(
        DATA_PATH, BATCH_SIZE, NUM_WORKERS, seed=SEED
    )

    print("Loaded total dataset samples:", len(full_dataset))
    print("Train:", len(train_loader.dataset), "Val:", len(val_loader.dataset), "Test:", len(test_dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = PointNet(classes=len(full_dataset.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = defaultdict(list)
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, optimizer, train_loader, device)
        val_loss, val_acc = validate(model, val_loader, device)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nEpoch {epoch}")
        print(f"  TRAIN loss={tr_loss:.4f} acc={tr_acc:.4f}")
        print(f"  VAL   loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state': model.state_dict()}, os.path.join(OUT_DIR, "best.pth"))
            print("Saved best checkpoint")

    print("\nTraining complete! Best validation accuracy =", best_val_acc)

    # Save history & plots
    with open(os.path.join(OUT_DIR, "train_history.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=2)

    # plot curves (one plot per figure)
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel("epoch")
    plt.legend()
    plt.title("Loss")
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.xlabel("epoch")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(os.path.join(OUT_DIR, "acc_curve.png"))
    plt.close()

    print("Training complete. Best val acc:", best_val_acc)
    print("Outputs saved to", OUT_DIR)

    