# benchmark.py
import os
import time
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, Subset

from data_reader import PointCloudDataset
from model import PointNet

# --------------------------
# Config
# --------------------------
DATA_PATH = r"C:\Users\CSE_SDPL\Downloads\archive\ModelNet40"
CKPT_PATH = "training_outputs/final.pth"   # or the best checkpoint saved
BATCH_SIZE = 16
WARMUP_ITERS = 10
MEASURE_ITERS = 100
OUT_DIR = "benchmark_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------
# prepare test dataset (same split used in training)
# We replicate the split exactly using the same seed logic as train_and_save.py
# --------------------------
SEED = 42
import random, numpy as np
random.seed(SEED); np.random.seed(SEED)

full_dataset = PointCloudDataset(root_dir=DATA_PATH, valid=False, get_testset=False)
chair_id = full_dataset.classes["chair"]
chair_indices = [i for i, lbl in enumerate(full_dataset.labels) if lbl == chair_id]
N = len(chair_indices)

n_train = int(0.70 * N)
n_val = int(0.15 * N)
# n_test = N - n_train - n_val
random.shuffle(chair_indices)
test_idx  = chair_indices[n_train+n_val:]

test_dataset = Subset(full_dataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# --------------------------
# load model
# --------------------------
model = PointNet(classes=len(full_dataset.classes)).to(device)
if os.path.exists(CKPT_PATH):
    ck = torch.load(CKPT_PATH, map_location=device)
    if 'model_state' in ck:
        model.load_state_dict(ck['model_state'])
    else:
        model.load_state_dict(ck)
else:
    raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")
model.eval()

# --------------------------
# Evaluate accuracy & confusion matrix
# --------------------------
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        pc = batch['pointcloud'].transpose(2,1).to(device).float()
        lbls = batch['category'].numpy()
        outputs, m3x3, m64x64 = model(pc)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(lbls.tolist())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

acc = (all_preds == all_labels).mean()
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

results = {
    "num_test_samples": len(test_dataset),
    "accuracy": float(acc),
    "classification_report": report
}

# save confusion matrix CSV
pd.DataFrame(cm).to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"), index=False)

# --------------------------
# Latency & throughput measurement
# --------------------------
# Create a small fixed batch for measurement (randomly pick from test set) to measure many iters
examples = []
labels = []
for i in range(min(64, len(test_dataset))):
    item = test_dataset[i]
    examples.append(item['pointcloud'].transpose(1,0).numpy())  # (3,N) after transpose
    labels.append(item['category'])
examples = np.stack(examples, axis=0)  # (B,3,N)
examples_t = torch.from_numpy(examples).to(device).float()

# Warmup
if device.type == 'cuda':
    torch.cuda.synchronize()
for i in range(WARMUP_ITERS):
    _ = model(examples_t)

if device.type == 'cuda':
    torch.cuda.synchronize()
# timed runs
start = time.time()
iter_count = 0
for i in range(MEASURE_ITERS):
    _ = model(examples_t)
    iter_count += 1
if device.type == 'cuda':
    torch.cuda.synchronize()
end = time.time()
total_time = end - start
samples_processed = examples_t.size(0) * iter_count
latency_per_sample = total_time / samples_processed
throughput = samples_processed / total_time  # samples/sec

# GPU memory peak (if cuda)
gpu_memory = None
if device.type == 'cuda':
    gpu_memory = torch.cuda.max_memory_allocated(device) / (1024**2)  # MB
    torch.cuda.reset_peak_memory_stats(device)

results.update({
    "latency_per_sample_s": float(latency_per_sample),
    "throughput_samples_per_s": float(throughput),
    "gpu_memory_peak_mb": float(gpu_memory) if gpu_memory is not None else None
})

# --------------------------
# Save results
# --------------------------
with open(os.path.join(OUT_DIR, "benchmark_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Also write a CSV summary
pd.DataFrame([results]).to_csv(os.path.join(OUT_DIR, "benchmark_summary.csv"), index=False)

print("Benchmark results:")
print(json.dumps(results, indent=2))
print("Saved outputs to", OUT_DIR)
