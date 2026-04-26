import os
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt

from pointsam import PointSAM_Segmentation   # must match training

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = r"C:\Users\CSE_SDPL\Desktop\ModelNet40-PointNet\pointsam_shapenet_final.pth"

SHAPENET_ROOT = r"C:\Users\CSE_SDPL\Desktop\ModelNet40-PointNet\ShapenetPart\PartAnnotation"

CATEGORY = "03001627"   # CHAIR
SAMPLE_ID = "1a6f615e8b1b5ae4dbbc9440457e303e"  

#C:\Users\CSE_SDPL\Desktop\ModelNet40-PointNet\ShapenetPart\PartAnnotation\03001627\points\1a6f615e8b1b5ae4dbbc9440457e303e.pts

PARTS = ["arm", "back", "seat", "leg"]
PART_TO_LABEL = {p: i for i, p in enumerate(PARTS)}

NUM_CLASSES = 2  # model output
NUM_POINTS = 4096   # must match training

# -------------------------------------------------
# LOAD POINTS
# -------------------------------------------------
points_path = os.path.join(
    SHAPENET_ROOT, CATEGORY, "points", f"{SAMPLE_ID}.pts"
)
assert os.path.exists(points_path)

points = np.loadtxt(points_path).astype(np.float32)
points = points[:, :3]   # XYZ only
N = points.shape[0]

# -------------------------------------------------
# BUILD GROUND TRUTH LABELS (MERGE .seg FILES)
# -------------------------------------------------
gt_labels = np.full(N, -1, dtype=np.int64)

for part_name in PARTS:
    seg_path = os.path.join(
        SHAPENET_ROOT,
        CATEGORY,
        "points_label",
        part_name,
        f"{SAMPLE_ID}.seg"
    )

    if not os.path.exists(seg_path):
        continue

    mask = np.loadtxt(seg_path).astype(np.int64)  # 0/1
    gt_labels[mask == 1] = PART_TO_LABEL[part_name]

# Safety check
assert (gt_labels >= 0).all(), "Some points have no part label!"

# -------------------------------------------------
# SAMPLE TO FIXED SIZE
# -------------------------------------------------
if N >= NUM_POINTS:
    idx = np.random.choice(N, NUM_POINTS, replace=False)
else:
    idx = np.random.choice(N, NUM_POINTS, replace=True)

points = points[idx]
gt_labels = gt_labels[idx]

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = PointSAM_Segmentation(num_classes=NUM_CLASSES)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt, strict=True)

model.to(DEVICE)
model.eval()

# -------------------------------------------------
# INFERENCE
# -------------------------------------------------
with torch.no_grad():
    pts = torch.from_numpy(points).unsqueeze(0)   # (1, N, 3)
    pts = pts.transpose(2, 1).to(DEVICE)          # (1, 3, N)

    logits = model(pts)                           # (1, N, 50) OR (1, 50, N)

    if logits.shape[1] == NUM_CLASSES:
        logits = logits.transpose(2, 1)

    preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

print("Pred labels:", np.unique(preds))
print("GT labels:", np.unique(gt_labels))

# -------------------------------------------------
# VISUALIZATION
# -------------------------------------------------
def visualize(points, labels, title):
    cmap = plt.get_cmap("tab20")
    colors = cmap(labels % 20)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(title)
    o3d.visualization.draw_geometries([pcd])

visualize(points, preds, "Predicted Segmentation")
visualize(points, gt_labels, "Ground Truth Segmentation")
