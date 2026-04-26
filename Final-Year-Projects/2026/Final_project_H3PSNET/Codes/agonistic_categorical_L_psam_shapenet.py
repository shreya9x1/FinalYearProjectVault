import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# --- IMPORTS (Ensure these match your files) ---
from shapenet_loader import ShapeNetPartDataset
from pointsam import PointSAM_Segmentation

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = r"C:\Users\CSE_SDPL\Desktop\ModelNet40-PointNet\ShapenetPart\PartAnnotation"
CHECKPOINT_PATH = 'best_psam_shapenet_relief.pth'
NUM_PART_CLASSES = 2
NUM_POINTS = 2048

# ShapeNet Part Names for readability
SHAPENET_PARTS = {
    'Airplane': [0, 1, 2, 3], 'Bag': [4, 5], 'Cap': [6, 7], 'Car': [8, 9, 10, 11],
    'Chair': [12, 13, 14, 15], 'Earphone': [16, 17, 18], 'Guitar': [19, 20, 21],
    'Knife': [22, 23], 'Lamp': [24, 25, 26, 27], 'Laptop': [28, 29],
    'Motorbike': [30, 31, 32, 33, 34, 35], 'Mug': [36, 37], 'Pistol': [38, 39, 40],
    'Rocket': [41, 42, 43], 'Skateboard': [44, 45, 46], 'Table': [47, 48, 49]
}

def evaluate_per_class():
    # 1. Load Model
    model = PointSAM_Segmentation(num_classes=NUM_PART_CLASSES).to(DEVICE)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model_state = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(filtered_state_dict)
    model.load_state_dict(model_state)
    model.eval()
    print(f"✓ Loaded weights from {CHECKPOINT_PATH}")

    # 2. Load Test Dataset
    test_dataset = ShapeNetPartDataset(root_dir=DATA_ROOT, num_points=NUM_POINTS, split='test')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    all_preds = []
    all_targets = []

    # 3. Inference
    print("Running Inference on Test Set...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if len(batch) == 3:
                points, label, target = batch
            else:
                points, target = batch
                label = torch.zeros(points.shape[0], 1).long()  # Dummy label if needed
            points, label, target = points.to(DEVICE), label.to(DEVICE), target.to(DEVICE)
            
            # Create Category One-Hot
            obj_cat = torch.zeros(points.shape[0], 16).to(DEVICE)
            obj_cat.scatter_(1, label.view(-1, 1), 1)

            pred = model(points)
            # Depending on your model output: [B, N, 50] or [B, 50, N]
            # We assume [B, N, 50] based on your previous max(2) usage
            pred_choice = pred.max(2)[1] 
            
            all_preds.append(pred_choice.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    # 4. Metric Calculation
    print("\n--- CALCULATING PER-CLASS MIOU ---")
    y_pred = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_targets).flatten()

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_PART_CLASSES)))
    
    # Intersection over Union formula: TP / (TP + FP + FN)
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection

    iou_per_class = intersection / (union + 1e-10)

    # 5. Display Results nicely grouped by category
    print(f"{'Category':<12} | {'Part Index':<10} | {'IoU':<6}")
    print("-" * 35)
    
    overall_miou = []
    
    for category, indices in SHAPENET_PARTS.items():
        cat_ious = []
        for idx in indices:
            # Only count classes that actually appeared in the test set
            if ground_truth_set[idx] > 0:
                val = iou_per_class[idx]
                print(f"{category:<12} | Part {idx:<5}    | {val:.4f}")
                cat_ious.append(val)
                overall_miou.append(val)
        
        if cat_ious:
            print(f"-> {category} Mean: {np.mean(cat_ious):.4f}\n")

    print("-" * 35)
    print(f"FINAL OVERALL mIoU: {np.mean(overall_miou):.4f}")

if __name__ == "__main__":
    evaluate_per_class()