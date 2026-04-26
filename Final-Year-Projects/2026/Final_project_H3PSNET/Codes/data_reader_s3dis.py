import os
import numpy as np
import torch
from torch.utils.data import Dataset

class S3DISDataset(Dataset):
    def __init__(self, root_dir, num_points=2048):
        self.root_dir = root_dir
        self.num_points = num_points
        self.rooms = []
        self.num_classes = 13
        # ✅ Official 13 S3DIS class mapping
        self.class_map = {
            0: "ceiling",
            1: "floor",
            2: "wall",
            3: "beam",
            4: "column",
            5: "window",
            6: "door",
            7: "table",
            8: "chair",
            9: "sofa",
            10: "bookcase",
            11: "board",
            12: "clutter",
        }

        # ✅ Inverse map for label normalization
        self.label_map = {v: k for k, v in self.class_map.items()}

        # Load all room file paths
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".txt"):
                    self.rooms.append(os.path.join(root, file))

        print(f"Found {len(self.rooms)} room files in {root_dir}")

    def __len__(self):
        return len(self.rooms)

    def __getitem__(self, idx):
        file_path = self.rooms[idx]
        try:
            data = np.loadtxt(file_path, dtype=np.float32, delimiter=' ')
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            # Return zero tensor as fallback
            return torch.zeros(6, self.num_points, dtype=torch.float32), torch.zeros(self.num_points, dtype=torch.long)

        if data.ndim == 1:  # Single point case
            data = data.reshape(1, -1)
        
        points = data[:, 0:6]  # XYZ + RGB
        labels = data[:, -1].astype(int)

        # ✅ Filter valid labels (0–12 only)
        valid_mask = np.isin(labels, list(range(13)))
        points = points[valid_mask]
        labels = labels[valid_mask]

        # Random sampling
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
        else:
            indices = np.arange(len(points))
        
        pts_sampled = points[indices]
        lbls_sampled = labels[indices]

        # Return per-point features and labels (transpose to [C, N] for model input)
        pts_transposed = torch.tensor(pts_sampled.T, dtype=torch.float32)  # [6, num_points]
        lbls_tensor = torch.tensor(lbls_sampled, dtype=torch.long)  # [num_points]

        return pts_transposed, lbls_tensor

if __name__ == "__main__":
    dataset = S3DISDataset(
        r"C:\Users\CSE_SDPL\Downloads\Stanford3dDataset_v1.2_Aligned_Version\Stanford3dDataset_v1.2_Aligned_Version\Area_5"
    )
    print("✅ Dataset loaded successfully.")
    print(f"Total rooms: {len(dataset)}")
    sample = dataset[0]
    print("Pointcloud shape:", sample["pointcloud"].shape)
    print("Category:", sample["category"])
