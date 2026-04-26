import os
import numpy as np
import torch
import json
from torch.utils.data import Dataset

class ShapeNetDRDataset(Dataset):
    def __init__(self, root_dir, num_points=2048):
        self.root_dir = root_dir
        self.num_points = num_points
        self.files = []

        # Load all ShapeNet part segmentation file paths (skip README and other non-data files)
        for root, _, files in os.walk(root_dir):
            for file in files:
                # Skip README, metadata, and non-data files
                if file.endswith(".txt") and not any(skip in file.lower() for skip in ["readme", "metadata", "json"]):
                    file_path = os.path.join(root, file)
                    # Quick validation: try to load first few lines to confirm it's numeric data
                    try:
                        test = np.loadtxt(file_path, max_rows=1)
                        if test.ndim >= 1 and len(test) >= 7:  # At least XYZ + RGB + label
                            self.files.append(file_path)
                    except Exception:
                        # Skip files that don't load as numeric data
                        continue

        print(f"Found {len(self.files)} valid ShapeNet part segmentation files in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        try:
            data = np.loadtxt(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            raise

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 7:
            raise ValueError(f"File {file_path} has only {data.shape[1]} columns, expected at least 7 (XYZ+RGB+label)")

        points = data[:, 0:6]  # XYZ + RGB
        labels = data[:, -1].astype(int)

        # Random sampling
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
        else:
            indices = np.arange(len(points))

        sampled_points = points[indices]
        sampled_labels = labels[indices]

        # Ensure labels are 0-based
        if sampled_labels.max() > 0 and sampled_labels.min() > 0:
            sampled_labels = sampled_labels - 1

        return torch.from_numpy(sampled_points.T).float(), torch.from_numpy(sampled_labels).long()
    def get_num_classes(self):
        return len(set(label for file_path in self.files 
                       for label in np.loadtxt(file_path)[:, -1].astype(int)))
    
#with open(r"C:\Users\CSE_SDPL\Desktop\ModelNet40-PointNet\ShapenetPart\PartAnnotation\metadata.json") as json_file:
#    metadata = json.load(json_file)

#print(f"ShapeNet part segmentation metadata: {metadata}")