import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ShapeNetPartDataset(Dataset):
    def __init__(self, root_dir, num_points=2048, split=None, train_ratio=0.8, seed=42):
        """
        ShapeNet Part segmentation dataset loader.
        
        Structure:
        root_dir/
            02691156/  (category ID)
                points/
                    1234567890.pts
                points_label/
                    body/
                        1234567890.seg
                    engine/
                        1234567890.seg
                    ...
        """
        self.root_dir = root_dir
        self.num_points = num_points
        self.samples = []  # List of (points_file, label_file)
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        
        # Map category IDs to names
        self.category_names = {
            '02691156': 'Airplane',
            '02773838': 'Bag',
            '02954340': 'Cap',
            '02958343': 'Car',
            '03001627': 'Chair',
            '03261776': 'Earphone',
            '03467517': 'Guitar',
            '03624134': 'Knife',
            '03636649': 'Lamp',
            '03642806': 'Laptop',
            '03790512': 'Motorbike',
            '03797390': 'Mug',
            '03948459': 'Pistol',
            '04099429': 'Rocket',
            '04225987': 'Skateboard',
            '04379243': 'Table',
        }
        
        # Scan for all point cloud files
        for category_id in sorted(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category_id)
            if not os.path.isdir(category_path):
                continue
            
            points_dir = os.path.join(category_path, 'points')
            labels_dir = os.path.join(category_path, 'points_label')
            
            if not os.path.exists(points_dir) or not os.path.exists(labels_dir):
                continue
            
            # Find matching point-label pairs
            for pts_file in os.listdir(points_dir):
                if not pts_file.endswith('.pts'):
                    continue
                
                base_name = pts_file[:-4]  # Remove .pts extension
                
                # Find corresponding label file in any subdirectory
                label_file = None
                seg_file_name = base_name + '.seg'
                
                for part_type in os.listdir(labels_dir):
                    part_dir = os.path.join(labels_dir, part_type)
                    if not os.path.isdir(part_dir):
                        continue
                    
                    seg_path = os.path.join(part_dir, seg_file_name)
                    if os.path.exists(seg_path):
                        label_file = seg_path
                        break
                
                if label_file:
                    pts_path = os.path.join(points_dir, pts_file)
                    self.samples.append((pts_path, label_file))
        
        print(f"Found {len(self.samples)} samples in ShapeNet Part dataset")

        # Apply optional split (train/test) if requested. If the dataset
        # does not provide explicit split files, perform a deterministic
        # split using `train_ratio` and `seed` so runs are reproducible.
        if split in ('train', 'test'):
            rng = np.random.RandomState(self.seed)
            indices = np.arange(len(self.samples))
            rng.shuffle(indices)
            split_idx = int(len(indices) * self.train_ratio)
            if split == 'train':
                selected = indices[:split_idx]
            else:
                selected = indices[split_idx:]

            self.samples = [self.samples[i] for i in selected]
            print(f"Using split='{split}' with {len(self.samples)} samples")
        
        # Determine unique part labels across all samples
        self.unique_labels = set()
        for i in range(min(100, len(self.samples))):
            _, label_file = self.samples[i]
            labels = np.loadtxt(label_file, dtype=int)
            if labels.ndim == 0:
                labels = np.array([labels])
            self.unique_labels.update(labels.tolist())
        
        self.num_classes = len(self.unique_labels)
        print(f"Detected {self.num_classes} part classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pts_path, label_path = self.samples[idx]
        
        try:
            # Load point cloud
            points = np.loadtxt(pts_path, dtype=np.float32)
            
            # Load labels
            labels = np.loadtxt(label_path, dtype=np.int32)
            if labels.ndim == 0:
                labels = np.array([labels])
            
            # Ensure we have same number of points and labels
            if len(points) != len(labels):
                min_len = min(len(points), len(labels))
                points = points[:min_len]
                labels = labels[:min_len]
            
            # Random sampling to fixed number of points
            if len(points) > self.num_points:
                indices = np.random.choice(len(points), self.num_points, replace=False)
            else:
                # Pad with repetition if needed
                indices = np.arange(len(points))
                if len(points) < self.num_points:
                    extra_indices = np.random.choice(len(points), self.num_points - len(points), replace=True)
                    indices = np.concatenate([indices, extra_indices])
            
            points = points[indices]
            labels = labels[indices]
            
            # Normalize points to [-1, 1]
            centroid = np.mean(points, axis=0)
            points = points - centroid
            furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
            if furthest_distance > 0:
                points = points / furthest_distance
            
            # Convert to tensors: [N, 3] -> [3, N] for model
            points_tensor = torch.tensor(points.T, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            return points_tensor, labels_tensor
            
        except Exception as e:
            print(f"Error loading {pts_path}: {e}")
            # Return zero tensors as fallback
            return torch.zeros(3, self.num_points, dtype=torch.float32), torch.zeros(self.num_points, dtype=torch.long)
