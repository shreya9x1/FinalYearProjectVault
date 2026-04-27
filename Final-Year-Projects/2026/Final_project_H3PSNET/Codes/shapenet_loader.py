import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ShapeNetPartDataset(Dataset):
    def __init__(self, root_dir, num_points=2048, split='train', train_ratio=0.8, seed=42):
        self.root_dir = root_dir
        self.num_points = num_points
        
        # Category ID to Name mapping
        self.cat_id_to_name = {
            '02691156': 'Airplane', '02773838': 'Bag', '02954340': 'Cap',
            '02958343': 'Car', '03001627': 'Chair', '03261776': 'Earphone',
            '03467517': 'Guitar', '03624134': 'Knife', '03636649': 'Lamp',
            '03642806': 'Laptop', '03790512': 'Motorbike', '03797390': 'Mug',
            '03948459': 'Pistol', '04099429': 'Rocket', '04225987': 'Skateboard',
            '04379243': 'Table'
        }
        
        # Standard Global Offsets for ShapeNet Part (50 parts total)
        self.category_offsets = {
            '02691156': 0, '02773838': 4, '02954340': 6, '02958343': 8,
            '03001627': 12, '03261776': 16, '03467517': 19, '03624134': 22,
            '03636649': 24, '03642806': 28, '03790512': 30, '03797390': 36,
            '03948459': 38, '04099429': 41, '04225987': 44, '04379243': 47
        }

        self.samples = []
        for cat_id, offset in self.category_offsets.items():
            cat_path = os.path.join(root_dir, cat_id)
            pts_dir = os.path.join(cat_path, 'points')
            lbl_dir = os.path.join(cat_path, 'points_label')
            
            if not os.path.exists(pts_dir): continue
            
            for f in os.listdir(pts_dir):
                if f.endswith('.pts'):
                    base = f[:-4]
                    # Find which subfolder the label is in
                    label_path = None
                    for part_folder in os.listdir(lbl_dir):
                        p_path = os.path.join(lbl_dir, part_folder, base + '.seg')
                        if os.path.exists(p_path):
                            label_path = p_path
                            break
                    if label_path:
                        self.samples.append({
                            'pts': os.path.join(pts_dir, f),
                            'lbl': label_path,
                            'offset': offset,
                            'cat_name': self.cat_id_to_name[cat_id]
                        })

        # Train/Test Split
        rng = np.random.RandomState(seed)
        indices = np.arange(len(self.samples))
        rng.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        self.indices = indices[:split_idx] if split == 'train' else indices[split_idx:]
        
        self.num_classes = 50

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.samples[self.indices[idx]]
        points = np.loadtxt(sample['pts'], dtype=np.float32)
        labels = np.loadtxt(sample['lbl'], dtype=np.int32)
        
        # Apply Offset to make labels GLOBAL (0-49)
        labels = labels + sample['offset']
        
        # Resample points
        if len(points) > self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
        else:
            choice = np.concatenate([np.arange(len(points)), 
                                    np.random.choice(len(points), self.num_points - len(points), replace=True)])
        
        points, labels = points[choice], labels[choice]
        
        # Normalize
        points = points - np.mean(points, axis=0)
        dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if dist > 0: points /= dist
            
        return torch.tensor(points.T, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)