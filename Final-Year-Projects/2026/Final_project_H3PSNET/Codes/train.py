


# train.py

import torch
from torch.utils.data import DataLoader, Subset
from model import PointNet, pointnetloss
from data_reader import PointCloudDataset
import random

# Dataset path
DATA_PATH = r"C:\Users\CSE_SDPL\Downloads\archive\ModelNet40"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the full dataset
full_dataset = PointCloudDataset(root_dir=DATA_PATH, valid=False, get_testset=False)
print(f"Loaded total samples: {len(full_dataset)}")

# Find the label ID for the "chair" class
chair_class_id = full_dataset.classes.get("chair")
if chair_class_id is None:
    raise ValueError("'chair' class not found in ModelNet40!")

# Get indices of only the 'chair' class
chair_indices = [i for i, lbl in enumerate(full_dataset.labels) if lbl == chair_class_id]

# Randomly select up to 2000 samples (or fewer if not available)
selected_indices = random.sample(chair_indices, min(2000, len(chair_indices)))

# Create a subset dataset of only chairs
chair_dataset = Subset(full_dataset, selected_indices)
print(f" Using {len(chair_dataset)} chair samples for training.")

# DataLoader
train_loader = DataLoader(chair_dataset, batch_size=8, shuffle=True, drop_last=True)

# Model setup
model = PointNet(classes=len(full_dataset.classes)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)

# Training loop
for epoch in range(50):
    running_loss = 0.0
    for batch in train_loader:
        pointcloud = batch['pointcloud']
        label = batch['category']

        # Convert to correct shape and dtype
        pointcloud = pointcloud.transpose(2, 1).to(device).float()
        label = label.to(device).long()

        # Forward + backward + optimize
        optimizer.zero_grad()
        outputs, m3x3, m64x64 = model(pointcloud)
        loss = pointnetloss(outputs, label, m3x3, m64x64)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()




        

    print(f"Epoch [{epoch+1}/50], Loss: {running_loss/len(train_loader):.4f}")

print("Training on 2000 chair samples complete!")

