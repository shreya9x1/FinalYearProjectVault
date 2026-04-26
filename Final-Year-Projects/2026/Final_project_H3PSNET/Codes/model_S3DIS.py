# model_s3dis.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Input and Feature Transformation Networks (T-Net)
# ---------------------------------------------------------
class Tnet(nn.Module):
    def __init__(self, k=3):
        super(Tnet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = torch.max(xb, 2, keepdim=True)[0]
        flat = pool.view(-1, 1024)

        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))
        matrix = self.fc3(xb).view(-1, self.k, self.k)

        # Add identity for stability
        init = torch.eye(self.k, device=input.device).unsqueeze(0).repeat(bs, 1, 1)
        matrix = matrix + init
        return matrix


# ---------------------------------------------------------
# 2. PointNet for Semantic Segmentation
# ---------------------------------------------------------
class PointNetSegmentation(nn.Module):
    def __init__(self, in_channels=6, num_classes=13):
        super(PointNetSegmentation, self).__init__()
        self.input_transform = Tnet(k=in_channels)
        self.feature_transform = Tnet(k=64)

        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, num_classes, 1)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

    def forward(self, x):
        bs, d, n = x.size()

        # Input Transform
        t_input = self.input_transform(x)
        x = torch.bmm(t_input, x)

        # Feature Extraction
        x = F.relu(self.bn1(self.conv1(x)))
        t_feat = self.feature_transform(x)
        x = torch.bmm(t_feat, x)
        point_features = x.clone()

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        global_features = torch.max(x, 2, keepdim=True)[0]

        # Concatenate global and local features
        global_features_expanded = global_features.repeat(1, 1, n)
        concat_features = torch.cat([point_features, global_features_expanded], 1)

        # Segmentation MLP
        x = F.relu(self.bn4(self.conv4(concat_features)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.conv7(x)

        return x, t_input, t_feat


# ---------------------------------------------------------
# 3. PointNet Segmentation Loss
# ---------------------------------------------------------
def pointnet_seg_loss(pred, target, m3x3, m64x64, alpha=0.001):
    criterion = nn.CrossEntropyLoss()
    bs = pred.size(0)
    id3x3 = torch.eye(3, device=pred.device).unsqueeze(0).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, device=pred.device).unsqueeze(0).repeat(bs, 1, 1)

    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))

    return criterion(pred, target) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)
