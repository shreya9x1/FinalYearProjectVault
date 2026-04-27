# data_reader.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import open3d as o3d

class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s) * pt2[i] + (1-t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros(len(faces))
        for i in range(len(areas)):
            areas[i] = self.triangle_area(verts[faces[i][0]], verts[faces[i][1]], verts[faces[i][2]])

        sampled_faces = random.choices(faces, weights=areas, k=self.output_size)
        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = self.sample_point(verts[sampled_faces[i][0]], verts[sampled_faces[i][1]], verts[sampled_faces[i][2]])

        return sampled_points

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        return torch.from_numpy(pointcloud)

def default_transforms():
    return transforms.Compose([
        PointSampler(1024),
        Normalize(),
        ToTensor()
    ])

class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
        return norm_pointcloud

def read_off(file_path):
    with open(file_path, 'r') as file:
        off_header = file.readline().strip()
        if 'OFF' == off_header:
            n_verts, n_faces, __ = [int(s) for s in file.readline().strip().split(' ')]
        else:
            n_verts, n_faces, __ = [int(s) for s in off_header[3:].split(' ')]
        verts = [[float(s) for s in file.readline().strip().split(' ')] for _ in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for _ in range(n_faces)]
        return verts, faces

class PointCloudDataset(Dataset):
    def __init__(self, root_dir, valid=False, get_testset=False, transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dr for dr in sorted(os.listdir(root_dir)) if os.path.isdir(f'{root_dir}/{dr}')]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        self.labels = []
        sub_folder = 'test' if get_testset else 'train'

        for class_name, label in self.classes.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                class_dir += f'/{sub_folder}/'
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.off'):
                        self.files.append(os.path.join(class_dir, file_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file_path):
        verts, faces = read_off(file_path)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]
        label = self.labels[idx]
        pointcloud = self.__preproc__(pcd_path)
        return {'pointcloud': pointcloud, 'category': label}
