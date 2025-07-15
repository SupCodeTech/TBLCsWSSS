import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ClassificationHead, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x: [B, C, H, W, D]
        x = self.global_pool(x).view(x.size(0), -1)  # [B, C]
        x = self.fc(x)  # [B, num_classes]
        return x

class CAMGenerator(nn.Module):
    def __init__(self, feature_channels, num_classes):
        super(CAMGenerator, self).__init__()
        self.conv = nn.Conv3d(feature_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)  # Output shape: [B, num_classes, H, W, D]

import numpy as np

def generate_initial_pseudo_label(cam_map):
    # cam_map: [num_classes, H, W, D], numpy or torch
    if torch.is_tensor(cam_map):
        cam_map = cam_map.detach().cpu().numpy()
    pseudo_label = np.argmax(cam_map, axis=0).astype(np.uint8)  # Shape: [H, W, D]
    return pseudo_label

class AffinityCalculator(nn.Module):
    def __init__(self, feature_dim):
        super(AffinityCalculator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=-1)  # [..., feature_dim*2]
        affinity = self.mlp(x).squeeze(-1)  # [...], affinity score between 0-1
        return affinity

import random

def generate_affinity_labels(image, pseudo_label, neighbor_offsets=[(1,0,0), (0,1,0), (0,0,1)], max_samples=500):
    # image: [H, W, D], torch.Tensor
    # pseudo_label: [H, W, D], numpy.ndarray
    affinity_pairs = []
    affinity_gt = []

    H, W, D = pseudo_label.shape
    coords = torch.nonzero(torch.ones(H, W, D)).tolist()
    random.shuffle(coords)

    count = 0
    for x, y, z in coords:
        for dx, dy, dz in neighbor_offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < H and 0 <= ny < W and 0 <= nz < D:
                feat1 = image[x, y, z].item()
                feat2 = image[nx, ny, nz].item()
                label1 = pseudo_label[x, y, z]
                label2 = pseudo_label[nx, ny, nz]

                affinity_pairs.append((feat1, feat2))
                affinity_gt.append(1 if label1 == label2 else 0)

                count += 1
                if count >= max_samples:
                    return affinity_pairs, affinity_gt

    return affinity_pairs, affinity_gt

def compute_affinity_loss(affinity_module, image, affinity_pairs, affinity_gt):
    # image: [H, W, D], torch.Tensor
    # affinity_pairs: List of (feat1, feat2)
    # affinity_gt: List of 0 or 1
    device = image.device
    feats = torch.tensor(affinity_pairs, dtype=torch.float32, device=device).unsqueeze(-1)  # [N, 2, 1]
    feats = feats.permute(0, 2, 1).reshape(-1, 2)  # [N, 2]
    gt = torch.tensor(affinity_gt, dtype=torch.float32, device=device)  # [N]

    affinity_pred = affinity_module(feats[:, 0:1], feats[:, 1:2])  # [N]
    loss = F.binary_cross_entropy(affinity_pred, gt)
    return loss
