import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ClassificationHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.avgpool(x).flatten(1)
        logits = self.fc(x)
        return logits


class CAMGenerator(nn.Module):
    def __init__(self, feature_channels, num_classes):
        super(CAMGenerator, self).__init__()
        self.classifier = nn.Linear(feature_channels, num_classes, bias=False)

    def forward(self, feature_map):  # feature_map: [B, C, H, W, D]
        weights = self.classifier.weight  # [num_classes, feature_channels]
        B, C, H, W, D = feature_map.shape
        cam = torch.einsum('oc, bchwd -> bohwd', weights, feature_map)
        cam = F.relu(cam)  # ReLU as per definition
        return cam  # [B, num_classes, H, W, D]

# Initial pseudo-label generation

def generate_initial_pseudo_label(cam, phi_l=0.2, phi_h=0.8):
    # cam: [B, num_classes, H, W, D]  (normalized before)
    max_vals, argmax_class = cam.max(dim=1)  # [B, H, W, D]

    Y_p = torch.ones_like(max_vals, dtype=torch.long)  # Init as ignored (1)
    Y_p[max_vals >= phi_h] = argmax_class[max_vals >= phi_h]  # Confident foreground
    Y_p[max_vals <= phi_l] = 0  # Background
    return Y_p  # Shape: [B, H, W, D]

class AffinityCalculator(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super(AffinityCalculator, self).__init__()
        self.fc1 = nn.Linear(2 * feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x1, x2):
        # x1, x2: [N, feature_dim]
        f = torch.cat([x1, x2], dim=-1)
        f = F.relu(self.fc1(f))
        f = torch.sigmoid(self.fc2(f))
        return f.squeeze(-1)  # Shape: [N]


def extract_pixel_features(I, positions):
    """
    I: [H, W, D] - Grayscale image tensor
    positions: [N, 3] - voxel positions
    Return: [N, 4] - grayscale + i, j, k
    """
    intensities = I[positions[:, 0], positions[:, 1], positions[:, 2]].unsqueeze(-1)
    features = torch.cat([intensities, positions.float()], dim=-1)
    return features  # Shape: [N, 4]
def generate_affinity_labels(I, Y_p, window_size=3):
    """
    I: [H, W, D] - input image
    Y_p: [H, W, D] - pseudo label
    Return: affinity pairs, affinity labels
    """
    H, W, D = I.shape
    coords = torch.stack(torch.meshgrid(
        torch.arange(H), torch.arange(W), torch.arange(D)), dim=-1).reshape(-1, 3)

    affinity_pairs = []
    affinity_labels = []

    for idx in range(coords.shape[0]):
        i, j, k = coords[idx]
        label_i = Y_p[i, j, k]
        if label_i == 1:  # ignore region
            continue

        # Define window range
        for di in range(-window_size, window_size + 1):
            for dj in range(-window_size, window_size + 1):
                for dk in range(-window_size, window_size + 1):
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 0 <= ni < H and 0 <= nj < W and 0 <= nk < D:
                        label_j = Y_p[ni, nj, nk]
                        if label_j == 1:  # ignore region
                            continue

                        pos_i = torch.tensor([i, j, k])
                        pos_j = torch.tensor([ni, nj, nk])

                        affinity_pairs.append((pos_i, pos_j))

                        if label_i == label_j:
                            affinity_labels.append(1.0)
                        else:
                            affinity_labels.append(0.1)  # low positive affinity for different class

    return affinity_pairs, torch.tensor(affinity_labels)
  
def compute_affinity_loss(affinity_module, I, affinity_pairs, affinity_labels):
    features_i = []
    features_j = []

    for (pos_i, pos_j) in affinity_pairs:
        feat_i = extract_pixel_features(I, pos_i.unsqueeze(0))
        feat_j = extract_pixel_features(I, pos_j.unsqueeze(0))
        features_i.append(feat_i)
        features_j.append(feat_j)

    features_i = torch.cat(features_i, dim=0)
    features_j = torch.cat(features_j, dim=0)

    pred_aff = affinity_module(features_i, features_j)
    loss = F.mse_loss(pred_aff, affinity_labels.to(pred_aff.device))
    return loss

