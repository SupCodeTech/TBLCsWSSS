import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from networks.SwinUNeLCsT import SwinUNeLCsT
from tool.classification_cam_affinity import ClassificationHead, CAMGenerator, generate_initial_pseudo_label, AffinityCalculator, generate_affinity_labels, compute_affinity_loss

from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd, ToTensord

import os
import numpy as np


class CTDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data_list[idx])
        data = np.load(img_path, allow_pickle=True).item()
        image = data['image']  # [H, W, D] numpy
        label = data['label']  # [H, W, D] numpy
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & Loader
    train_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        AddChanneld(keys=['image']),
        ScaleIntensityd(keys=['image']),
        ToTensord(keys=['image', 'label']),
    ])

    train_dataset = CTDataset(data_dir='./data/train', transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)

    # Model + Modules
    net = SwinUNeLCsT(in_channels=1, out_channels=3).to(device)
    classifier_head = ClassificationHead(in_channels=512, num_classes=2).to(device)
    cam_generator = CAMGenerator(feature_channels=512, num_classes=2).to(device)
    affinity_module = AffinityCalculator(feature_dim=4).to(device)

    optimizer = optim.Adam(list(net.parameters()) + list(classifier_head.parameters()) + 
                           list(cam_generator.parameters()) + list(affinity_module.parameters()), lr=1e-4)

    ce_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(50):
        net.train()
        total_loss = 0
        for batch in train_loader:
            img = batch['image'].to(device)  # [B, 1, H, W, D]
            gt = batch['label'].long().to(device)  # [B, H, W, D]

            optimizer.zero_grad()

            # --- Forward Pass ---
            logits, _, _, _ = net(img)
            seg_loss = ce_loss_fn(logits, gt)

            # --- CAM & Initial Pseudo Label ---
            deep_feature = logits  # Replace with correct feature before final conv if needed
            cls_logits = classifier_head(deep_feature)
            cam_map = cam_generator(deep_feature)
            cam_map = F.softmax(cam_map, dim=1)
            pseudo_label = generate_initial_pseudo_label(cam_map[0].detach())  # Single batch for affinity

            # --- Affinity Loss ---
            affinity_pairs, affinity_gt = generate_affinity_labels(img[0, 0].detach().cpu(), pseudo_label.cpu())
            if len(affinity_pairs) > 0:
                affinity_loss = compute_affinity_loss(affinity_module, img[0, 0], affinity_pairs, affinity_gt)
            else:
                affinity_loss = torch.tensor(0.0, requires_grad=True).to(device)

            total_batch_loss = seg_loss + 0.1 * affinity_loss
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()

        print(f"Epoch {epoch + 1} - Total Loss: {total_loss:.4f}")

if __name__ == "__main__":
    main()
