import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from affinity_refinement_module import (
    compute_affinity_cosine,
    focal_affinity_loss,
    AffinityRefinementModule
)

from swin_unelcst_backbone import SwinUNeLCsT
from my_dataset import My3DMedicalDataset


# ==== 超参数 ====
num_epochs = 200
batch_size = 2
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== 数据集与DataLoader ====
train_dataset = My3DMedicalDataset(split='train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ==== 模型、Affinity 模块与优化器 ====
backbone = SwinUNeLCsT().to(device)
affinity_refiner = AffinityRefinementModule(eta=0.4, num_regions=15, overlap=5).to(device)

optimizer = optim.Adam(list(backbone.parameters()) + list(affinity_refiner.parameters()), lr=learning_rate)
segmentation_loss_fn = nn.BCEWithLogitsLoss()


# ==== 训练循环 ====
for epoch in range(num_epochs):
    backbone.train()
    affinity_refiner.train()
    
    total_loss = 0
    for images, gt_masks in train_loader:
        images = images.to(device)           # [B, 1, H, W, D]
        gt_masks = gt_masks.to(device)       # [B, 1, H, W, D]

        optimizer.zero_grad()

        # ---- 特征提取与伪标签预测 ----
        features, pred_masks = backbone(images)   # features: [B, C, H, W, D], pred_masks: [B, 1, H, W, D]

        # ---- 亲和力矩阵计算 ----
        affinity_matrix = compute_affinity_cosine(features)   # [B, N, N]

        # ---- 伪标签传播 ----
        act_map = torch.sigmoid(pred_masks.permute(0, 2, 3, 4, 1))  # [B, H, W, D, 1]
        propagated_map = affinity_refiner(act_map, affinity_matrix.view(images.size(0), *[images.size(2)]*3, *[images.size(2)]*3))

        propagated_map = propagated_map.permute(0, 4, 1, 2, 3)  # [B, 1, H, W, D]

        # ---- 分割损失 ----
        seg_loss = segmentation_loss_fn(pred_masks, gt_masks)

        # ---- Affinity 损失 ----
        with torch.no_grad():
            affinity_gt = (gt_masks.view(batch_size, -1, 1) == gt_masks.view(batch_size, 1, -1)).float()
        affinity_loss = focal_affinity_loss(affinity_matrix, affinity_gt)

        loss = seg_loss + 0.1 * affinity_loss  # 损失加权

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss/len(train_loader):.4f}")
