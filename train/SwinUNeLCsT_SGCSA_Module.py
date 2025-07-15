import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from networks.SwinUNeLCsT import SwinUNeLCsT_WSSS
from tool.SGCSA_Module import SGCSA_Module

# ------------------------------
#  Configurations
# ------------------------------
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------
#  Dummy Dataset (replace with real dataset)
# ------------------------------
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, ScaleIntensityd, ToTensord, EnsureTyped, Orientationd, Spacingd
)
import os

data_dir = "/data/train/"

images = sorted([os.path.join(data_dir, "images", f) for f in os.listdir(os.path.join(data_dir, "images")) if f.endswith(".nii.gz")])
labels = sorted([os.path.join(data_dir, "labels", f) for f in os.listdir(os.path.join(data_dir, "labels")) if f.endswith(".nii.gz")])

train_files = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    EnsureTyped(keys=["image", "label"]),
])

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

for batch_data in train_loader:
    inputs = batch_data["image"].to(DEVICE)
    labels = batch_data["label"].long().to(DEVICE)  # 转为 long tensor 用于 cross-entropy
    outputs, loc_out, num_out = model(inputs)
    # 后续 loss 和训练步骤...


train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------
#  Model & SGCSA Integration
# ------------------------------
model = SwinUNeLCsT(
    in_channels=1,
    out_channels=3,
    img_size=(128, 128, 96),
    feature_size=16
).to(DEVICE)

sgcsa_module = SGCSA_Module(num_layers=2, in_features=1, hidden_features=64).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ------------------------------
#  Exponential Moving Average (EMA) Init
# ------------------------------
ema_decay = 0.99
ema_model = SwinUNeLCsT(
    in_channels=1,
    out_channels=3,
    img_size=(128, 128, 96),
    feature_size=16
).to(DEVICE)
ema_model.load_state_dict(model.state_dict())

def update_ema(model, ema_model, decay):
    with torch.no_grad():
        msd = model.state_dict()
        emsd = ema_model.state_dict()
        for key in msd.keys():
            emsd[key].data.copy_(emsd[key].data * decay + msd[key].data * (1 - decay))

# ------------------------------
#  Training Loop
# ------------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs, loc_out, num_out = model(inputs)

        # Calculate segmentation loss with smoothing
        one_hot_labels = F.one_hot(labels, num_classes=3).permute(0, 4, 1, 2, 3).float()
        smooth = 0.15
        preds = F.softmax(outputs, dim=1)
        loss_seg = -( (1 - smooth) * one_hot_labels * torch.log(preds + 1e-7) + 
                      smooth * (1/3) * torch.log(preds + 1e-7) ).mean()

        loss_seg.backward()
        optimizer.step()

        # EMA update
        update_ema(model, ema_model, ema_decay)

        running_loss += loss_seg.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f}")

print("Training Finished ✅")
