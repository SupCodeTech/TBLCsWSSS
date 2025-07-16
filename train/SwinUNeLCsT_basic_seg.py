
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, num_classes, epsilon=0.15):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, preds, targets):
        """
        preds: [B, K, H, W, D] - raw logits
        targets: [B, H, W, D] - ground truth labels (long)
        """
        log_probs = F.log_softmax(preds, dim=1)  # Apply softmax on class dim
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        smoothed_labels = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes
        loss = -(smoothed_labels * log_probs).sum(dim=1).mean()
        return loss

class EMAPrediction:
    def __init__(self, shape, device, delta=0.1):
        """
        shape: (B, K, H, W, D) - Shape of the prediction
        delta: EMA smoothing factor
        """
        self.ema_prediction = torch.zeros(shape, device=device)
        self.delta = delta
        self.iteration = 0

    def update(self, current_pred):
        """
        current_pred: [B, K, H, W, D] - softmax probabilities
        """
        if self.iteration == 0:
            self.ema_prediction = current_pred.detach()
        else:
            self.ema_prediction = self.delta * current_pred.detach() + (1 - self.delta) * self.ema_prediction
        self.iteration += 1

    def get_ema(self):
        return self.ema_prediction.clone()

# Example usage inside a training loop:
num_classes = 2
criterion = LabelSmoothingCrossEntropy(num_classes=num_classes, epsilon=0.15)
ema_tracker = None  # Will initialize with first batch shape
delta = 0.1  # EMA smoothing factor

for epoch in range(total_epochs):
    for batch_idx, (images, pseudo_labels) in enumerate(dataloader):
        images = images.to(device)  # [B, C, H, W, D]
        pseudo_labels = pseudo_labels.to(device)  # [B, H, W, D]

        optimizer.zero_grad()
        logits = model(images)  # [B, K, H, W, D]

        loss = criterion(logits, pseudo_labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            softmax_probs = F.softmax(logits, dim=1)
            if ema_tracker is None:
                ema_tracker = EMAPrediction(softmax_probs.shape, device, delta)
            ema_tracker.update(softmax_probs)

        if batch_idx % gamma == 0:  # e.g., gamma = 10
            ema_output = ema_tracker.get_ema()
            # You can use ema_output for evaluation or pseudo-label refinement
