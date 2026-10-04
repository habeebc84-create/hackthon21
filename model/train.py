"""
model/train.py
==============
STAGE 3 - MODEL TRAINING & LOSSES
Trains U-Net segmentation network with 9-channel input for flood detection.
Uses combined Dice + Focal Loss to handle extreme class imbalance.
Saves model weights to model/weights/.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.dataset import FloodDataset, create_synthetic_flood_tiles

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


class FloodUNet(nn.Module):
    """
    Lightweight, robust 9-channel U-Net for satellite radar flood mapping.
    Compatible across CPU and GPU environments with identical weight format.
    """

    def __init__(self, in_channels: int = 9, num_classes: int = 3):
        super().__init__()
        # Encoder
        self.enc1 = self._block(in_channels, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self._block(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._block(64, 32)

        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def _block(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


class DiceFocalLoss(nn.Module):
    """Combined Dice and Focal Loss for imbalanced multiclass segmentation."""

    def __init__(self, gamma: float = 2.0, smooth: float = 1e-5):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Focal term
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_loss = -((1.0 - pt) ** self.gamma) * torch.log(pt.clamp(min=1e-7))
        focal_loss = focal_loss.mean()

        # Dice term per class
        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = (probs + targets_one_hot).sum(dims)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return focal_loss + dice_loss


def calculate_iou(preds: torch.Tensor, targets: torch.Tensor, target_class: int = 2) -> float:
    """Calculate IoU for a target class (class 2 = flood)."""
    pred_mask = preds == target_class
    target_mask = targets == target_class
    intersection = (pred_mask & target_mask).sum().item()
    union = (pred_mask | target_mask).sum().item()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def train_flood_model(
    epochs: int = 3,
    batch_size: int = 2,
    device: str = "cpu",
    save_name: str = "unet_flood_v1.pt",
) -> Tuple[FloodUNet, float]:
    """Train or fine-tune flood segmentation model."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = WEIGHTS_DIR / save_name

    # Create synthetic dataset for reproducible training and validation
    all_tiles = create_synthetic_flood_tiles(num_tiles=12, tile_size=64)
    train_tiles = all_tiles[:8]
    val_tiles = all_tiles[8:]

    train_ds = FloodDataset(train_tiles, is_training=True)
    val_ds = FloodDataset(val_tiles, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = FloodUNet(in_channels=9, num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = DiceFocalLoss()

    best_val_iou = 0.0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_ious = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                iou = calculate_iou(preds, y, target_class=2)
                val_ious.append(iou)

        mean_iou = float(sum(val_ious) / len(val_ious))
        if mean_iou >= best_val_iou:
            best_val_iou = mean_iou
            torch.save(model.state_dict(), save_path)

    return model, best_val_iou


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {dev}")
    m, iou = train_flood_model(epochs=2, device=dev)
    print(f"Trained model checkpoint saved with validation IoU: {iou:.4f}")
