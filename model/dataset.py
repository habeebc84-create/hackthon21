"""
model/dataset.py
================
STAGE 3 - DATASET & AUGMENTATIONS
Dataloader for Kuro Siwo & Sen1Floods11 multi-sensor flood datasets.
Input channels (9 channels):
  [pre_vv, pre_vh, post_vv, post_vh, diff_vv, diff_vh, slope, hand, dist_river]
Target classes:
  0: background
  1: permanent water
  2: flood water
Splits data by GEOGRAPHY (holdout by river basin / geographic region) to avoid spatial autocorrelation leakage.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class FloodDataset(Dataset):
    """
    Multi-sensor flood segmentation dataset.
    Loads 9-channel tensor:
      Ch 0: Pre-event S1 VV (dB)
      Ch 1: Pre-event S1 VH (dB)
      Ch 2: Post-event S1 VV (dB)
      Ch 3: Post-event S1 VH (dB)
      Ch 4: S1 Difference VV (post - pre dB)
      Ch 5: S1 Difference VH (post - pre dB)
      Ch 6: Normalized Slope (0-1)
      Ch 7: Normalized HAND (0-1)
      Ch 8: Normalized Distance to River (0-1)
    Label:
      0: Background
      1: Permanent water
      2: Flood water
    """

    def __init__(
        self,
        samples: List[dict],
        transform: Optional[Callable] = None,
        is_training: bool = True,
    ):
        self.samples = samples
        self.transform = transform
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[idx]

        # In-memory arrays or loaded from tile files
        if "inputs" in item and "mask" in item:
            inputs = item["inputs"].copy()
            mask = item["mask"].copy()
        else:
            inputs = np.load(item["input_path"]).astype(np.float32)
            mask = np.load(item["mask_path"]).astype(np.int64)

        if self.is_training:
            # Data augmentations: random flips and rotations
            if np.random.rand() > 0.5:
                inputs = np.flip(inputs, axis=1)
                mask = np.flip(mask, axis=0)
            if np.random.rand() > 0.5:
                inputs = np.flip(inputs, axis=2)
                mask = np.flip(mask, axis=1)
            k = np.random.choice([0, 1, 2, 3])
            if k > 0:
                inputs = np.rot90(inputs, k, axes=(1, 2))
                mask = np.rot90(mask, k, axes=(0, 1))

            # SAR speckle / brightness shift on SAR channels (0..5)
            if np.random.rand() > 0.5:
                noise = np.random.normal(0.0, 0.5, size=inputs[:6].shape).astype(np.float32)
                inputs[:6] = inputs[:6] + noise

        # To contiguous PyTorch tensors
        tensor_inputs = torch.from_numpy(np.ascontiguousarray(inputs)).float()
        tensor_mask = torch.from_numpy(np.ascontiguousarray(mask)).long()

        return tensor_inputs, tensor_mask


def create_synthetic_flood_tiles(
    num_tiles: int = 10,
    tile_size: int = 64,
    random_seed: int = 42,
) -> List[dict]:
    """Generate synthetic 9-channel tiles for testing, CI and validation."""
    np.random.seed(random_seed)
    tiles = []

    for i in range(num_tiles):
        # 9 channels: [pre_vv, pre_vh, post_vv, post_vh, diff_vv, diff_vh, slope, hand, dist_river]
        inputs = np.zeros((9, tile_size, tile_size), dtype=np.float32)

        # Baseline SAR backscatter
        inputs[0] = np.random.normal(-12.0, 3.0, (tile_size, tile_size))  # pre VV
        inputs[1] = np.random.normal(-18.0, 3.0, (tile_size, tile_size))  # pre VH
        inputs[2] = inputs[0].copy()  # post VV
        inputs[3] = inputs[1].copy()  # post VH

        # Terrain channels
        inputs[6] = np.random.uniform(0.0, 0.5, (tile_size, tile_size))   # slope
        inputs[7] = np.random.uniform(0.0, 0.4, (tile_size, tile_size))   # HAND
        inputs[8] = np.random.uniform(0.0, 0.3, (tile_size, tile_size))   # dist river

        mask = np.zeros((tile_size, tile_size), dtype=np.int64)

        # Permanent river down the center
        river_col = tile_size // 2
        inputs[0, :, river_col - 1 : river_col + 2] = -22.0
        inputs[2, :, river_col - 1 : river_col + 2] = -22.0
        mask[:, river_col - 1 : river_col + 2] = 1

        # Flood inundation zone next to river
        flood_box = (slice(10, 40), slice(river_col + 2, river_col + 12))
        inputs[2][flood_box] = -21.0  # drop in post VV
        inputs[3][flood_box] = -26.0  # drop in post VH
        mask[flood_box] = 2

        # Differences
        inputs[4] = inputs[2] - inputs[0]
        inputs[5] = inputs[3] - inputs[1]

        tiles.append({"inputs": inputs, "mask": mask, "region": f"basin_{i % 3}"})

    return tiles
