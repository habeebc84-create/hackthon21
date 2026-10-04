"""
tests/test_detect.py
====================
Unit tests for Stage 3 Flood Detection and Model Inference.
Tests model forward pass, DiceFocal loss, threshold-only baseline,
and detection pipeline output dimensions and values.
"""
import numpy as np
import pytest
import torch

from model.train import FloodUNet, DiceFocalLoss
from pipeline.detect import detect_flood, run_threshold_baseline


def test_unet_forward_pass_and_shape():
    model = FloodUNet(in_channels=9, num_classes=3)
    x = torch.randn(2, 9, 64, 64)
    out = model(x)
    assert out.shape == (2, 3, 64, 64)


def test_dice_focal_loss():
    loss_fn = DiceFocalLoss()
    logits = torch.randn(2, 3, 32, 32, requires_grad=True)
    targets = torch.randint(0, 3, (2, 32, 32))
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None


def test_detect_flood_pipeline():
    h, w = 32, 32
    sar_features = {
        "pre_vv": np.ones((h, w), dtype=np.float32) * -12.0,
        "pre_vh": np.ones((h, w), dtype=np.float32) * -18.0,
        "post_vv": np.ones((h, w), dtype=np.float32) * -12.0,
        "post_vh": np.ones((h, w), dtype=np.float32) * -18.0,
        "diff_vv": np.zeros((h, w), dtype=np.float32),
        "diff_vh": np.zeros((h, w), dtype=np.float32),
    }
    terrain_masks = {
        "slope_deg": np.ones((h, w), dtype=np.float32) * 5.0,
        "hand_m": np.ones((h, w), dtype=np.float32) * 10.0,
        "distance_to_river_m": np.ones((h, w), dtype=np.float32) * 100.0,
        "valid_flood_mask": np.ones((h, w), dtype=bool),
    }
    config = {
        "detection": {
            "thresholds": {"conservative": 0.65, "liberal": 0.40},
            "tta": {"enabled": True},
        }
    }

    # Test baseline fallback
    prob, uncert, cons, lib, perm, meta = detect_flood(
        sar_features, terrain_masks, config, use_baseline_fallback=True
    )
    assert prob.shape == (h, w)
    assert uncert.shape == (h, w)
    assert meta.fallback_mode is True

    # Test neural model inference
    prob_m, uncert_m, cons_m, lib_m, perm_m, meta_m = detect_flood(
        sar_features, terrain_masks, config, use_baseline_fallback=False
    )
    assert prob_m.shape == (h, w)
    assert meta_m.fallback_mode is False
    assert 0.0 <= prob_m.max() <= 1.0
