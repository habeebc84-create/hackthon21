"""
tests/test_preprocess_and_masks.py
==================================
Unit tests for Stage 2 SAR Preprocessing and Physical Terrain Masks.
Tests Refined Lee filter, slope computation, HAND calculation, and terrain exclusion.
"""
import numpy as np
import pytest

from pipeline.preprocess import refined_lee_filter, compute_sar_features, linear_to_db
from pipeline.masks import (
    compute_slope_degrees,
    compute_hand_approx,
    compute_distance_to_river_m,
    build_terrain_masks,
)


def test_refined_lee_filter_reduces_variance():
    # Synthetic flat area with speckle noise
    np.random.seed(42)
    clean = np.ones((50, 50), dtype=np.float32) * 10.0
    speckle = np.random.exponential(scale=1.0, size=(50, 50)).astype(np.float32)
    noisy = clean * speckle

    filtered = refined_lee_filter(noisy, window_size=5)
    # Variance of filtered image should be substantially lower than noisy
    assert np.var(filtered) < np.var(noisy)
    # Mean should be preserved
    assert np.isclose(np.mean(filtered), np.mean(noisy), rtol=0.2)


def test_compute_sar_features():
    pre_vv = np.random.uniform(-15, -5, size=(40, 40)).astype(np.float32)
    pre_vh = np.random.uniform(-25, -15, size=(40, 40)).astype(np.float32)
    post_vv = pre_vv.copy()
    post_vh = pre_vh.copy()

    # Inject water drop in center
    post_vv[15:25, 15:25] -= 8.0  # -8 dB drop characteristic of specular water reflection

    features = compute_sar_features(pre_vv, pre_vh, post_vv, post_vh, filter_speckle=True)
    assert "diff_vv" in features
    assert "log_ratio_vv" in features
    # Check that difference in center is strongly negative
    assert np.mean(features["diff_vv"][15:25, 15:25]) < -5.0


def test_terrain_masks_steep_slope_exclusion():
    # Synthetic V-shaped valley DEM
    dem = np.zeros((100, 100), dtype=np.float32)
    # Center column is river channel at elevation 1000m
    # Elevation rises steeply away from column 50 to 2500m
    for col in range(100):
        dist_from_river = abs(col - 50)
        dem[:, col] = 1000.0 + dist_from_river * 30.0  # 30m rise per 30m pixel = ~45 deg slope

    river_mask = np.zeros((100, 100), dtype=bool)
    river_mask[:, 50] = True

    config = {
        "preprocessing": {
            "masks": {
                "max_slope_deg": 18.0,
                "max_hand_m": 25.0,
                "max_distance_to_channel_m": 500.0,
            }
        }
    }

    masks = build_terrain_masks(dem, river_mask, config, cell_size_m=30.0)
    valid_flood = masks["valid_flood_mask"]

    # Valley floor (col 50) should be valid
    assert valid_flood[:, 50].all()
    # Mountain slopes (>18 deg, HAND > 25m) should be completely excluded
    assert not valid_flood[:, 0].any()
    assert not valid_flood[:, 99].any()
