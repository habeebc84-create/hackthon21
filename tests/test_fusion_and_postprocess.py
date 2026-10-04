"""
tests/test_fusion_and_postprocess.py
====================================
Unit tests for Stage 5 Multi-Sensor Fusion & Spatial Post-Processing.
Tests MNDWI computation, cloud mask dilation, sensor disagreement flagging,
small object removal, and river connectivity filtering.
"""
import numpy as np
import pytest

from pipeline.fuse import compute_mndwi, dilate_cloud_mask, fuse_s1_s2
from pipeline.postprocess import (
    remove_small_objects,
    filter_river_connectivity,
    postprocess_hazard_masks,
)


def test_fuse_s1_s2_disagreement():
    h, w = 25, 25
    s1_mask = np.zeros((h, w), dtype=bool)
    s1_prob = np.zeros((h, w), dtype=np.float32)

    # Pixel (10, 10): S1 detected flood
    s1_mask[10, 10] = True
    s1_prob[10, 10] = 0.75

    # Optical bands: high Green, low SWIR -> positive MNDWI (water) at (10, 10)
    s2_bands = {
        "B03": np.ones((h, w), dtype=np.float32) * 0.1,
        "B11": np.ones((h, w), dtype=np.float32) * 0.2,
    }
    s2_bands["B03"][10, 10] = 0.35
    s2_bands["B11"][10, 10] = 0.05

    # Pixel (5, 5): S1 detected flood, but optical is completely dry (SWIR >> Green)
    s1_mask[5, 5] = True
    s1_prob[5, 5] = 0.65
    s2_bands["B03"][5, 5] = 0.05
    s2_bands["B11"][5, 5] = 0.45

    cloud_mask = np.zeros((h, w), dtype=bool)

    fused, uncertain = fuse_s1_s2(s1_mask, s1_prob, s2_bands, cloud_mask)

    # At (10, 10): both agree -> fused true, not uncertain
    assert fused[10, 10] == True
    # At (5, 5): disagreement -> marked uncertain and rejected from primary mask
    assert uncertain[5, 5] == True
    assert fused[5, 5] == False


def test_postprocess_filters():
    h, w = 50, 50
    flood_raw = np.zeros((h, w), dtype=bool)

    # Isolated 2-pixel noise speck (at top left)
    flood_raw[2:4, 2] = True

    # Connected flood blob (10x10) adjacent to river
    flood_raw[20:30, 20:30] = True

    # Disconnected blob on mountain (10x10)
    flood_raw[35:45, 35:45] = True

    river = np.zeros((h, w), dtype=bool)
    river[:, 25] = True  # passes through column 25

    terrain = {
        "hand_m": np.zeros((h, w), dtype=np.float32),
    }
    config = {
        "postprocessing": {
            "min_object_size_px": 5,
            "fill_small_holes": True,
            "require_river_connectivity": True,
        },
        "preprocessing": {"masks": {"max_hand_m": 30.0}},
    }

    clean_flood, _ = postprocess_hazard_masks(
        flood_raw, np.zeros_like(flood_raw), river, terrain, config
    )

    # 2-pixel speck should be removed
    assert not clean_flood[2:4, 2].any()
    # Blob touching river (col 25) should be retained
    assert clean_flood[25, 25] == True
    # Disconnected blob (col 35..45) should be removed by river connectivity filter
    assert not clean_flood[35:45, 35:45].any()
