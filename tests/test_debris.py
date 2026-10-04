"""
tests/test_debris.py
====================
Unit tests for Stage 4 Multi-Sensor Heuristic Debris Detection.
Tests Bare Soil Index calculation, multi-signal voting, and corridor restriction.
"""
import numpy as np
import pytest

from pipeline.debris import compute_s2_bsi, detect_debris


def test_compute_s2_bsi():
    h, w = 20, 20
    # Mineral bare soil has high SWIR (B11) and Red (B04), low NIR (B08) and Blue (B02)
    b02 = np.ones((h, w), dtype=np.float32) * 0.1
    b04 = np.ones((h, w), dtype=np.float32) * 0.35
    b08 = np.ones((h, w), dtype=np.float32) * 0.15
    b11 = np.ones((h, w), dtype=np.float32) * 0.45

    bsi = compute_s2_bsi(b02, b04, b08, b11)
    # BSI should be strongly positive for bare rock/soil
    assert np.all(bsi > 0.4)


def test_detect_debris_two_of_three_agreement():
    h, w = 30, 30
    sar_features = {
        "diff_vh": np.zeros((h, w), dtype=np.float32),
    }
    terrain_masks = {
        "distance_to_river_m": np.ones((h, w), dtype=np.float32) * 100.0,
        "slope_deg": np.ones((h, w), dtype=np.float32) * 10.0,
    }
    config = {
        "debris": {
            "vh_roughness_threshold_db": 2.0,
            "bsi_change_threshold": 0.1,
            "corridor_buffer_m": 500.0,
        }
    }

    # Pixel (15, 15) has:
    # 1. S1 VH roughness increase (+3.0 dB)
    sar_features["diff_vh"][15, 15] = 3.0

    # 2. S2 BSI change
    s2_pre = {
        "B02": np.ones((h, w), dtype=np.float32) * 0.2,
        "B04": np.ones((h, w), dtype=np.float32) * 0.2,
        "B08": np.ones((h, w), dtype=np.float32) * 0.4,  # dense vegetation pre
        "B11": np.ones((h, w), dtype=np.float32) * 0.2,
    }
    s2_post = {
        "B02": np.ones((h, w), dtype=np.float32) * 0.1,
        "B04": np.ones((h, w), dtype=np.float32) * 0.4,
        "B08": np.ones((h, w), dtype=np.float32) * 0.15,
        "B11": np.ones((h, w), dtype=np.float32) * 0.5,  # bare debris post
    }
    s2_clear = np.ones((h, w), dtype=bool)

    deb_cons, deb_lib, meta = detect_debris(
        sar_features,
        terrain_masks,
        config,
        s2_bands_pre=s2_pre,
        s2_bands_post=s2_post,
        s2_clear_mask=s2_clear,
    )

    assert meta["debris_signals_available"] == 2
    assert meta["debris_s1_signal_used"] is True
    assert meta["debris_s2_bsi_used"] is True
    # At (15, 15), both signals agree -> positive in both conservative and liberal
    assert deb_cons[15, 15] == True
    assert deb_lib[15, 15] == True

    # At (0, 0), no signals -> False
    assert deb_cons[0, 0] == False
