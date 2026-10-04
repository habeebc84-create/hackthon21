"""
pipeline/debris.py
==================
STAGE 4 - HEURISTIC MULTI-SENSOR DEBRIS DETECTION
Note: Neither Kuro Siwo nor Sen1Floods11 contains a debris class.
This module implements a physical 3-signal voting heuristic inside a buffered river corridor:
  1. Sentinel-1 VH roughness increase (post VH - pre VH > threshold)
  2. InSAR coherence loss / decorrelation proxy
  3. Sentinel-2 Bare Soil Index (BSI) change on clear pixels
At least 2 of available signals must agree.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger("spacetrack.debris")


def compute_s2_bsi(
    b02_blue: np.ndarray,
    b04_red: np.ndarray,
    b08_nir: np.ndarray,
    b11_swir: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Bare Soil Index (BSI):
      BSI = [ (SWIR + Red) - (NIR + Blue) ] / [ (SWIR + Red) + (NIR + Blue) ]
    Highlights bare rock, debris deposits, sediment, and stripped soil.
    """
    num = (b11_swir + b04_red) - (b08_nir + b02_blue)
    denom = (b11_swir + b04_red) + (b08_nir + b02_blue) + epsilon
    return (num / denom).astype(np.float32)


def detect_debris(
    sar_features: Dict[str, np.ndarray],
    terrain_masks: Dict[str, np.ndarray],
    config: Dict[str, any],
    s2_bands_pre: Optional[Dict[str, np.ndarray]] = None,
    s2_bands_post: Optional[Dict[str, np.ndarray]] = None,
    s2_clear_mask: Optional[np.ndarray] = None,
    coherence_loss_raster: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
    """
    Execute Stage 4 Debris Detection.
    Returns:
      debris_conservative: (H, W) bool binary mask
      debris_liberal: (H, W) bool binary mask
      debris_meta: Dictionary of signal contributions for results.json
    """
    deb_cfg = config.get("debris", {})
    vh_increase_thresh_db = deb_cfg.get("vh_roughness_threshold_db", 2.2)
    bsi_delta_thresh = deb_cfg.get("bsi_change_threshold", 0.15)
    max_dist_corridor_m = deb_cfg.get("corridor_buffer_m", 450.0)

    # 1. Geographic corridor restriction: debris occurs along valley floors and fan deposits
    dist_river = terrain_masks["distance_to_river_m"]
    corridor_mask = (dist_river <= max_dist_corridor_m) & (terrain_masks["slope_deg"] <= 35.0)

    votes = np.zeros_like(corridor_mask, dtype=np.int32)
    signals_available = 0

    # Signal 1: Sentinel-1 VH roughness increase (coarse gravel, boulders, mud debris)
    diff_vh = sar_features["diff_vh"]
    s1_debris_signal = (diff_vh >= vh_increase_thresh_db) & corridor_mask
    votes += s1_debris_signal.astype(np.int32)
    s1_used = True
    signals_available += 1

    # Signal 2: InSAR coherence loss / decorrelation
    coherence_used = False
    if coherence_loss_raster is not None:
        coherence_loss_thresh = deb_cfg.get("coherence_loss_threshold", 0.35)
        coherence_signal = (coherence_loss_raster >= coherence_loss_thresh) & corridor_mask
        votes += coherence_signal.astype(np.int32)
        coherence_used = True
        signals_available += 1

    # Signal 3: Sentinel-2 Bare Soil Index increase
    s2_bsi_used = False
    if s2_bands_pre and s2_bands_post and s2_clear_mask is not None:
        bsi_pre = compute_s2_bsi(
            s2_bands_pre["B02"], s2_bands_pre["B04"], s2_bands_pre["B08"], s2_bands_pre["B11"]
        )
        bsi_post = compute_s2_bsi(
            s2_bands_post["B02"], s2_bands_post["B04"], s2_bands_post["B08"], s2_bands_post["B11"]
        )
        bsi_delta = bsi_post - bsi_pre
        s2_signal = (bsi_delta >= bsi_delta_thresh) & s2_clear_mask & corridor_mask
        votes += s2_signal.astype(np.int32)
        s2_bsi_used = True
        signals_available += 1

    # Multi-signal agreement rule:
    # If 2 or 3 signals available: require >= 2 votes
    # If only 1 signal available: require >= 1 vote (with cautionary warning)
    min_votes_conservative = 2 if signals_available >= 2 else 1
    min_votes_liberal = 1

    debris_conservative = (votes >= min_votes_conservative) & corridor_mask
    debris_liberal = (votes >= min_votes_liberal) & corridor_mask

    meta = {
        "debris_s1_signal_used": s1_used,
        "debris_coherence_used": coherence_used,
        "debris_s2_bsi_used": s2_bsi_used,
        "debris_signals_available": signals_available,
        "heuristic_description": (
            "Multi-sensor heuristic combining S1 VH roughness, InSAR coherence loss, and S2 BSI change. "
            "Reported honestly as an uncalibrated heuristic since no benchmark dataset labels debris."
        ),
    }

    return debris_conservative, debris_liberal, meta
