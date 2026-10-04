"""
pipeline/fuse.py
================
STAGE 5 - OPTICAL & RADAR MULTI-SENSOR FUSION
Fuses Sentinel-1 radar detections with Sentinel-2 optical indices (MNDWI, NDVI, BSI)
on cloud-free pixels (with dilated cloud/shadow masks).
Where S1 radar and S2 optical strongly disagree, labels pixels as 'uncertain'.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.ndimage import binary_dilation

logger = logging.getLogger("spacetrack.fuse")


def compute_mndwi(b03_green: np.ndarray, b11_swir: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Modified Normalized Difference Water Index (MNDWI):
      MNDWI = (Green - SWIR) / (Green + SWIR)
    Distinguishes open water from built-up features and bare soil.
    """
    return ((b03_green - b11_swir) / (b03_green + b11_swir + epsilon)).astype(np.float32)


def dilate_cloud_mask(cloud_shadow_mask: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Dilate cloud and shadow mask to eliminate cloud boundary false detections."""
    return binary_dilation(cloud_shadow_mask, iterations=iterations)


def fuse_s1_s2(
    s1_flood_mask: np.ndarray,
    s1_flood_prob: np.ndarray,
    s2_bands_post: Optional[Dict[str, np.ndarray]],
    s2_cloud_shadow_mask: Optional[np.ndarray],
    mndwi_water_thresh: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse S1 radar and S2 optical observations.
    Returns:
      fused_flood_mask: (H, W) bool
      uncertain_pixels_mask: (H, W) bool (pixels where sensors disagree or clouds partially occlude)
    """
    if s2_bands_post is None or s2_cloud_shadow_mask is None:
        # S2 unavailable (cloudy / missing) -> rely solely on S1 radar
        return s1_flood_mask.copy(), np.zeros_like(s1_flood_mask, dtype=bool)

    dilated_clouds = dilate_cloud_mask(s2_cloud_shadow_mask)
    clear_pixels = ~dilated_clouds

    # Compute S2 optical water index
    mndwi = compute_mndwi(s2_bands_post["B03"], s2_bands_post["B11"])
    s2_water_confirmed = (mndwi >= mndwi_water_thresh) & clear_pixels

    # Disagreement analysis:
    # 1. S1 says flood, but clear S2 says dry (MNDWI < -0.1) -> potential false radar reflection
    s1_pos_s2_neg = s1_flood_mask & clear_pixels & (mndwi < -0.1)

    # 2. S2 says clear water, but S1 didn't detect -> potential SAR shadow/corner effect
    s2_pos_s1_neg = s2_water_confirmed & (~s1_flood_mask) & (s1_flood_prob < 0.25)

    uncertain_pixels = s1_pos_s2_neg | s2_pos_s1_neg

    # Fused decision:
    # Where clear, both agree OR strong S2 confirms
    fused_mask = s1_flood_mask.copy()
    # Remove obvious dry pixels confirmed by clear optical
    fused_mask[s1_pos_s2_neg] = False
    # Add clear optical confirmed water if in valley
    fused_mask[s2_water_confirmed & (s1_flood_prob >= 0.3)] = True

    return fused_mask, uncertain_pixels
