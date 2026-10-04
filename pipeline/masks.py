"""
pipeline/masks.py
=================
STAGE 2 - TERRAIN & PHYSICAL EXCLUSION MASKS
Computes slope, HAND, distance-to-river channel, and layover/shadow masks.
These physical masks filter false positives on steep Himalayan mountainsides
where radar shadow and steep topography mimic water backscatter.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger("spacetrack.masks")


def compute_slope_degrees(dem: np.ndarray, cell_size_m: float = 30.0) -> np.ndarray:
    """Compute slope in degrees from a DEM raster using centered differences."""
    dy, dx = np.gradient(dem, cell_size_m, cell_size_m)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    return np.degrees(slope_rad).astype(np.float32)


def compute_hand_approx(dem: np.ndarray, river_mask: np.ndarray) -> np.ndarray:
    """
    Height Above Nearest Drainage (HAND) calculation.
    Computes the elevation of each pixel minus the elevation of the nearest drainage/channel pixel.
    """
    if not np.any(river_mask):
        # If no river detected, fallback to relative elevation above local minimum
        return (dem - np.min(dem)).astype(np.float32)

    # Nearest river pixel index via distance transform
    _, indices = distance_transform_edt(~river_mask, return_indices=True)
    nearest_river_elevation = dem[indices[0], indices[1]]
    hand = np.maximum(dem - nearest_river_elevation, 0.0)
    return hand.astype(np.float32)


def compute_distance_to_river_m(river_mask: np.ndarray, cell_size_m: float = 30.0) -> np.ndarray:
    """Calculate Euclidean distance in meters from each pixel to the nearest river channel."""
    if not np.any(river_mask):
        return np.zeros_like(river_mask, dtype=np.float32)
    dist_cells = distance_transform_edt(~river_mask)
    return (dist_cells * cell_size_m).astype(np.float32)


def compute_layover_shadow_mask(
    dem: np.ndarray,
    slope_deg: np.ndarray,
    look_angle_deg: float = 35.0,
    slope_shadow_thresh: float = 40.0,
) -> np.ndarray:
    """
    Identify potential radar layover and shadow zones based on steep slopes facing toward/away from sensor.
    """
    # Slopes steeper than radar look angle or critical shadow angles
    shadow_mask = slope_deg > slope_shadow_thresh
    return shadow_mask.astype(bool)


def build_terrain_masks(
    dem: np.ndarray,
    river_channel_mask: np.ndarray,
    config: Dict[str, any],
    cell_size_m: float = 30.0,
) -> Dict[str, np.ndarray]:
    """
    Generate all physical constraints and the composite valid flood search mask.
    """
    slope_cutoff = config.get("preprocessing", {}).get("masks", {}).get("max_slope_deg", 18.0)
    hand_cutoff = config.get("preprocessing", {}).get("masks", {}).get("max_hand_m", 25.0)
    max_river_dist_m = config.get("preprocessing", {}).get("masks", {}).get("max_distance_to_channel_m", 1500.0)

    slope = compute_slope_degrees(dem, cell_size_m=cell_size_m)
    hand = compute_hand_approx(dem, river_channel_mask)
    dist_river = compute_distance_to_river_m(river_channel_mask, cell_size_m=cell_size_m)
    layover_shadow = compute_layover_shadow_mask(dem, slope)

    # Valid flood search criteria:
    # 1. Slope <= slope_cutoff (water cannot pool on 25-degree cliff walls)
    # 2. HAND <= hand_cutoff (flood water remains within valley floodplains)
    # 3. Distance to river <= max_river_dist_m
    # 4. Not radar shadow / layover artifact
    valid_flood_mask = (
        (slope <= slope_cutoff) &
        (hand <= hand_cutoff) &
        (dist_river <= max_river_dist_m) &
        (~layover_shadow)
    )

    return {
        "slope_deg": slope,
        "hand_m": hand,
        "distance_to_river_m": dist_river,
        "layover_shadow": layover_shadow,
        "valid_flood_mask": valid_flood_mask,
    }
