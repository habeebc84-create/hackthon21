"""
pipeline/postprocess.py
=======================
STAGE 5 - POST-PROCESSING & SPATIAL MORPHOLOGY
Cleans fused flood & debris rasters:
  1. Small object removal (< 8 pixels / noise specks)
  2. Small hole filling
  3. Drainage network connectivity filter (keeps blobs connected to river)
  4. Ridge & high HAND rejection
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, label

logger = logging.getLogger("spacetrack.postprocess")


def remove_small_objects(mask: np.ndarray, min_size_px: int = 8) -> np.ndarray:
    """Remove disconnected foreground components smaller than min_size_px."""
    labeled, num_features = label(mask)
    if num_features == 0:
        return mask.copy()

    component_sizes = np.bincount(labeled.ravel())
    too_small = component_sizes < min_size_px
    too_small_mask = too_small[labeled]
    cleaned = mask.copy()
    cleaned[too_small_mask] = False
    return cleaned


def filter_river_connectivity(
    flood_mask: np.ndarray,
    river_mask: np.ndarray,
    max_dilation_tolerance: int = 2,
) -> np.ndarray:
    """
    Retain only flood blobs that physically connect or touch the river network.
    Rejects isolated pooling artifacts on mountaintops.
    """
    labeled, num_features = label(flood_mask)
    if num_features == 0 or not np.any(river_mask):
        return flood_mask.copy()

    # Slightly dilate river mask by tolerance to catch adjacent water pixels
    dilated_river = binary_dilation(river_mask, iterations=max_dilation_tolerance)

    # Find which labels intersect the river
    intersecting_labels = np.unique(labeled[dilated_river])
    intersecting_labels = intersecting_labels[intersecting_labels > 0]

    # Create mask of only connected components
    connected_mask = np.isin(labeled, intersecting_labels)
    return connected_mask


def postprocess_hazard_masks(
    flood_mask_raw: np.ndarray,
    debris_mask_raw: np.ndarray,
    river_mask: np.ndarray,
    terrain_masks: Dict[str, np.ndarray],
    config: Dict[str, any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute full spatial cleaning and hydrological filtering on flood and debris masks.
    """
    post_cfg = config.get("postprocessing", {})
    min_size_px = post_cfg.get("min_object_size_px", 8)
    fill_holes = post_cfg.get("fill_small_holes", True)
    enforce_connectivity = post_cfg.get("require_river_connectivity", True)
    max_hand_cutoff = config.get("preprocessing", {}).get("masks", {}).get("max_hand_m", 25.0)

    # 1. Fill holes
    flood_clean = binary_fill_holes(flood_mask_raw) if fill_holes else flood_mask_raw.copy()
    debris_clean = binary_fill_holes(debris_mask_raw) if fill_holes else debris_mask_raw.copy()

    # 2. Remove noise specks / small objects
    flood_clean = remove_small_objects(flood_clean, min_size_px=min_size_px)
    debris_clean = remove_small_objects(debris_clean, min_size_px=min_size_px)

    # 3. Enforce river connectivity (optional but critical in steep gorges)
    if enforce_connectivity and np.any(river_mask):
        flood_clean = filter_river_connectivity(flood_clean, river_mask)

    # 4. Strict physical elevation constraint: reject high HAND
    hand = terrain_masks.get("hand_m")
    if hand is not None:
        flood_clean = flood_clean & (hand <= max_hand_cutoff)
        debris_clean = debris_clean & (hand <= max_hand_cutoff * 1.5)

    return flood_clean.astype(bool), debris_clean.astype(bool)
