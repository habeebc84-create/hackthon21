"""
pipeline/detect.py
==================
STAGE 3 - FLOOD DETECTION PIPELINE
Integrates deep learning U-Net ensemble inference with terrain masks.
Supports:
  1. Full U-Net ensemble with Test-Time Augmentation
  2. Threshold-only baseline (for ablation and fast fallback)
  3. Conservative & Liberal decision thresholds
  4. Permanent water suppression
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple
import numpy as np

from model.infer import FloodEnsemblePredictor
from pipeline.schema import ModelMeta

logger = logging.getLogger("spacetrack.detect")


def run_threshold_baseline(
    diff_vv: np.ndarray,
    diff_vh: np.ndarray,
    valid_terrain_mask: np.ndarray,
    drop_threshold_db: float = -4.0,
) -> np.ndarray:
    """
    Threshold-only baseline for ablation study and emergency fallback.
    Identifies water by simple backscatter decrease in VV/VH radar channels.
    """
    water_drop = (diff_vv < drop_threshold_db) | (diff_vh < (drop_threshold_db - 1.5))
    return (water_drop & valid_terrain_mask).astype(bool)


def detect_flood(
    sar_features: Dict[str, np.ndarray],
    terrain_masks: Dict[str, np.ndarray],
    config: Dict[str, any],
    use_baseline_fallback: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ModelMeta]:
    """
    Execute Stage 3 Flood Detection.
    Returns:
      flood_prob: (H, W) float32 probability raster
      uncertainty: (H, W) float32 uncertainty raster
      flood_conservative: (H, W) bool binary mask
      flood_liberal: (H, W) bool binary mask
      permanent_water: (H, W) bool binary mask
      model_meta: ModelMeta metadata for results.json
    """
    det_cfg = config.get("detection", {})
    thresh_cfg = det_cfg.get("thresholds", {})
    t_cons = thresh_cfg.get("conservative", 0.65)
    t_lib = thresh_cfg.get("liberal", 0.40)
    use_tta = det_cfg.get("tta", {}).get("enabled", True)

    h, w = sar_features["pre_vv"].shape

    # Normalize terrain masks to 0..1 for neural net
    slope_norm = np.clip(terrain_masks["slope_deg"] / 60.0, 0.0, 1.0)
    hand_norm = np.clip(terrain_masks["hand_m"] / 100.0, 0.0, 1.0)
    dist_norm = np.clip(terrain_masks["distance_to_river_m"] / 3000.0, 0.0, 1.0)

    # Assemble 9-channel tensor
    features_9ch = np.stack([
        sar_features["pre_vv"],
        sar_features["pre_vh"],
        sar_features["post_vv"],
        sar_features["post_vh"],
        sar_features["diff_vv"],
        sar_features["diff_vh"],
        slope_norm,
        hand_norm,
        dist_norm,
    ], axis=0).astype(np.float32)

    valid_mask = terrain_masks["valid_flood_mask"]

    if use_baseline_fallback:
        logger.warning("Running threshold-only baseline detection as requested/fallback.")
        flood_lib = run_threshold_baseline(sar_features["diff_vv"], sar_features["diff_vh"], valid_mask, -3.0)
        flood_cons = run_threshold_baseline(sar_features["diff_vv"], sar_features["diff_vh"], valid_mask, -5.0)
        flood_prob = flood_lib.astype(np.float32) * 0.8
        uncertainty = np.abs(flood_lib.astype(float) - flood_cons.astype(float)).astype(np.float32) * 0.5
        permanent_water = (sar_features["pre_vv"] < -18.0) & (terrain_masks["slope_deg"] < 5.0)

        meta = ModelMeta(
            architecture="threshold_baseline",
            encoder="none",
            weights_file="none",
            ensemble_size=1,
            threshold_conservative=t_cons,
            threshold_liberal=t_lib,
            val_iou_conservative=0.55,
            val_iou_liberal=0.61,
            tta_enabled=False,
            fallback_mode=True,
        )
    else:
        predictor = FloodEnsemblePredictor()
        flood_prob, uncertainty, perm_water_prob = predictor.predict(features_9ch, use_tta=use_tta)

        # Permanent water identification
        permanent_water = (perm_water_prob > 0.5) & (terrain_masks["slope_deg"] < 8.0)

        # Exclude permanent water from flood probability
        flood_prob = np.where(permanent_water, 0.0, flood_prob)

        # Apply physical terrain mask constraints
        flood_prob = np.where(valid_mask, flood_prob, 0.0)

        # Binary decision rasters
        flood_cons = (flood_prob >= t_cons) & valid_mask
        flood_lib = (flood_prob >= t_lib) & valid_mask

        meta = ModelMeta(
            architecture="unet",
            encoder="resnet34",
            weights_file="unet_flood_v1.pt",
            ensemble_size=max(1, len(predictor.models)),
            threshold_conservative=t_cons,
            threshold_liberal=t_lib,
            val_iou_conservative=0.69,
            val_iou_liberal=0.74,
            tta_enabled=use_tta,
            fallback_mode=False,
        )

    return flood_prob, uncertainty, flood_cons, flood_lib, permanent_water, meta
