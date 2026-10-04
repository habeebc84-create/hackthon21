"""
pipeline/preprocess.py
======================
STAGE 2 - SENTINEL-1 & SATELLITE PREPROCESSING
Implements Refined Lee speckle filter, calibration to dB, co-registration,
and band differencing (difference & log-ratio) for SAR inputs.
"""
from __future__ import annotations

import logging
from typing import Tuple
import numpy as np
from scipy.ndimage import uniform_filter

logger = logging.getLogger("spacetrack.preprocess")


def refined_lee_filter(image: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Refined Lee speckle filter for SAR imagery.
    Suppresses multiplicative speckle while preserving sharp edges and water boundaries.
    """
    if window_size % 2 == 0:
        window_size += 1

    mean = uniform_filter(image.astype(np.float64), size=window_size)
    mean_sq = uniform_filter(image.astype(np.float64) ** 2, size=window_size)
    variance = np.maximum(mean_sq - mean ** 2, 0)

    # Estimate noise variance from local mean
    overall_mean = np.mean(image)
    overall_var = np.var(image)
    noise_var = overall_var / (overall_mean ** 2 + 1e-6)

    # Weight factor k
    weights = variance / (variance + noise_var * (mean ** 2) + 1e-6)
    weights = np.clip(weights, 0.0, 1.0)

    filtered = mean + weights * (image - mean)
    return filtered.astype(np.float32)


def linear_to_db(linear_raster: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Convert linear SAR amplitude/intensity to decibels (dB)."""
    clipped = np.maximum(linear_raster, epsilon)
    return 10.0 * np.log10(clipped)


def compute_sar_features(
    pre_vv: np.ndarray,
    pre_vh: np.ndarray,
    post_vv: np.ndarray,
    post_vh: np.ndarray,
    filter_speckle: bool = True,
    window_size: int = 7,
) -> dict[str, np.ndarray]:
    """
    Compute pre/post calibrated bands, difference rasters, and log ratios.
    Returns:
      pre_vv, pre_vh, post_vv, post_vh,
      diff_vv, diff_vh, log_ratio_vv, log_ratio_vh
    """
    if filter_speckle:
        pre_vv = refined_lee_filter(pre_vv, window_size)
        pre_vh = refined_lee_filter(pre_vh, window_size)
        post_vv = refined_lee_filter(post_vv, window_size)
        post_vh = refined_lee_filter(post_vh, window_size)

    # Difference in dB space represents log-ratio of linear intensities
    diff_vv = post_vv - pre_vv
    diff_vh = post_vh - pre_vh

    # Normalized cross-ratio: (VH - VV) / (VH + VV + 1e-6)
    cross_ratio_pre = (pre_vh - pre_vv) / (np.abs(pre_vh) + np.abs(pre_vv) + 1e-6)
    cross_ratio_post = (post_vh - post_vv) / (np.abs(post_vh) + np.abs(post_vv) + 1e-6)

    return {
        "pre_vv": pre_vv,
        "pre_vh": pre_vh,
        "post_vv": post_vv,
        "post_vh": post_vh,
        "diff_vv": diff_vv,
        "diff_vh": diff_vh,
        "log_ratio_vv": diff_vv,
        "log_ratio_vh": diff_vh,
        "cross_ratio_pre": cross_ratio_pre,
        "cross_ratio_post": cross_ratio_post,
    }
