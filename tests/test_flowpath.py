"""
tests/test_flowpath.py
======================
Unit tests for Stage 8 Downstream Flow Path Tracing.
Tests steepest descent D8 routing down an inclined valley,
settlement proximity and ordering, and path flood overlap calculation.
"""
import numpy as np
import pytest

from pipeline.flowpath import trace_d8_flow_path, compute_flow_path


def test_trace_d8_flow_path():
    # Inclined plane: row 0 is high elevation (1000m), row 20 is low (800m)
    rows, cols = 20, 20
    dem = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        dem[r, :] = 1000.0 - r * 10.0

    path = trace_d8_flow_path(dem, (0, 10))
    # Steepest descent must move straight down the rows
    assert len(path) == 20
    for idx, (pr, pc) in enumerate(path):
        assert pr == idx
        assert pc == 10


def test_compute_flow_path_full():
    rows, cols = 20, 20
    dem = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        dem[r, :] = 1000.0 - r * 10.0

    flood_mask = np.zeros((rows, cols), dtype=bool)
    flood_mask[10:15, 10] = True  # 5 pixels flooded along the path

    transform = (85.0, 28.2, 0.01, 0.01, rows, cols)

    settlements = [
        {"name": "Lower Village", "lon": 85.10, "lat": 28.05},  # r=15, c=10
        {"name": "Upper Village", "lon": 85.10, "lat": 28.15},  # r=5, c=10
    ]

    res = compute_flow_path(
        source_lon=85.10,
        source_lat=28.20,  # r=0, c=10
        dem=dem,
        flood_mask=flood_mask,
        settlements=settlements,
        transform=transform,
    )

    assert res.path_length_km > 0.0
    assert 0.0 < res.path_flood_overlap_fraction <= 1.0
    assert len(res.settlements) == 2
    # Upper Village must come before Lower Village in downstream sequence
    assert res.settlements[0].name == "Upper Village"
    assert res.settlements[1].name == "Lower Village"
    assert res.settlements[0].distance_from_source_km < res.settlements[1].distance_from_source_km
