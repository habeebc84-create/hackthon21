"""
tests/test_damage.py
====================
Unit tests for Stage 6 Infrastructure Damage Assessment.
Tests overlap classification, building damage calculations, road length calculations,
and bridge susceptibility heuristics.
"""
import numpy as np
import pytest

from pipeline.schema import DamageClass
from pipeline.damage import classify_overlap, assess_damage


def test_classify_overlap():
    assert classify_overlap(0.85) == DamageClass.LIKELY_HIT
    assert classify_overlap(0.20) == DamageClass.POSSIBLY_HIT
    assert classify_overlap(0.01, is_bridge=False) == DamageClass.NOT_HIT
    # Bridges with minor touch default to possibly hit
    assert classify_overlap(0.01, is_bridge=True) == DamageClass.POSSIBLY_HIT


def test_assess_damage_counts():
    h, w = 10, 10
    hazard_cons = np.zeros((h, w), dtype=bool)
    hazard_lib = np.zeros((h, w), dtype=bool)

    # Inundate row 5
    hazard_cons[5, :] = True
    hazard_lib[4:6, :] = True

    # Affine transform covering lon 85.0..85.1, lat 27.9..28.0
    transform = (85.0, 28.0, 0.01, 0.01, h, w)

    buildings = [
        {"osm_id": 101, "centroid_lon": 85.05, "centroid_lat": 27.95}, # row 5 -> flooded
        {"osm_id": 102, "centroid_lon": 85.05, "centroid_lat": 27.99}, # row 1 -> dry
    ]
    roads = [
        {
            "osm_id": 201,
            "highway_type": "primary",
            "length_m": 500.0,
            "is_bridge": True,
            "points": [(85.05, 27.95)], # hit bridge
        }
    ]

    b_list, r_list, stats = assess_damage(
        buildings, roads, hazard_cons, hazard_lib, transform, {}
    )

    assert stats.buildings_likely_hit_conservative >= 1
    assert stats.bridges_likely_hit >= 1
    assert b_list[1].damage_class_conservative == DamageClass.NOT_HIT
