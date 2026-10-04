"""
tests/test_cutoff.py
====================
Unit tests for Stage 7 Network Isolation & Settlement Cut-off.
Tests Dijkstra routing, UNKNOWN state for pre-event disconnected settlements,
CUT_OFF state upon severed bridge, detour calculation, and critical bridge detection.
"""
import pytest

from pipeline.schema import CutoffState, DamageClass, DamagedRoad
from pipeline.cutoff import analyze_cutoff


def test_analyze_cutoff_workflow():
    # Linear road network:
    # Node A (Hospital) <---> Node B <---(Bridge Seg 501)---> Node C (Village 1)
    # Node D (Village 2, isolated mountain hut with no road)
    roads = [
        {
            "osm_id": 500,
            "length_m": 2000.0,
            "is_bridge": False,
            "points": [(85.00, 28.00), (85.02, 28.00)],
        },
        {
            "osm_id": 501,
            "length_m": 1500.0,
            "is_bridge": True,
            "points": [(85.02, 28.00), (85.04, 28.00)],
        },
    ]

    hospitals = [{"lon": 85.00, "lat": 28.00, "name": "District Hospital"}]

    settlements = [
        {
            "osm_id": 1,
            "name": "Village Near Bridge",
            "lon": 85.04,
            "lat": 28.00,
            "building_count": 45,
            "place_type": "village",
        },
        {
            "osm_id": 2,
            "name": "Isolated Remote Settlement",
            "lon": 85.20,
            "lat": 28.30,  # Far away, no road connects here
            "building_count": 8,
            "place_type": "hamlet",
        },
    ]

    # Sever bridge 501 in both conservative and liberal
    damaged_roads = [
        DamagedRoad(
            osm_id=501,
            highway_type="primary",
            length_m=1500.0,
            damage_class_conservative=DamageClass.LIKELY_HIT,
            damage_class_liberal=DamageClass.LIKELY_HIT,
            overlap_fraction=0.85,
            is_bridge=True,
        )
    ]

    settlements_res, stats = analyze_cutoff(
        settlements, roads, damaged_roads, hospitals, {}
    )

    v1 = next(s for s in settlements_res if s.osm_id == 1)
    v2 = next(s for s in settlements_res if s.osm_id == 2)

    # Village 1 was connected pre-event (dist ~3.5km), but bridge 501 severed it post-event -> CUT_OFF
    assert v1.cutoff_state_conservative == CutoffState.CUT_OFF
    assert v1.cutoff_state_liberal == CutoffState.CUT_OFF
    assert v1.failed_segment_osm_id == 501
    assert v1.priority_rank == 1

    # Village 2 had NO road pre-event -> Must be UNKNOWN (never claimed cut off)
    assert v2.cutoff_state_conservative == CutoffState.UNKNOWN
    assert v2.cutoff_state_liberal == CutoffState.UNKNOWN

    assert stats.critical_bridge_osm_id == 501
    assert stats.buildings_affected_by_critical_bridge == 45
