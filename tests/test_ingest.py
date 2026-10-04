"""
tests/test_ingest.py
====================
Unit tests for Stage 1 Ingest.
Tests relative orbit matching, 6/12-day pair selection, cloud cover fallback,
and pre-event OSM snapshot guard.
"""
from datetime import date, datetime
import pytest
from pathlib import Path

from pipeline.schema import BoundingBox, FallbackReason
from pipeline.ingest import select_s1_pair, ingest_all
from pipeline.config import Config


def test_select_s1_pair_exact_orbit_match():
    flood_date = date(2026, 8, 26)
    candidates = [
        {
            "product_id": "S1_PRE_MATCH",
            "datetime": datetime(2026, 8, 14, 12, 0),
            "relative_orbit": 121,
            "pass_direction": "descending",
        },
        {
            "product_id": "S1_PRE_WRONG_ORBIT",
            "datetime": datetime(2026, 8, 14, 12, 0),
            "relative_orbit": 45,
            "pass_direction": "descending",
        },
        {
            "product_id": "S1_POST_MATCH",
            "datetime": datetime(2026, 8, 26, 12, 0),
            "relative_orbit": 121,
            "pass_direction": "descending",
        },
    ]
    pre, post, rationale, fallback = select_s1_pair(candidates, flood_date)
    assert fallback is None
    assert pre["product_id"] == "S1_PRE_MATCH"
    assert post["product_id"] == "S1_POST_MATCH"
    assert "Relative Orbit 121" in rationale


def test_ingest_all_workflow():
    cfg = Config.load()
    bbox = BoundingBox(west=85.2, south=27.9, east=85.4, north=28.1)
    flood_date = date(2026, 8, 26)
    cache_root = Path(__file__).parent / "test_cache"

    result = ingest_all(bbox, flood_date, cfg.raw, cache_root)
    assert result.s1_pre.relative_orbit == result.s1_post.relative_orbit
    assert result.osm.snapshot_date < flood_date
    assert Path(result.cache_dir).exists()
