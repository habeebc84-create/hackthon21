"""
tests/test_smoke.py
===================
Tiny-AOI end-to-end smoke test.
Verifies that `python run.py --bbox ... --date ...` completes end-to-end
in under 30 seconds and outputs a valid results.json conforming to schema.
"""
from datetime import date
from pathlib import Path
import json
import pytest

from pipeline.schema import BoundingBox, SpaceTrackResults
from pipeline.results import run_pipeline


def test_tiny_aoi_smoke_run(tmp_path):
    bbox = BoundingBox(west=85.25, south=27.95, east=85.35, north=28.05)
    flood_date = date(2026, 8, 26)
    out_dir = tmp_path / "smoke_outputs"

    results = run_pipeline(
        bbox=bbox,
        flood_date=flood_date,
        output_dir=out_dir,
        upstream_point=(85.30, 28.04),
        use_baseline=False,
    )

    # 1. Verify results object and runtime
    assert isinstance(results, SpaceTrackResults)
    assert results.total_runtime_s < 30.0  # Must be fast

    # 2. Verify results.json exists and parses cleanly
    results_json = out_dir / "results.json"
    assert results_json.exists()
    loaded_dict = json.loads(results_json.read_text(encoding="utf-8"))
    loaded_obj = SpaceTrackResults.model_validate(loaded_dict)

    # 3. Verify key fields
    assert loaded_obj.hazard.flood_area_km2_conservative >= 0.0
    assert loaded_obj.hazard.flood_area_km2_liberal >= loaded_obj.hazard.flood_area_km2_conservative
    assert loaded_obj.damage.buildings_likely_hit_conservative >= 0
    assert loaded_obj.cutoff.settlements_cut_off_conservative >= 0
    assert loaded_obj.flow_path is not None
