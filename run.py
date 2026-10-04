"""
run.py
======
Root entry point for SpaceTrack Flood.
Command:
    python run.py --bbox W,S,E,N --date YYYY-MM-DD

Runs end-to-end from raw satellite data for ANY mountain area and date.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from pipeline.schema import BoundingBox
from pipeline.results import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("spacetrack")


def parse_bbox(bbox_str: str) -> BoundingBox:
    parts = [float(p.strip()) for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Bounding box must be W,S,E,N (4 comma-separated floats)")
    w, s, e, n = parts
    return BoundingBox(west=w, south=s, east=e, north=n)


def parse_point(pt_str: str) -> tuple[float, float]:
    parts = [float(p.strip()) for p in pt_str.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Point must be LON,LAT (2 comma-separated floats)")
    return parts[0], parts[1]


def main():
    parser = argparse.ArgumentParser(
        description="SpaceTrack Flood: Rapid Satellite Flood & Debris Mapping Pipeline",
        epilog="Educational prototype, not an operational tool.",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        required=True,
        help="Bounding box in EPSG:4326 as W,S,E,N (e.g. 85.2,27.8,85.5,28.2)",
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Flood event date as YYYY-MM-DD (e.g. 2026-08-26)",
    )
    parser.add_argument(
        "--upstream-point",
        type=str,
        default=None,
        help="Optional clicked upstream point for flow-path tracing as LON,LAT",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run threshold-only baseline instead of full U-Net ensemble",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save results.json, report, and layers (default: outputs)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate one-page situation report PDF (EN and NE)",
    )

    args = parser.parse_args()

    bbox = parse_bbox(args.bbox)
    flood_date = date.fromisoformat(args.date)
    upstream = parse_point(args.upstream_point) if args.upstream_point else None
    out_dir = Path(args.output_dir)

    print("=" * 60)
    print("SpaceTrack Flood - Rapid Response Satellite Pipeline")
    print("Educational prototype, not an operational tool.")
    print(f"Target AOI:  {bbox.west}, {bbox.south} to {bbox.east}, {bbox.north}")
    print(f"Flood Date:  {flood_date}")
    print("=" * 60)

    results = run_pipeline(
        bbox=bbox,
        flood_date=flood_date,
        output_dir=out_dir,
        upstream_point=upstream,
        use_baseline=args.baseline,
    )

    print("\n" + "=" * 60)
    print("SUMMARY RESULTS (Strictly from results.json):")
    print(f"- Flood Extent:    {results.hazard.flood_area_km2_conservative} km² (cons) to {results.hazard.flood_area_km2_liberal} km² (lib)")
    print(f"- Debris Extent:   {results.hazard.debris_area_km2_conservative} km² (cons) to {results.hazard.debris_area_km2_liberal} km² (lib)")
    print(f"- Buildings Hit:   {results.damage.buildings_likely_hit_conservative} likely (cons) to {results.damage.buildings_likely_hit_liberal} (lib)")
    print(f"- Roads Damaged:   {results.damage.road_km_likely_hit_conservative} km (cons) to {results.damage.road_km_likely_hit_liberal} km (lib)")
    print(f"- Cut-off Villages:{results.cutoff.settlements_cut_off_conservative} definitely, {results.cutoff.settlements_possibly_cut_off} possibly")
    if results.cutoff.critical_bridge_osm_id:
        print(f"- Critical Bridge: OSM ID {results.cutoff.critical_bridge_osm_id} isolates {results.cutoff.buildings_affected_by_critical_bridge} buildings")
    print(f"- Total Runtime:   {results.total_runtime_s:.2f} seconds")
    print("=" * 60)

    if args.report:
        from app.report.generate_pdf import generate_situation_report
        generate_situation_report(results, out_dir / "reports")


if __name__ == "__main__":
    main()
