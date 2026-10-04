"""
pipeline/damage.py
==================
STAGE 6 - EXPOSURE & INFRASTRUCTURE DAMAGE ASSESSMENT
Intersects pre-event OpenStreetMap buildings, road segments, and bridges
with the buffered hazard raster (flood + debris) to account for positional uncertainty.
Classifies each feature as likely_hit, possibly_hit, or not_hit at conservative & liberal thresholds.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
from shapely.geometry import Point, LineString, Polygon, mapping
from shapely import wkt

from pipeline.schema import (
    DamageClass,
    DamagedBuilding,
    DamagedRoad,
    DamageStats,
)

logger = logging.getLogger("spacetrack.damage")


def classify_overlap(fraction: float, is_bridge: bool = False) -> DamageClass:
    """Classify damage based on fraction of feature inside the buffered hazard."""
    if fraction >= 0.50:
        return DamageClass.LIKELY_HIT
    elif fraction >= 0.05 or (is_bridge and fraction > 0.0):
        # Critical infrastructure rule: bridges adjacent to/touching hazard default to possibly_hit
        return DamageClass.POSSIBLY_HIT
    else:
        return DamageClass.NOT_HIT


def assess_damage(
    buildings: List[Dict[str, Any]],
    roads: List[Dict[str, Any]],
    hazard_cons: np.ndarray,
    hazard_lib: np.ndarray,
    transform: Tuple[float, float, float, float, float, float],
    config: Dict[str, Any],
) -> Tuple[List[DamagedBuilding], List[DamagedRoad], DamageStats]:
    """
    Assess structural damage to buildings, roads, and bridges.
    Transform: affine tuple (lon_min, lat_max, pixel_width, pixel_height, h, w)
    """
    lon_min, lat_max, px_w, px_h, rows, cols = transform

    def point_to_rc(lon: float, lat: float) -> Tuple[int, int]:
        c = int(round((lon - lon_min) / px_w))
        r = int(round((lat_max - lat) / px_h))
        return r, c

    damaged_buildings: List[DamagedBuilding] = []
    damaged_roads: List[DamagedRoad] = []

    # Counters
    b_likely_cons = 0
    b_likely_lib = 0
    b_poss_cons = 0
    b_poss_lib = 0
    r_km_likely_cons = 0.0
    r_km_likely_lib = 0.0
    br_likely = 0
    br_possibly = 0

    # 1. Process Buildings
    for b in buildings:
        osm_id = b["osm_id"]
        c_lon = b["centroid_lon"]
        c_lat = b["centroid_lat"]
        geom_wkt = b.get("geometry_wkt", f"POINT({c_lon} {c_lat})")

        r, c = point_to_rc(c_lon, c_lat)
        if 0 <= r < rows and 0 <= c < cols:
            # Check local 3x3 window representing ~15m positional buffer
            r_slice = slice(max(0, r - 1), min(rows, r + 2))
            c_slice = slice(max(0, c - 1), min(cols, c + 2))

            win_cons = float(np.mean(hazard_cons[r_slice, c_slice]))
            win_lib = float(np.mean(hazard_lib[r_slice, c_slice]))

            # If centroid itself is submerged or high window fraction
            frac_cons = 1.0 if hazard_cons[r, c] else win_cons
            frac_lib = 1.0 if hazard_lib[r, c] else win_lib
        else:
            frac_cons, frac_lib = 0.0, 0.0

        cls_cons = classify_overlap(frac_cons, is_bridge=False)
        cls_lib = classify_overlap(frac_lib, is_bridge=False)

        if cls_cons == DamageClass.LIKELY_HIT:
            b_likely_cons += 1
        elif cls_cons == DamageClass.POSSIBLY_HIT:
            b_poss_cons += 1

        if cls_lib == DamageClass.LIKELY_HIT:
            b_likely_lib += 1
        elif cls_lib == DamageClass.POSSIBLY_HIT:
            b_poss_lib += 1

        damaged_buildings.append(
            DamagedBuilding(
                osm_id=osm_id,
                geometry_wkt=geom_wkt,
                damage_class_conservative=cls_cons,
                damage_class_liberal=cls_lib,
                overlap_fraction=round(frac_lib, 4),
                centroid_lon=c_lon,
                centroid_lat=c_lat,
            )
        )

    # 2. Process Roads & Bridges
    for rd in roads:
        osm_id = rd["osm_id"]
        hw_type = rd.get("highway_type", "unclassified")
        is_bridge = rd.get("is_bridge", False)
        length_m = float(rd.get("length_m", 100.0))
        pts = rd.get("points", [])

        # Sample points along road
        hit_cons_samples = 0
        hit_lib_samples = 0
        valid_samples = 0

        for lon, lat in pts:
            r, c = point_to_rc(lon, lat)
            if 0 <= r < rows and 0 <= c < cols:
                valid_samples += 1
                if hazard_cons[r, c]:
                    hit_cons_samples += 1
                if hazard_lib[r, c]:
                    hit_lib_samples += 1

        frac_cons = (hit_cons_samples / valid_samples) if valid_samples > 0 else 0.0
        frac_lib = (hit_lib_samples / valid_samples) if valid_samples > 0 else 0.0

        cls_cons = classify_overlap(frac_cons, is_bridge=is_bridge)
        cls_lib = classify_overlap(frac_lib, is_bridge=is_bridge)

        if cls_cons == DamageClass.LIKELY_HIT:
            r_km_likely_cons += length_m / 1000.0
        if cls_lib == DamageClass.LIKELY_HIT:
            r_km_likely_lib += length_m / 1000.0

        if is_bridge:
            if cls_lib == DamageClass.LIKELY_HIT:
                br_likely += 1
            elif cls_lib == DamageClass.POSSIBLY_HIT:
                br_possibly += 1

        damaged_roads.append(
            DamagedRoad(
                osm_id=osm_id,
                highway_type=hw_type,
                length_m=round(length_m, 1),
                damage_class_conservative=cls_cons,
                damage_class_liberal=cls_lib,
                overlap_fraction=round(frac_lib, 4),
                is_bridge=is_bridge,
            )
        )

    stats = DamageStats(
        buildings_likely_hit_conservative=b_likely_cons,
        buildings_likely_hit_liberal=b_likely_lib,
        buildings_possibly_hit_conservative=b_poss_cons,
        buildings_possibly_hit_liberal=b_poss_lib,
        road_km_likely_hit_conservative=round(r_km_likely_cons, 2),
        road_km_likely_hit_liberal=round(r_km_likely_lib, 2),
        bridges_likely_hit=br_likely,
        bridges_possibly_hit=br_possibly,
    )

    return damaged_buildings, damaged_roads, stats
