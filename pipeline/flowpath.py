"""
pipeline/flowpath.py
====================
STAGE 8 - HYDROLOGIC D8 DOWNSTREAM FLOW PATH TRACING (BONUS)
Given an arbitrary upstream source point (e.g., glacier lake outburst / landslide dam):
  1. Computes D8 steepest descent flow routing down the valley DEM
  2. Measures downstream path length (km)
  3. Calculates path overlap with observed satellite flood mask
  4. Identifies vulnerable settlements along the downstream corridor in sequential order
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from pipeline.schema import FlowPathResult, FlowPathSettlement

logger = logging.getLogger("spacetrack.flowpath")


def trace_d8_flow_path(
    dem: np.ndarray,
    start_rc: Tuple[int, int],
    max_steps: int = 1000,
    cell_size_m: float = 30.0,
) -> List[Tuple[int, int]]:
    """
    Trace steepest downhill path from start_rc using D8 gradient routing.
    Neighbor offsets: 8 directions with Euclidean distance weighting (1.0 orthogonal, sqrt(2) diagonal).
    """
    rows, cols = dem.shape
    r, c = start_rc
    path = [(r, c)]
    visited = {(r, c)}

    # 8 neighbor offsets: (dr, dc, dist_factor)
    d8_neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)
    ]

    for _ in range(max_steps):
        current_elev = dem[r, c]
        best_slope = 0.0
        best_neighbor = None

        for dr, dc, dist in d8_neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if (nr, nc) in visited:
                    continue
                drop = current_elev - dem[nr, nc]
                slope = drop / (dist * cell_size_m)
                if slope > best_slope:
                    best_slope = slope
                    best_neighbor = (nr, nc)

        if best_neighbor is None or best_slope <= 0.0:
            # Reached pit, flat lake surface, or edge of domain
            break

        r, c = best_neighbor
        path.append((r, c))
        visited.add((r, c))

    return path


def compute_flow_path(
    source_lon: float,
    source_lat: float,
    dem: np.ndarray,
    flood_mask: np.ndarray,
    settlements: List[Dict[str, Any]],
    transform: Tuple[float, float, float, float, float, float],
    corridor_buffer_px: int = 4,
    cell_size_m: float = 30.0,
) -> FlowPathResult:
    """
    Execute Stage 8 Flow Path Tracing.
    """
    lon_min, lat_max, px_w, px_h, rows, cols = transform

    start_c = int(round((source_lon - lon_min) / px_w))
    start_r = int(round((lat_max - source_lat) / px_h))

    start_r = np.clip(start_r, 0, rows - 1)
    start_c = np.clip(start_c, 0, cols - 1)

    path_rc = trace_d8_flow_path(dem, (start_r, start_c), cell_size_m=cell_size_m)

    # Calculate cumulative distance along path in km
    path_len_m = 0.0
    cum_dist_map: Dict[Tuple[int, int], float] = {(path_rc[0][0], path_rc[0][1]): 0.0}

    for i in range(1, len(path_rc)):
        r1, c1 = path_rc[i - 1]
        r2, c2 = path_rc[i]
        step_dist = np.sqrt((r2 - r1)**2 + (c2 - c1)**2) * cell_size_m
        path_len_m += step_dist
        cum_dist_map[(r2, c2)] = path_len_m

    path_length_km = round(path_len_m / 1000.0, 2)

    # Overlap with satellite flood mask
    overlap_count = 0
    for r, c in path_rc:
        if flood_mask[r, c]:
            overlap_count += 1
    overlap_frac = round(overlap_count / max(1, len(path_rc)), 4)

    # Identify downstream settlements within corridor
    path_set = set(path_rc)
    flow_settlements: List[FlowPathSettlement] = []

    for s in settlements:
        s_lon = s["lon"]
        s_lat = s["lat"]
        sc = int(round((s_lon - lon_min) / px_w))
        sr = int(round((lat_max - s_lat) / px_h))

        if not (0 <= sr < rows and 0 <= sc < cols):
            continue

        # Check proximity to path
        min_dist_to_path_px = float("inf")
        closest_path_node = None
        for pr, pc in path_rc:
            d_px = np.sqrt((sr - pr)**2 + (sc - pc)**2)
            if d_px < min_dist_to_path_px:
                min_dist_to_path_px = d_px
                closest_path_node = (pr, pc)

        if min_dist_to_path_px <= corridor_buffer_px and closest_path_node:
            d_km = round(cum_dist_map[closest_path_node] / 1000.0, 2)
            in_flood = bool(flood_mask[sr, sc])
            elev = float(dem[sr, sc])

            flow_settlements.append(
                FlowPathSettlement(
                    name=s.get("name"),
                    centroid_lon=s_lon,
                    centroid_lat=s_lat,
                    distance_from_source_km=d_km,
                    in_flood_mask=in_flood,
                    elevation_m=elev,
                )
            )

    # Sort sequentially downstream by distance from source
    flow_settlements.sort(key=lambda x: x.distance_from_source_km)

    return FlowPathResult(
        source_lon=source_lon,
        source_lat=source_lat,
        path_length_km=path_length_km,
        path_flood_overlap_fraction=overlap_frac,
        settlements=flow_settlements,
    )
