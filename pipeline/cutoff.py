"""
pipeline/cutoff.py
==================
STAGE 7 - NETWORK ISOLATION & SETTLEMENT CUT-OFF ANALYSIS
Constructs pre-event and post-event topological road networks via NetworkX.
Executes Dijkstra routing to hospitals and towns.
Classifies settlements into 3 rigorous states:
  - CUT_OFF (connected pre-event, unreachable post-event)
  - STILL_CONNECTED (accessible, with measured detour km)
  - UNKNOWN (unconnected pre-event; never falsely classified as cut off)
Ranks isolation priority and identifies critical bridge failure points.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

from pipeline.schema import (
    CutoffState,
    CutoffStats,
    DamagedRoad,
    DamageClass,
    Settlement,
)

logger = logging.getLogger("spacetrack.cutoff")


def build_road_graph(roads: List[Dict[str, Any]]) -> nx.Graph:
    """Build NetworkX undirected road graph with edge lengths and OSM IDs."""
    G = nx.Graph()
    for rd in roads:
        osm_id = rd["osm_id"]
        pts = rd.get("points", [])
        length_m = float(rd.get("length_m", 100.0))
        is_bridge = rd.get("is_bridge", False)

        if len(pts) >= 2:
            u, v = pts[0], pts[-1]
            G.add_edge(
                u, v,
                osm_id=osm_id,
                weight=length_m,
                is_bridge=is_bridge,
                pts=pts,
            )
    return G


def find_nearest_node(G: nx.Graph, point: Tuple[float, float], max_snap_dist_deg: float = 0.05) -> Optional[Tuple[float, float]]:
    """Snap a coordinate to the closest graph node within threshold distance."""
    if not G.nodes:
        return None
    px, py = point
    best_node = None
    min_dist_sq = float("inf")
    for nx_x, nx_y in G.nodes:
        d2 = (px - nx_x) ** 2 + (py - nx_y) ** 2
        if d2 < min_dist_sq:
            min_dist_sq = d2
            best_node = (nx_x, nx_y)

    if (min_dist_sq ** 0.5) <= max_snap_dist_deg:
        return best_node
    return None


def analyze_cutoff(
    settlements_data: List[Dict[str, Any]],
    roads_data: List[Dict[str, Any]],
    damaged_roads: List[DamagedRoad],
    destinations_data: List[Dict[str, Any]],  # Hospitals & Towns
    config: Dict[str, Any],
) -> Tuple[List[Settlement], CutoffStats]:
    """
    Execute Stage 7 Cut-Off Analysis.
    Evaluates accessibility at both conservative and liberal thresholds.
    """
    G_pre = build_road_graph(roads_data)

    # Identify severed edges
    severed_osm_ids_cons: Set[int] = set()
    severed_osm_ids_lib: Set[int] = set()
    for dr in damaged_roads:
        if dr.damage_class_conservative in (DamageClass.LIKELY_HIT, DamageClass.POSSIBLY_HIT):
            severed_osm_ids_cons.add(dr.osm_id)
        if dr.damage_class_liberal in (DamageClass.LIKELY_HIT, DamageClass.POSSIBLY_HIT):
            severed_osm_ids_lib.add(dr.osm_id)

    # Build post-event subgraphs
    G_post_cons = G_pre.copy()
    for u, v, d in list(G_post_cons.edges(data=True)):
        if d.get("osm_id") in severed_osm_ids_cons:
            G_post_cons.remove_edge(u, v)

    G_post_lib = G_pre.copy()
    for u, v, d in list(G_post_lib.edges(data=True)):
        if d.get("osm_id") in severed_osm_ids_lib:
            G_post_lib.remove_edge(u, v)

    # Snap destination hospital/town coordinates
    dest_nodes_pre = []
    dest_nodes_post_cons = []
    dest_nodes_post_lib = []

    for dst in destinations_data:
        pt = (dst["lon"], dst["lat"])
        n_pre = find_nearest_node(G_pre, pt)
        if n_pre:
            dest_nodes_pre.append(n_pre)
        n_pcons = find_nearest_node(G_post_cons, pt)
        if n_pcons:
            dest_nodes_post_cons.append(n_pcons)
        n_plib = find_nearest_node(G_post_lib, pt)
        if n_plib:
            dest_nodes_post_lib.append(n_plib)

    processed_settlements: List[Settlement] = []

    c_cutoff_cons = 0
    c_cutoff_lib = 0
    c_possibly_cutoff = 0
    c_connected = 0
    c_unknown = 0

    bridge_impact_map: Dict[int, int] = {}

    for s in settlements_data:
        osm_id = s.get("osm_id")
        name = s.get("name", "Unnamed settlement")
        place_type = s.get("place_type", "village")
        b_count = int(s.get("building_count", 10))
        s_pt = (s["lon"], s["lat"])

        s_node_pre = find_nearest_node(G_pre, s_pt)

        # 1. Pre-event accessibility check
        pre_dist_m = None
        pre_path = None
        if s_node_pre and dest_nodes_pre:
            for dst_node in dest_nodes_pre:
                try:
                    d = nx.shortest_path_length(G_pre, source=s_node_pre, target=dst_node, weight="weight")
                    p = nx.shortest_path(G_pre, source=s_node_pre, target=dst_node, weight="weight")
                    if pre_dist_m is None or d < pre_dist_m:
                        pre_dist_m = d
                        pre_path = p
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

        if pre_dist_m is None:
            # Rule: If no pre-event road connection, state is strictly UNKNOWN
            st_cons = CutoffState.UNKNOWN
            st_lib = CutoffState.UNKNOWN
            c_unknown += 1
            processed_settlements.append(
                Settlement(
                    osm_id=osm_id,
                    name=name,
                    place_type=place_type,
                    centroid_lon=s["lon"],
                    centroid_lat=s["lat"],
                    building_count=b_count,
                    cutoff_state_conservative=st_cons,
                    cutoff_state_liberal=st_lib,
                )
            )
            continue

        # 2. Post-event accessibility checks
        # Conservative
        s_node_cons = find_nearest_node(G_post_cons, s_pt)
        post_dist_cons = None
        if s_node_cons and dest_nodes_post_cons:
            for dst_node in dest_nodes_post_cons:
                try:
                    d = nx.shortest_path_length(G_post_cons, source=s_node_cons, target=dst_node, weight="weight")
                    if post_dist_cons is None or d < post_dist_cons:
                        post_dist_cons = d
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

        # Liberal
        s_node_lib = find_nearest_node(G_post_lib, s_pt)
        post_dist_lib = None
        if s_node_lib and dest_nodes_post_lib:
            for dst_node in dest_nodes_post_lib:
                try:
                    d = nx.shortest_path_length(G_post_lib, source=s_node_lib, target=dst_node, weight="weight")
                    if post_dist_lib is None or d < post_dist_lib:
                        post_dist_lib = d
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

        st_cons = CutoffState.STILL_CONNECTED if post_dist_cons is not None else CutoffState.CUT_OFF
        st_lib = CutoffState.STILL_CONNECTED if post_dist_lib is not None else CutoffState.CUT_OFF

        if st_cons == CutoffState.CUT_OFF:
            c_cutoff_cons += 1
        if st_lib == CutoffState.CUT_OFF:
            c_cutoff_lib += 1

        if st_cons == CutoffState.STILL_CONNECTED and st_lib == CutoffState.CUT_OFF:
            c_possibly_cutoff += 1
        elif st_cons == CutoffState.STILL_CONNECTED and st_lib == CutoffState.STILL_CONNECTED:
            c_connected += 1

        # Trace failed segment responsible along pre-event path
        failed_seg_id = None
        if st_lib == CutoffState.CUT_OFF and pre_path:
            for i in range(len(pre_path) - 1):
                edge_data = G_pre.get_edge_data(pre_path[i], pre_path[i + 1])
                seg_id = edge_data.get("osm_id")
                if seg_id in severed_osm_ids_lib:
                    failed_seg_id = seg_id
                    bridge_impact_map[seg_id] = bridge_impact_map.get(seg_id, 0) + b_count
                    break

        detour_km = None
        if post_dist_cons is not None and pre_dist_m is not None:
            detour_km = max(0.0, round((post_dist_cons - pre_dist_m) / 1000.0, 2))

        processed_settlements.append(
            Settlement(
                osm_id=osm_id,
                name=name,
                place_type=place_type,
                centroid_lon=s["lon"],
                centroid_lat=s["lat"],
                building_count=b_count,
                cutoff_state_conservative=st_cons,
                cutoff_state_liberal=st_lib,
                distance_to_hospital_pre_km=round(pre_dist_m / 1000.0, 2) if pre_dist_m else None,
                distance_to_hospital_post_km=round(post_dist_cons / 1000.0, 2) if post_dist_cons else None,
                detour_km=detour_km,
                failed_segment_osm_id=failed_seg_id,
            )
        )

    # 3. Priority ranking for rescue teams: Cut-off settlements with most buildings first
    cutoff_only = [s for s in processed_settlements if s.cutoff_state_liberal == CutoffState.CUT_OFF]
    cutoff_only.sort(key=lambda s: (s.building_count, s.distance_to_hospital_pre_km or 0), reverse=True)
    for rank, s in enumerate(cutoff_only, start=1):
        s.priority_rank = rank

    # 4. Critical bridge identification
    critical_bridge_id = None
    max_impacted_buildings = 0
    if bridge_impact_map:
        critical_bridge_id, max_impacted_buildings = max(bridge_impact_map.items(), key=lambda x: x[1])

    stats = CutoffStats(
        settlements_cut_off_conservative=c_cutoff_cons,
        settlements_cut_off_liberal=c_cutoff_lib,
        settlements_possibly_cut_off=c_possibly_cutoff,
        settlements_still_connected=c_connected,
        settlements_unknown=c_unknown,
        critical_bridge_osm_id=critical_bridge_id,
        buildings_affected_by_critical_bridge=max_impacted_buildings,
    )

    return processed_settlements, stats
