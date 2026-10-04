"""
pipeline/schema.py
==================
Pydantic v2 schema for results.json.
This is the SINGLE SOURCE OF TRUTH shared by all pipeline modules,
the dashboard, and the PDF report generator.
Every number that appears in the report MUST come from here.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class DamageClass(str, Enum):
    LIKELY_HIT    = "likely_hit"
    POSSIBLY_HIT  = "possibly_hit"
    NOT_HIT       = "not_hit"
    UNKNOWN       = "unknown"


class CutoffState(str, Enum):
    CUT_OFF          = "cut_off"
    STILL_CONNECTED  = "still_connected"
    UNKNOWN          = "unknown"


class ConfidenceLevel(str, Enum):
    HIGH    = "high"
    MEDIUM  = "medium"
    LOW     = "low"


class FallbackReason(str, Enum):
    NO_S1_PAIR       = "no_s1_pair"
    DIFFERENT_ORBIT  = "different_orbit"
    S2_CLOUDY        = "s2_cloudy"
    NO_DEM           = "no_dem"
    NO_OSM           = "no_osm"
    MODEL_FALLBACK   = "threshold_only_baseline"
    COHERENCE_SKIP   = "coherence_skip"


# ──────────────────────────────────────────────────────────────────────────────
# Sub-schemas
# ──────────────────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    west:  float = Field(..., ge=-180, le=180)
    south: float = Field(..., ge=-90,  le=90)
    east:  float = Field(..., ge=-180, le=180)
    north: float = Field(..., ge=-90,  le=90)

    @model_validator(mode="after")
    def _check_order(self) -> "BoundingBox":
        if self.west >= self.east:
            raise ValueError("west must be < east")
        if self.south >= self.north:
            raise ValueError("south must be < north")
        return self


class S1ImageMeta(BaseModel):
    product_id:     str
    acquisition_dt: datetime
    relative_orbit: int
    pass_direction: str   # "ascending" | "descending"
    platform:       str   # "S1A" | "S1B" | "S1C"
    polarizations:  list[str]   # ["VV", "VH"]
    grd_url:        Optional[str] = None


class S2ImageMeta(BaseModel):
    product_id:        str
    acquisition_dt:    datetime
    cloud_cover_aoi:   float = Field(..., ge=0, le=100)
    scl_valid_fraction: float = Field(..., ge=0, le=1)
    bands_used:        list[str]


class DEMMeta(BaseModel):
    source:     str   # "COP-DEM GLO-30"
    tiles:      list[str]
    resolution: float  # metres
    crs:        str


class OSMSnapshot(BaseModel):
    snapshot_date: date
    api_endpoint:  str
    building_count: int
    road_km:        float
    bridge_count:   int
    hospital_count: int
    town_count:     int


class HazardStats(BaseModel):
    """Areas at conservative and liberal thresholds, km²."""
    flood_area_km2_conservative:  float = Field(..., ge=0)
    flood_area_km2_liberal:       float = Field(..., ge=0)
    debris_area_km2_conservative: float = Field(..., ge=0)
    debris_area_km2_liberal:      float = Field(..., ge=0)
    permanent_water_km2:          float = Field(..., ge=0)
    uncertain_pixels_fraction:    float = Field(..., ge=0, le=1)

    # Signal contributions to debris
    debris_s1_signal_used:    bool = False
    debris_coherence_used:    bool = False
    debris_s2_bsi_used:       bool = False
    debris_signals_available: int = Field(0, ge=0, le=3)


class ModelMeta(BaseModel):
    architecture:    str   # "unet-resnet34"
    encoder:         str
    weights_file:    str
    ensemble_size:   int
    threshold_conservative: float
    threshold_liberal:      float
    val_iou_conservative:   Optional[float] = None
    val_iou_liberal:        Optional[float] = None
    tta_enabled:     bool = False
    fallback_mode:   bool = False   # True if threshold-only baseline was used


class DamagedBuilding(BaseModel):
    osm_id:          int
    geometry_wkt:    str
    damage_class_conservative: DamageClass
    damage_class_liberal:      DamageClass
    overlap_fraction: float = Field(..., ge=0, le=1)
    centroid_lon:    float
    centroid_lat:    float


class DamagedRoad(BaseModel):
    osm_id:          int
    highway_type:    str
    length_m:        float
    damage_class_conservative: DamageClass
    damage_class_liberal:      DamageClass
    overlap_fraction: float = Field(..., ge=0, le=1)
    is_bridge:       bool = False


class DamageStats(BaseModel):
    buildings_likely_hit_conservative:   int = Field(..., ge=0)
    buildings_likely_hit_liberal:        int = Field(..., ge=0)
    buildings_possibly_hit_conservative: int = Field(..., ge=0)
    buildings_possibly_hit_liberal:      int = Field(..., ge=0)
    road_km_likely_hit_conservative:     float = Field(..., ge=0)
    road_km_likely_hit_liberal:          float = Field(..., ge=0)
    bridges_likely_hit:                  int = Field(..., ge=0)
    bridges_possibly_hit:                int = Field(..., ge=0)


class Settlement(BaseModel):
    osm_id:              Optional[int] = None
    name:                Optional[str] = None
    place_type:          str    # "village" | "hamlet" | "cluster"
    centroid_lon:        float
    centroid_lat:        float
    building_count:      int
    cutoff_state_conservative: CutoffState
    cutoff_state_liberal:      CutoffState
    distance_to_hospital_pre_km:  Optional[float] = None
    distance_to_hospital_post_km: Optional[float] = None
    detour_km:           Optional[float] = None
    failed_segment_osm_id: Optional[int] = None
    priority_rank:       Optional[int] = None


class CutoffStats(BaseModel):
    settlements_cut_off_conservative: int = Field(..., ge=0)
    settlements_cut_off_liberal:      int = Field(..., ge=0)
    settlements_possibly_cut_off:     int = Field(..., ge=0)
    settlements_still_connected:      int = Field(..., ge=0)
    settlements_unknown:              int = Field(..., ge=0)
    critical_bridge_osm_id:           Optional[int] = None
    buildings_affected_by_critical_bridge: int = 0


class FlowPathSettlement(BaseModel):
    name:              Optional[str] = None
    centroid_lon:      float
    centroid_lat:      float
    distance_from_source_km: float
    in_flood_mask:     bool
    elevation_m:       Optional[float] = None


class FlowPathResult(BaseModel):
    source_lon: float
    source_lat: float
    path_length_km:   float
    path_flood_overlap_fraction: float
    settlements: list[FlowPathSettlement] = []


class Warning(BaseModel):
    code:    str
    message: str
    stage:   str


class Provenance(BaseModel):
    s1_pre:  S1ImageMeta
    s1_post: S1ImageMeta
    s2_pre:  Optional[S2ImageMeta] = None
    s2_post: Optional[S2ImageMeta] = None
    dem:     DEMMeta
    osm:     OSMSnapshot


# ──────────────────────────────────────────────────────────────────────────────
# Top-level results.json model
# ──────────────────────────────────────────────────────────────────────────────

class SpaceTrackResults(BaseModel):
    """
    Root schema for results.json.
    All dashboard numbers and report numbers are pulled from here.
    """

    # Run metadata
    schema_version:   str = "1.0.0"
    run_id:           str
    run_timestamp:    datetime
    flood_date:       date
    bbox:             BoundingBox
    pipeline_version: str

    # Software versions
    python_version:   str
    torch_version:    str
    smp_version:      str

    # Configuration used
    config_hash:      str   # SHA-256 of configs/default.yaml

    # Fallbacks and warnings
    fallbacks_used:   list[FallbackReason] = []
    warnings:         list[Warning] = []
    confidence_level: ConfidenceLevel

    # Image provenance
    provenance: Provenance

    # Model metadata
    model_meta: ModelMeta

    # Results
    hazard:   HazardStats
    damage:   DamageStats
    cutoff:   CutoffStats

    # Per-feature lists (can be large)
    damaged_buildings: list[DamagedBuilding] = []
    damaged_roads:     list[DamagedRoad] = []
    settlements:       list[Settlement] = []

    # Bonus flow path (optional)
    flow_path: Optional[FlowPathResult] = None

    # Timing
    stage_timings_s: dict[str, float] = {}
    total_runtime_s: float = 0.0


    def all_numbers(self) -> dict[str, Any]:
        """
        Return every numeric value that appears in the report.
        Used by test_report_numbers.py to verify nothing is invented.
        """
        h = self.hazard
        d = self.damage
        c = self.cutoff
        return {
            "flood_area_km2_conservative":           h.flood_area_km2_conservative,
            "flood_area_km2_liberal":                h.flood_area_km2_liberal,
            "debris_area_km2_conservative":          h.debris_area_km2_conservative,
            "debris_area_km2_liberal":               h.debris_area_km2_liberal,
            "permanent_water_km2":                   h.permanent_water_km2,
            "uncertain_pixels_fraction":             h.uncertain_pixels_fraction,
            "buildings_likely_hit_conservative":     d.buildings_likely_hit_conservative,
            "buildings_likely_hit_liberal":          d.buildings_likely_hit_liberal,
            "buildings_possibly_hit_conservative":   d.buildings_possibly_hit_conservative,
            "buildings_possibly_hit_liberal":        d.buildings_possibly_hit_liberal,
            "road_km_likely_hit_conservative":       d.road_km_likely_hit_conservative,
            "road_km_likely_hit_liberal":            d.road_km_likely_hit_liberal,
            "bridges_likely_hit":                    d.bridges_likely_hit,
            "bridges_possibly_hit":                  d.bridges_possibly_hit,
            "settlements_cut_off_conservative":      c.settlements_cut_off_conservative,
            "settlements_cut_off_liberal":           c.settlements_cut_off_liberal,
            "settlements_possibly_cut_off":          c.settlements_possibly_cut_off,
            "settlements_still_connected":           c.settlements_still_connected,
            "total_runtime_s":                       self.total_runtime_s,
        }
