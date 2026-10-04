# 🛰️ SpaceTrack Flood

[![CI Pipeline](https://img.shields.io/badge/CI-Passing-brightgreen.svg)](tests/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **⚠️ Educational prototype, not an operational tool.**
> Built for a 15-day hackathon. Designed for mountain flash floods and debris flows in rugged terrain (e.g., Himalayas, Andes, Alps).

---

## 🎯 1. Project Overview

SpaceTrack Flood answers three critical questions for rescue coordination within 30 minutes of tasking:
1. **WHERE DID THE FLOOD HIT?** Maps flood inundation and debris fans by fusing Sentinel-1 radar (all-weather, penetrates monsoon cloud) with clear Sentinel-2 optical imagery and physical terrain models.
2. **WHAT WAS DAMAGED?** Overlays pre-event OpenStreetMap infrastructure to classify buildings, roads, and bridges as `likely_hit`, `possibly_hit`, or `not_hit`.
3. **WHO IS CUT OFF?** Conducts topological road network Dijkstra routing to classify settlements as `CUT_OFF`, `STILL_CONNECTED` (with measured detour km), or `UNKNOWN` (no pre-event road).

---

## 🚀 2. Quickstart & Smoke Test (< 30 Seconds)

### Option A: Local Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run tiny-AOI smoke test (Himalayan Trishuli Corridor)
python run.py --bbox 85.2,27.9,85.4,28.1 --date 2026-08-26 --report
```

### Option B: Docker Container
```bash
# Build reproducible container
docker build -t spacetrack-flood:latest .

# Run CLI analysis
docker run --rm spacetrack-flood:latest python run.py --bbox 85.2,27.9,85.4,28.1 --date 2026-08-26

# Launch interactive web dashboard
docker run --rm -p 8501:8501 spacetrack-flood:latest streamlit run app/dashboard/app.py
```

---

## 📐 3. System Architecture & Pipeline Stages

```
                        ┌────────────────────────────────────────┐
                        │        Target BBox + Flood Date        │
                        └───────────────────┬────────────────────┘
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: INGEST (ingest.py)                                                            │
│ • S1 IW GRD same relative orbit (6/12 days apart) • S2 L2A optical scenes             │
│ • Copernicus GLO-30 DEM • OSM pre-event snapshot (< flood date via ohsome)             │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: SAR PREPROCESSING & PHYSICAL MASKS (preprocess.py, masks.py)                  │
│ • Refined Lee speckle filter • Calibration to dB • Slope & HAND hydrologic masks       │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: DETECTION (detect.py, model/*)                                                │
│ • 9-Channel U-Net Ensemble with TTA • Epistemic uncertainty raster                    │
│ • Conservative & liberal thresholds • Permanent water suppression                      │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: DEBRIS HEURISTIC (debris.py)                                                  │
│ • 3-signal consensus: S1 VH roughness + InSAR coherence loss + S2 BSI Bare Soil Index │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: FUSION & POSTPROCESSING (fuse.py, postprocess.py)                             │
│ • S2 MNDWI optical verification • Small object removal • River connectivity filter     │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: DAMAGE ASSESSMENT (damage.py)                                                 │
│ • Positional uncertainty buffering (15 m) • OSM buildings, roads, bridges hit class   │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 7: CUT-OFF & ISOLATION ROUTING (cutoff.py)                                       │
│ • NetworkX Dijkstra routing • CUT_OFF vs STILL_CONNECTED vs UNKNOWN                    │
│ • Evacuation priority ranking • Critical bridge identification                         │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 8 & 9: FLOW PATH TRACING & OUTPUTS (flowpath.py, results.py, app/*)              │
│ • D8 downstream descent • results.json single source of truth • 1-Page PDF (EN/NE)     │
│ • Interactive Streamlit Dashboard with sub-second threshold reaction                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔒 4. Anti-Leakage & Consistency Guarantees

1. **Zero Data Leakage (`tests/test_no_leakage.py`)**:
   - Automated CI AST scan guarantees that `/pipeline` and `/model` never import `/eval` or touch EMSR/UNOSAT files.
   - Enforces that the OSM snapshot date is strictly prior to the flood date.
2. **Zero Invented Numbers (`tests/test_report_numbers.py`)**:
   - Automated test extracts every number appearing in the generated PDF/HTML situation report and verifies that it exists inside `results.json`.

---

## ⚠️ 5. Known Limitations

- **Revisit Time**: Sentinel-1's 6-to-12 day revisit period cannot provide minutes-ahead early warning of glacial lake collapses or dam breaks.
- **Mountain Radar Geometry**: Layover, shadow, and foreshortening mask portions of steep slopes (controlled via $\text{slope} \le 18^\circ$ and HAND masks).
- **Heuristic Debris**: Because no benchmark dataset currently labels debris flows, debris mapping is an explicit multi-sensor consensus heuristic.
- **Himalayan OSM Completeness**: Missing road tracks in remote gorges do not prove the absence of local footpaths; disconnected settlements are categorized as `UNKNOWN`, never claimed cut off.

---

## 🎙️ 6. Rehearsal Q&A

- **Q: Why require identical relative orbit pairs?**
  *A: Sentinel-1 is side-looking. Comparing ascending to descending or different orbits introduces massive geometric distortion and backscatter shifts from mountain slopes, causing catastrophic false positives.*
- **Q: Why not use Sentinel-2 optical alone?**
  *A: Himalayan monsoon cloud cover obscures optical view 75–90% of the time during flood season. Radar penetrates cloud cover.*
- **Q: How do you prevent mountain shadows from being detected as water?**
  *A: We apply dual-constraint filtering: slope cutoff ($\le 18^\circ$), HAND cutoff ($\le 25$ m above nearest river), and cross-pol VH backscatter checks.*

---

## 📜 7. Attributions & Citations

- **Sentinel Imagery**: Contains modified Copernicus Sentinel data 2026.
- **Copernicus DEM**: Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
- **OpenStreetMap**: © OpenStreetMap contributors.
- **Datasets**: Bountos et al., 2024 (*Kuro Siwo*); Bonafilia et al., 2020 (*Sen1Floods11*).
