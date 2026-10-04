<USER_REQUEST>
use these always You are a senior geospatial ML engineer. Build a complete, working, reproducible system called "SpaceTrack Flood" for a 15-day hackathon (team of 4). Write real code, not pseudocode. Work stage by stage, test each stage on a small sample before moving on, and never stop to ask questions: make a sensible assumption, state it in one line, and continue.

====================================================
1. PROJECT GOAL
====================================================
Input: a bounding box (W,S,E,N) and a flood date (YYYY-MM-DD).
Command: python run.py --bbox W,S,E,N --date YYYY-MM-DD
The system must run end-to-end from raw satellite data for ANY mountain area and date a judge chooses. Nothing may be hard-coded to one location.

It must answer three questions for rescue teams:
1. WHERE DID THE FLOOD HIT? Map flooded areas and debris-covered areas by comparing before/after images. Use Sentinel-1 radar (primary, sees through monsoon cloud) and Sentinel-2 optical (secondary, only on clear pixels).
2. WHAT WAS DAMAGED? Overlay pre-event OpenStreetMap buildings, roads and bridges and classify each as likely hit / possibly hit / not hit.
3. WHO IS CUT OFF? Using the road network, find settlements that lost their road connection to the nearest town or hospital.

Case study: the August 2026 Trishuli (Bhote Koshi-Trishuli corridor, Nepal) flood, event date 2026-08-26. Compare results with Copernicus EMS activation EMSR927 (CHECKING ONLY).

Outputs: interactive map dashboard, one-page situation report (English + Nepali) as PDF, results.json, GeoPackage.
Bonus: given any upstream point, trace the flood path down the valley with elevation data and list settlements along it.

====================================================
2. HARD RULES (violating these means disqualification)
====================================================
- ALLOWED inputs: Sentinel-1, Sentinel-2, Copernicus DEM GLO-30, OpenStreetMap as of a snapshot BEFORE 26 Aug 2026 (use 2026-07-27, via the ohsome API https://api.ohsome.org/v1), and the listed training datasets (Kuro Siwo recommended; Sen1Floods11 optional).
- FORBIDDEN as inputs: Copernicus EMS (incl. EMSR927), UNOSAT or any published damage map, any OSM edit made after the event. These may ONLY be used inside /eval to check results.
- Enforce this in code: nothing in /pipeline or /model may import from /eval or read any EMSR/UNOSAT file. Add a CI test (tests/test_no_leakage.py) that scans imports and file paths and fails the build on any violation. Hard-fail if the OSM snapshot date is on or after the flood date.
- Every number in the report/dashboard must come from results.json. Never invent numbers. Add an automated consistency check that fails the build if a report number is not in results.json.
- Educational prototype: show the banner "Educational prototype, not an operational tool." No images of victims anywhere.
- Required attribution strings in dashboard, report, README and ATTRIBUTION.md:
  "Contains modified Copernicus Sentinel data 2026."
  "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."
  "© OpenStreetMap contributors."
  Cite Bountos et al., 2024 (Kuro Siwo) and Bonafilia et al., CVPR Workshops 2020 (Sen1Floods11, if used).
  EMSR927 credit: "European Union, Copernicus Emergency Management Service data."

====================================================
3. REPO LAYOUT
====================================================
/pipeline  ingest.py preprocess.py masks.py detect.py debris.py fuse.py postprocess.py damage.py cutoff.py flowpath.py results.py
/model     dataset.py train.py eval.py infer.py weights/
/app       dashboard/  report/  templates/en  templates/ne
/eval      emsr927_compare.py  (checking only, never imported by pipeline)
/tests     unit tests + test_no_leakage.py + test_report_numbers.py + tiny-AOI smoke test
/configs   default.yaml (all thresholds, paths, model settings)
/docs
run.py  README.md  ATTRIBUTION.md  Dockerfile  environment.yml  Makefile

Use Python 3.11, rasterio, geopandas, shapely, xarray, numpy, scipy, scikit-image, torch, segmentation_models_pytorch, osmnx, networkx, pysheds (or whitebox), pyroSAR/SNAP gpt (with a documented fallback), pystac-client, requests, jinja2, weasyprint, streamlit + leafmap/folium (or FastAPI + MapLibre). Pin versions. Cache every download by (bbox, date, version) so reruns are fast. Log every stage with timings.

====================================================
4. PIPELINE STAGES
====================================================
Define a fixed results.json schema FIRST (pydantic model) shared by all modules: run metadata (bbox, date, image dates, orbit, fallbacks used, versions, thresholds), hazard stats (km2 flood/debris at conservative/liberal), per-feature damage lists, per-settlement cut-off status, flow-path settlements, warnings, provenance.

STAGE 1 - INGEST (ingest.py)
- Query Copernicus Data Space (CDSE STAC/OData, credentials from env vars) for Sentinel-1 IW GRD. Automatically select a pre/post pair on the SAME relative orbit and same direction, 12 days apart (accept 6 days if available), post-event image as soon after the flood date as possible. Reject different-track pairs. Log WHY the pair was chosen. If no valid pair exists: warn, fall back to lower-confidence mode, record it in results.json.
- Sentinel-2 L2A: nearest clear pre/post scenes with the SCL cloud mask; if cloud cover over the AOI is too high, skip S2 and record it.
- DEM: Copernicus GLO-30 tiles for the AOI.
- OSM via ohsome /elements/geometry at snapshot 2026-07-27: buildings, highways, bridges (bridge=*), amenity=hospital, place=*.
- Never crash on missing data: warn and degrade gracefully.

STAGE 2 - SENTINEL-1 PREPROCESSING (preprocess.py, masks.py)
- Apply orbit file, thermal noise removal, calibration, terrain-flattened gamma0 (RTC) using the DEM, Refined Lee (or multi-temporal) speckle filter, convert to dB, co-register pre/post on a 10 m grid.
- Build masks: layover/shadow, slope (>~15-20 deg excluded for water), HAND (>~15-30 m excluded), distance-to-river channel. Make cutoffs configurable and tuned on validation scenes.

STAGE 3 - DETECTION (detect.py, model/*)
- Model: segmentation_models_pytorch U-Net, ResNet/EfficientNet encoder, pretrained, fine-tuned.
- Inputs: pre VV/VH, post VV/VH, difference, log-ratio, slope, HAND, distance-to-river.
- Classes: permanent water, flood water, background (matches Kuro Siwo).
- Loss: Dice + focal. Augment: flips, rotations, speckle noise, brightness shifts.
- Train on Kuro Siwo (write the dataloader). Hold out by GEOGRAPHY. Use Sen1Floods11 Nepal/South Asia scenes as unseen Himalayan test data (verify the split yourself, avoid leakage).
- Ensemble 3-5 models + test-time augmentation; output a flood PROBABILITY raster plus a variance-based UNCERTAINTY raster. Choose thresholds from validation F1/IoU, and output conservative and liberal thresholds. Remove permanent water so only change is reported.
- Provide a threshold-only baseline for the ablation.
- Training script must run on a free Kaggle/Colab GPU, with a documented command and saved weights.

STAGE 4 - DEBRIS (debris.py)
- Neither dataset has a debris class, so build it as a heuristic and say so. Combine three independent signals inside a buffered river corridor: S1 backscatter log-ratio (VH roughness), InSAR coherence loss (S1 SLC, optional), S2 bare-soil-index change (clear pixels). Mark debris where at least 2 of 3 available signals agree; report each signal's contribution.

STAGE 5 - FUSION + POST-PROCESSING (fuse.py, postprocess.py)
- S2 MNDWI / BSI / NDVI confirmation on cloud-free pixels (dilate the cloud and shadow masks). Where S1 and S2 disagree, label the pixel "uncertain".
- Connectivity filter (keep blobs connected to the river network), remove objects under ~5-10 px, fill small holes, reject flood on ridges/high HAND.

STAGE 6 - DAMAGE (damage.py)
- Intersect OSM buildings, road segments and bridges with the hazard mask buffered 10-20 m for positional uncertainty. Score by fraction of the feature inside the hazard.
- Classes: likely hit / possibly hit / not hit, at both thresholds. Bridges within or adjacent to the corridor default to "possibly hit". Report counts and lengths per class.

STAGE 7 - CUT-OFF (cutoff.py)
- Build the road graph (osmnx/networkx). Remove edges intersecting the flood/debris mask. Destinations: pre-event hospitals and towns. Dijkstra from each settlement (place=village/hamlet, or building clusters snapped to the nearest road node within a distance limit).
- THREE states: CUT OFF (reachable pre-event, unreachable post-event), STILL CONNECTED (with extra detour distance), UNKNOWN (no pre-event road link; never claim cut off). Run at both thresholds to give "definitely" and "possibly" cut off lists.
- Add: ranked priority list (building count as population proxy, distance to hospital); bridge/segment criticality (which single segment, if lost, cuts off the most buildings); for each cut-off village the failed segment responsible.

STAGE 8 - BONUS FLOW PATH (flowpath.py)
- Fill depressions, D8 flow direction, trace downstream from a clicked upstream point, HAND-based buffer along the path, list settlements in downstream order, and report what fraction of the traced path overlaps the observed flood mask.

STAGE 9 - OUTPUTS
- results.json, GeoPackage, GeoJSON, COG tiles for the dashboard.
- DASHBOARD: left panel (bbox draw tool, date, Run button, progress log with stage timings and ETA); center map (S1/S2 before-after swipe, toggle layers: flood, debris, damaged features, cut-off settlements, uncertainty; legend; opacity sliders; basemap switch); right panel (key numbers with conservative-to-liberal ranges: km2 flooded, buildings likely hit, roads cut, villages cut off, plus sortable priority table, click a row to zoom). Confidence slider updates map, numbers and lists in under 1 s. Click a village to see status, reason, failed road segment, detour, confidence. Click a road/bridge to see hit class and overlap %. Click-to-trace upstream point. Export buttons: PDF, GeoPackage, GeoJSON, PNG. EN/NE language toggle. Colour-blind-safe palette (flood blue, debris brown/gold, cut-off red, unknown grey hatch, connected green) plus icons/patterns. Yellow warning badges for every fallback used. Footer with all attributions and the educational-prototype banner. Low-bandwidth mode.
- SITUATION REPORT (Jinja2 -> WeasyPrint PDF, EN and NE, Noto Sans Devanagari, exactly one page): headline line ("X km2 flooded, Y buildings likely hit, Z villages cut off (N possibly), as of [image dates]"), map with legend and scale bar, top-5 priority villages table, damage table, data/confidence box (image dates, orbit, S2 availability, model confidence, fallbacks), short specific limitations box, attributions. Every number pulled from results.json.
- OPTIONAL LLM COPILOT: answers rescuer questions and writes short reports in English and Nepali. The LLM only rephrases data from results.json; an automatic check rejects any output containing a number not present in results.json and falls back to the template text.

====================================================
5. VALIDATION (/eval, checking only)
====================================================
- emsr927_compare.py: flood-extent IoU, precision, recall, F1; building and road agreement; a disagreement map classifying each disagreement as omission, commission or reference-map uncertainty. Output a table + figures.
- Metrics per scene and per terrain type (valley floor, steep gorge, glacial lake). Ablation table: threshold baseline vs U-Net vs U-Net + terrain inputs vs ensemble. Calibration plot. Show that low-confidence pixels have higher error. Before/after figure of false positives removed by the terrain masks (target at least 50% reduction on steep slopes).
- Generalization test on other real events (Chamoli 2021, Melamchi 2021, South Lhonak GLOF 2023) plus random mountain AOIs, shown as a small-multiples figure.
- Targets: flood IoU 0.6-0.75 vs EMSR927 on water (report debris honestly, it will be lower), new-AOI runtime under 30 minutes, zero report numbers outside results.json.

====================================================
6. LIMITATIONS SECTION (write specifically, with measured numbers where possible)
====================================================
Revisit time of days means no early warning (cannot warn of a glacier collapse minutes ahead); layover/shadow/steep slopes cause false positives and blind spots; wet snow and shadow mimic water in radar; debris detection is heuristic with no labeled training data; 10 m pixels cannot resolve individual houses or footbridges; Himalayan OSM is incomplete so a missing road is not proof of no road; road passability, flood depth and velocity are not observable; educational prototype only. Add a "what would change our answer" section showing how results shift with threshold, date pair and OSM snapshot.

====================================================
7. DELIVERABLES
====================================================
- Working repo with a README that takes a clean machine to a successful run (Docker + Makefile), and a tiny-AOI smoke test that runs in under 5 minutes.
- Report (max 6 pages) with methods, results, EMSR927 comparison, and limitations.
- Script for a 3-minute demo video (problem 0:20, live Trishuli run 1:10, proof via EMSR927 numbers + second event 0:45, honesty 0:30, impact 0:15).
- ATTRIBUTION.md and dataset citations.
- A rehearsal Q&A document: why this orbit pair, why not S2 only, how do you know the model is not detecting shadow, what if there is no pre-event road, how do you avoid hallucinated numbers.

====================================================
8. HOW TO WORK
====================================================
1. First output: repo skeleton, the pydantic results.json schema, configs/default.yaml, Dockerfile, and the leakage test.
2. Then implement stages in order 1 to 9, each with a unit test and a runnable demo on a small AOI. After each stage, print a one-paragraph summary of what works, what is stubbed, and the next step.
3. Prefer correct and testable over clever. Handle errors with clear warnings and graceful fallbacks, never silent failure.
4. Keep functions small, typed, documented. Keep all thresholds in configs/default.yaml.
5. Do not ask me questions. Start now with step 1.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-04T20:11:35+05:30.

The user's current state is as follows:
No browser pages are currently open.
</ADDITIONAL_METADATA>