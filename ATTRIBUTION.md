# Data Sources and Attributions

This project adheres strictly to open science and attribution requirements.

## Required Attributions

1. **Copernicus Sentinel-1 & Sentinel-2 Data**:
   > "Contains modified Copernicus Sentinel data 2026."
   Operated by the European Space Agency (ESA) under the European Union Copernicus Programme.

2. **Copernicus Digital Elevation Model (COP-DEM GLO-30)**:
   > "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."

3. **OpenStreetMap**:
   > "© OpenStreetMap contributors."
   Data retrieved via the HeiGIT ohsome API (`https://api.ohsome.org/v1`) using snapshot strictly prior to 2026-08-26 (snapshot date: `2026-07-27`).

4. **Training Datasets**:
   - **Kuro Siwo**:
     > Bountos, N. I., et al. (2024). *Kuro Siwo: A multi-sensor, multi-temporal benchmark dataset for global flood mapping*.
   - **Sen1Floods11**:
     > Bonafilia, D., et al. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1*. CVPR Workshops 2020.

5. **Validation Reference (Evaluation Only)**:
   - **Copernicus Emergency Management Service (EMS)**:
     > "European Union, Copernicus Emergency Management Service data." (EMSR927 - Trishuli River Flood, Nepal, August 2026).
     *Note: EMSR927 is strictly reserved for the `/eval` post-hoc evaluation suite and never accessed or leaked into the inference pipeline.*

---
**Disclaimer**:
*Educational prototype, not an operational tool. No victim imagery is used or collected.*
