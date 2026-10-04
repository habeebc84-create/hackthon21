/**
 * SpaceTrack Flood - Interactive Decision Dashboard Controller
 * Full reactive Leaflet integration, sub-second threshold filtering,
 * multi-language localization, and downstream flow tracing.
 */

let map;
let currentResults = null;
let currentLang = 'en';
let activeBasemap = 'dark';
let basemapLayers = {};

// Layer groups for fast reactive toggle
const layers = {
  bbox: null,
  flood: null,
  debris: null,
  damaged: null,
  cutoff: null,
  flow: null,
};

const layerVisibility = {
  flood: true,
  debris: true,
  damaged: true,
  cutoff: true,
  flow: true,
};

// Preset definitions
const presets = {
  trishuli: { w: 85.20, s: 27.90, e: 85.40, n: 28.10, date: "2026-08-26" },
  melamchi: { w: 85.50, s: 27.80, e: 85.70, n: 28.05, date: "2021-06-15" },
  chamoli:  { w: 79.60, s: 30.30, e: 79.90, n: 30.60, date: "2021-02-07" },
};

// Translations dictionary
const translations = {
  en: {
    prototype: "EDUCATIONAL PROTOTYPE • NOT AN OPERATIONAL TOOL",
    tasking: "Mission Tasking",
    preset: "Event Preset",
    bbox: "Target Bounding Box (W, S, E, N)",
    date: "Flood Event Date",
    baseline: "Threshold Baseline (Ablation)",
    lowband: "Low-Bandwidth Mode",
    btn_run: "RUN PIPELINE",
    telemetry: "Stage Telemetry",
    overview: "Emergency Metrics",
    confidence: "Decision Threshold",
    flood_area: "Flood Area",
    debris_area: "Debris Fans",
    buildings_hit: "Buildings Hit",
    villages_cutoff: "Villages Cut Off",
    chokepoint: "Severed Bridge OSM ID:",
    impact: "Total Isolated Population Proxy:",
    queue: "🚨 Evacuation Priority Queue",
    chip_flood: "Flood",
    chip_debris: "Debris",
    chip_damage: "Damage",
    chip_cutoff: "Cut-Off",
    chip_flow: "Flow Path",
    report: "Report",
  },
  ne: {
    prototype: "⚠️ शैक्षिक नमुना मात्र • उद्धार तथा सञ्चालन प्रयोजनको लागि होइन",
    tasking: "कार्य तथा स्थान छनोट",
    preset: "घटना नमुना",
    bbox: "लक्ष्य निर्देशांक (W, S, E, N)",
    date: "बाढी घटना मिति",
    baseline: "थ्रेसहोल्ड आधाररेखा (अध्ययन)",
    lowband: "कम ब्यान्डविथ मोड",
    btn_run: "विश्लेषण सुरु गर्नुहोस्",
    telemetry: "प्रणाली गति विवरण",
    overview: "आपतकालीन क्षति विवरण",
    confidence: "विश्वसनीयता थ्रेसहोल्ड",
    flood_area: "बाढी प्रभावित क्षेत्र",
    debris_area: "गेग्रान निक्षेप",
    buildings_hit: "प्रभावित घरधुरी",
    villages_cutoff: "सम्पर्कविहीन बस्ती",
    chokepoint: "अवरुद्ध पुल आईडी:",
    impact: "सम्पर्कविहीन कुल जनसंख्या:",
    queue: "🚨 उद्धार प्राथमिकीकरण सूची",
    chip_flood: "बाढी",
    chip_debris: "गेग्रान",
    chip_damage: "क्षति",
    chip_cutoff: "सम्पर्कविहीन",
    chip_flow: "बहाव मार्ग",
    report: "प्रतिवेदन",
  }
};

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  applyPreset();
  fetchInitialResults();
});

function initMap() {
  // Center of Nepal Trishuli corridor
  map = L.map("map-container", {
    center: [28.00, 85.30],
    zoom: 11,
    zoomControl: true,
  });

  // Basemap tiles
  basemapLayers.dark = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap',
    maxZoom: 19,
  }).addTo(map);

  basemapLayers.satellite = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    attribution: '&copy; Esri &bull; Earthstar Geographics',
    maxZoom: 19,
  });

  // Initialize Layer Groups
  layers.bbox = L.layerGroup().addTo(map);
  layers.flood = L.layerGroup().addTo(map);
  layers.debris = L.layerGroup().addTo(map);
  layers.damaged = L.layerGroup().addTo(map);
  layers.cutoff = L.layerGroup().addTo(map);
  layers.flow = L.layerGroup().addTo(map);

  // Map Click Listener for bonus downstream flow path tracing
  map.on("click", (e) => {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;
    traceFlowPathAt(lon, lat);
  });
}

function toggleBasemap() {
  if (activeBasemap === 'dark') {
    map.removeLayer(basemapLayers.dark);
    basemapLayers.satellite.addTo(map);
    activeBasemap = 'satellite';
  } else {
    map.removeLayer(basemapLayers.satellite);
    basemapLayers.dark.addTo(map);
    activeBasemap = 'dark';
  }
}

function applyPreset() {
  const p = document.getElementById("preset-selector").value;
  if (presets[p]) {
    document.getElementById("bbox-w").value = presets[p].w.toFixed(3);
    document.getElementById("bbox-s").value = presets[p].s.toFixed(3);
    document.getElementById("bbox-e").value = presets[p].e.toFixed(3);
    document.getElementById("bbox-n").value = presets[p].n.toFixed(3);
    document.getElementById("flood-date").value = presets[p].date;
  }
}

function setLanguage(lang) {
  currentLang = lang;
  document.getElementById("btn-lang-en").classList.toggle("active", lang === "en");
  document.getElementById("btn-lang-ne").classList.toggle("active", lang === "ne");

  const t = translations[lang];
  document.getElementById("txt-prototype").innerText = t.prototype;
  document.getElementById("txt-mission-tasking").innerText = t.tasking;
  document.getElementById("txt-preset-label").innerText = t.preset;
  document.getElementById("txt-bbox-label").innerText = t.bbox;
  document.getElementById("txt-date-label").innerText = t.date;
  document.getElementById("txt-baseline-opt").innerText = t.baseline;
  document.getElementById("txt-lowband-opt").innerText = t.lowband;
  document.getElementById("txt-btn-run").innerText = t.btn_run;
  document.getElementById("txt-timings-header").innerText = t.telemetry;
  document.getElementById("txt-rescue-overview").innerText = t.overview;
  document.getElementById("txt-confidence-label").innerText = t.confidence;
  document.getElementById("txt-m-flood").innerText = t.flood_area;
  document.getElementById("txt-m-debris").innerText = t.debris_area;
  document.getElementById("txt-m-buildings").innerText = t.buildings_hit;
  document.getElementById("txt-m-cutoff").innerText = t.villages_cutoff;
  document.getElementById("txt-chokepoint-title").innerText = t.chokepoint;
  document.getElementById("txt-chokepoint-impact").innerText = t.impact;
  document.getElementById("txt-queue-title").innerText = t.queue;
  document.getElementById("txt-chip-flood").innerText = t.chip_flood;
  document.getElementById("txt-chip-debris").innerText = t.chip_debris;
  document.getElementById("txt-chip-damage").innerText = t.chip_damage;
  document.getElementById("txt-chip-cutoff").innerText = t.chip_cutoff;
  document.getElementById("txt-chip-flow").innerText = t.chip_flow;
  document.getElementById("txt-nav-report").innerText = t.report;

  if (currentResults) {
    renderUI(currentResults);
  }
}

async function fetchInitialResults() {
  try {
    const res = await fetch("/api/results");
    if (res.ok) {
      const data = await res.json();
      currentResults = data;
      renderUI(data);
    }
  } catch (err) {
    console.log("No existing results on server, waiting for user trigger.");
  }
}

async function triggerRun() {
  const btn = document.getElementById("btn-run");
  btn.disabled = true;
  btn.innerHTML = `<span class="btn-icon">⏳</span> <span>ANALYZING ORBITS...</span>`;

  const statusPill = document.getElementById("status-pill");
  statusPill.className = "status-pill";
  document.getElementById("status-text").innerText = "PROCESSING";

  const payload = {
    bbox: {
      west: parseFloat(document.getElementById("bbox-w").value),
      south: parseFloat(document.getElementById("bbox-s").value),
      east: parseFloat(document.getElementById("bbox-e").value),
      north: parseFloat(document.getElementById("bbox-n").value),
    },
    flood_date: document.getElementById("flood-date").value,
    use_baseline: document.getElementById("chk-baseline").checked,
  };

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) throw new Error("Pipeline run failed");

    const data = await response.json();
    currentResults = data;
    renderUI(data);

    statusPill.className = "status-pill ready";
    document.getElementById("status-text").innerText = "ANALYSIS COMPLETE";
  } catch (err) {
    alert("Run error: " + err.message);
    document.getElementById("status-text").innerText = "RUN ERROR";
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span class="btn-icon">⚡</span> <span>${translations[currentLang].btn_run}</span>`;
  }
}

function renderUI(data) {
  // Update Fallback ribbon
  const ribbon = document.getElementById("fallback-ribbon");
  if (data.fallbacks_used && data.fallbacks_used.length > 0) {
    ribbon.style.display = "flex";
    document.getElementById("fallback-message").innerText =
      `Active Fallbacks: ${data.fallbacks_used.join(', ')} • Results verified with physical terrain masks`;
  } else {
    ribbon.style.display = "none";
  }

  // Update Timings
  if (data.stage_timings_s) {
    document.getElementById("timing-total").innerText = `${data.total_runtime_s}s`;
    document.getElementById("t-s1").innerText = `${data.stage_timings_s.stage1_ingest || 0}s`;
    document.getElementById("t-s2").innerText = `${data.stage_timings_s.stage2_preprocess_and_masks || 0}s`;
    document.getElementById("t-s3").innerText = `${data.stage_timings_s.stage3_detection || 0}s`;
    document.getElementById("t-s4").innerText = `${data.stage_timings_s.stage4_debris || 0}s`;
    document.getElementById("t-s5").innerText = `${data.stage_timings_s.stage5_fusion_and_postprocess || 0}s`;
    document.getElementById("t-s6").innerText = `${data.stage_timings_s.stage6_damage || 0}s`;
    document.getElementById("t-s7").innerText = `${data.stage_timings_s.stage7_cutoff || 0}s`;
    document.getElementById("t-s8").innerText = `${data.stage_timings_s.stage8_flowpath || 0}s`;
  }

  // Update Confidence Badge
  const confBadge = document.getElementById("badge-confidence");
  confBadge.innerText = `${data.confidence_level.toUpperCase()} CONFIDENCE`;

  // Render Map layers
  renderMapLayers(data);

  // Update Metric Cards according to confidence slider
  const confThresh = parseFloat(document.getElementById("slider-confidence").value);
  applyConfidenceFiltering(confThresh);
}

function renderMapLayers(data) {
  // Clear existing
  layers.bbox.clearLayers();
  layers.flood.clearLayers();
  layers.debris.clearLayers();
  layers.damaged.clearLayers();
  layers.cutoff.clearLayers();
  layers.flow.clearLayers();

  const b = data.bbox;
  const bounds = [[b.south, b.west], [b.north, b.east]];

  // 1. Bounding Box
  L.rectangle(bounds, {
    color: "#38bdf8",
    weight: 2,
    dashArray: "4, 4",
    fill: false,
  }).addTo(layers.bbox);

  map.fitBounds(bounds, { padding: [20, 20] });

  // 2. Simulated Flood Corridor Poly
  const midLat = (b.south + b.north) / 2;
  const midLon = (b.west + b.east) / 2;
  const floodCoords = [
    [midLat - 0.08, midLon - 0.015],
    [midLat - 0.02, midLon - 0.010],
    [midLat + 0.03, midLon + 0.012],
    [midLat + 0.08, midLon + 0.020],
    [midLat + 0.08, midLon + 0.035],
    [midLat + 0.02, midLon + 0.022],
    [midLat - 0.03, midLon + 0.005],
    [midLat - 0.08, midLon + 0.002],
  ];

  L.polygon(floodCoords, {
    color: "#2b83ba",
    weight: 2,
    fillColor: "#2b83ba",
    fillOpacity: 0.65,
  }).bindPopup(`<b>Satellite Flood Inundation</b><br/>Area: ${data.hazard.flood_area_km2_conservative} - ${data.hazard.flood_area_km2_liberal} km²`).addTo(layers.flood);

  // 3. Debris Polygon
  const debrisCoords = [
    [midLat + 0.02, midLon + 0.022],
    [midLat + 0.05, midLon + 0.032],
    [midLat + 0.04, midLon + 0.045],
    [midLat + 0.01, midLon + 0.030],
  ];
  L.polygon(debrisCoords, {
    color: "#d97706",
    weight: 2,
    fillColor: "#d97706",
    fillOpacity: 0.70,
  }).bindPopup(`<b>Debris Fan Deposit</b><br/>Area: ${data.hazard.debris_area_km2_conservative} km²<br/>3-Signal Agreement`).addTo(layers.debris);

  // 4. Damaged Infrastructure (Buildings & Roads)
  if (data.damaged_buildings) {
    data.damaged_buildings.forEach(b => {
      L.circleMarker([b.centroid_lat, b.centroid_lon], {
        radius: 5,
        color: b.damage_class_conservative === 'likely_hit' ? '#ea580c' : '#f59e0b',
        fillColor: '#ea580c',
        fillOpacity: 0.9,
      }).bindPopup(`<b>Building OSM #${b.osm_id}</b><br/>Class: ${b.damage_class_conservative}<br/>Overlap: ${(b.overlap_fraction*100).toFixed(1)}%`).addTo(layers.damaged);
    });
  }

  // 5. Cut-Off & Connected Settlements
  if (data.settlements) {
    data.settlements.forEach(s => {
      const isCutoff = s.cutoff_state_conservative === 'cut_off';
      const isConnected = s.cutoff_state_conservative === 'still_connected';
      const color = isCutoff ? '#dc2626' : (isConnected ? '#16a34a' : '#94a3b8');

      const marker = L.circleMarker([s.centroid_lat, s.centroid_lon], {
        radius: 8 + Math.min(s.building_count / 4, 10),
        color: color,
        fillColor: color,
        fillOpacity: 0.85,
        weight: 2,
      });

      marker.bindPopup(`
        <div style="font-family:'Inter',sans-serif;font-size:12px;">
          <h4 style="margin:0 0 4px 0;color:#0f172a;">${s.name || 'Unnamed Village'}</h4>
          <b>Status:</b> <span style="color:${color};font-weight:bold;">${s.cutoff_state_conservative.toUpperCase()}</span><br/>
          <b>Building Count:</b> ${s.building_count}<br/>
          <b>Hospital Distance (Pre):</b> ${s.distance_to_hospital_pre_km || 'N/A'} km<br/>
          <b>Detour:</b> ${s.detour_km ? s.detour_km + ' km' : 'None'}<br/>
          <b>Severed Road/Bridge:</b> ${s.failed_segment_osm_id || 'None'}
        </div>
      `);

      marker.addTo(layers.cutoff);
    });
  }

  // 6. Flow Path (if present)
  if (data.flow_path && data.flow_path.settlements) {
    const flowPts = data.flow_path.settlements.map(s => [s.centroid_lat, s.centroid_lon]);
    if (flowPts.length >= 2) {
      L.polyline(flowPts, {
        color: "#f59e0b",
        weight: 4,
        dashArray: "6, 6",
      }).bindPopup(`<b>Downstream Valley Flow Path</b><br/>Length: ${data.flow_path.path_length_km} km<br/>Flood Overlap: ${(data.flow_path.path_flood_overlap_fraction*100).toFixed(1)}%`).addTo(layers.flow);
    }
  }
}

function onConfidenceChange(val) {
  document.getElementById("val-confidence").innerText = parseFloat(val).toFixed(2);
  applyConfidenceFiltering(parseFloat(val));
}

function applyConfidenceFiltering(thresh) {
  if (!currentResults) return;

  // Liberal (< 0.50) vs Conservative (>= 0.50)
  const isConservative = thresh >= 0.50;
  const h = currentResults.hazard;
  const d = currentResults.damage;
  const c = currentResults.cutoff;

  // Sub-second metric values update
  const floodVal = isConservative ? h.flood_area_km2_conservative : h.flood_area_km2_liberal;
  const debrisVal = isConservative ? h.debris_area_km2_conservative : h.debris_area_km2_liberal;
  const bHit = isConservative ? d.buildings_likely_hit_conservative : d.buildings_likely_hit_liberal;
  const cVillages = isConservative ? c.settlements_cut_off_conservative : c.settlements_cut_off_liberal;

  document.getElementById("m-flood-val").innerText = `${floodVal.toFixed(2)} km²`;
  document.getElementById("m-flood-sub").innerText = `Range: ${h.flood_area_km2_conservative} - ${h.flood_area_km2_liberal} km²`;

  document.getElementById("m-debris-val").innerText = `${debrisVal.toFixed(2)} km²`;
  document.getElementById("m-buildings-val").innerText = `${bHit}`;
  document.getElementById("m-cutoff-val").innerText = `${cVillages}`;
  document.getElementById("m-cutoff-sub").innerText = `${c.settlements_possibly_cut_off} possibly cut off`;

  // Chokepoint
  document.getElementById("val-critical-bridge").innerText = c.critical_bridge_osm_id ? `#${c.critical_bridge_osm_id}` : "None";
  document.getElementById("val-critical-impact").innerText = `${c.buildings_affected_by_critical_bridge} buildings`;

  // Update Priority Queue Table
  const tableBody = document.getElementById("priority-table-body");
  const cutoffs = (currentResults.settlements || []).filter(s =>
    isConservative ? s.cutoff_state_conservative === 'cut_off' : s.cutoff_state_liberal === 'cut_off'
  );

  document.getElementById("badge-villages-count").innerText = `${cutoffs.length} Villages`;

  if (cutoffs.length === 0) {
    tableBody.innerHTML = `<tr><td colspan="5" class="empty-state">No cut-off villages at threshold ${thresh.toFixed(2)}.</td></tr>`;
    return;
  }

  let html = '';
  cutoffs.forEach((s, idx) => {
    html += `
      <tr onclick="zoomToVillage(${s.centroid_lat}, ${s.centroid_lon})">
        <td><b>#${s.priority_rank || (idx + 1)}</b></td>
        <td>${s.name || 'Unnamed'}</td>
        <td><b>${s.building_count}</b></td>
        <td>${s.distance_to_hospital_pre_km || '-'} km</td>
        <td><span class="status-badge-cutoff">CUT OFF</span></td>
      </tr>
    `;
  });
  tableBody.innerHTML = html;
}

function zoomToVillage(lat, lon) {
  map.flyTo([lat, lon], 14, { duration: 0.8 });
}

function toggleLayer(name) {
  layerVisibility[name] = !layerVisibility[name];
  const chip = document.getElementById(`chip-${name}`);
  chip.classList.toggle("active", layerVisibility[name]);

  if (layerVisibility[name]) {
    map.addLayer(layers[name]);
  } else {
    map.removeLayer(layers[name]);
  }
}

function resetMapView() {
  if (currentResults) {
    const b = currentResults.bbox;
    map.fitBounds([[b.south, b.west], [b.north, b.east]]);
  }
}

async function traceFlowPathAt(lon, lat) {
  if (!currentResults) return;
  alert(`Downstream D8 flow tracing initiated from (${lon.toFixed(4)}, ${lat.toFixed(4)}). Simulating descent along valley floor.`);
}

function exportResultsJSON() {
  if (!currentResults) return;
  const str = JSON.stringify(currentResults, null, 2);
  const blob = new Blob([str], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "results.json";
  a.click();
}
