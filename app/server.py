"""
app/server.py
=============
Lightweight, high-performance web dashboard server.
Uses Python's standard library HTTP server with full REST JSON API:
  - Serves static assets (HTML, CSS, JS, Leaflet maps)
  - POST /api/run: executes SpaceTrack Flood pipeline and returns live results.json
  - GET /api/results: returns active results
  - GET /reports: renders the single-page situation report
Runs without needing extra server packages.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.schema import BoundingBox
from pipeline.results import run_pipeline
from app.report.generate_pdf import generate_situation_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("spacetrack.server")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUTS_DIR = BASE_DIR.parent / "outputs"


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(STATIC_DIR / "index.html", "rb") as f:
                self.wfile.write(f.read())
            return

        elif parsed.path == "/static/index.css":
            self.send_response(200)
            self.send_header("Content-Type", "text/css; charset=utf-8")
            self.end_headers()
            with open(STATIC_DIR / "index.css", "rb") as f:
                self.wfile.write(f.read())
            return

        elif parsed.path == "/static/app.js":
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.end_headers()
            with open(STATIC_DIR / "app.js", "rb") as f:
                self.wfile.write(f.read())
            return

        elif parsed.path == "/api/results":
            res_file = OUTPUTS_DIR / "results.json"
            if res_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                with open(res_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "No results generated yet"}')
            return

        elif parsed.path == "/reports":
            lang = parse_qs(parsed.query).get("lang", ["en"])[0]
            rep_file = OUTPUTS_DIR / "reports" / f"situation_report_{lang}.html"
            if not rep_file.exists():
                rep_file = OUTPUTS_DIR / "reports" / "situation_report_en.html"

            if rep_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(rep_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.wfile.write(b"Report not yet generated. Run the pipeline first.")
            return

        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/run":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            req = json.loads(body.decode("utf-8"))

            b = req.get("bbox", {})
            bbox = BoundingBox(
                west=float(b.get("west", 85.2)),
                south=float(b.get("south", 27.9)),
                east=float(b.get("east", 85.4)),
                north=float(b.get("north", 28.1)),
            )
            flood_dt = date.fromisoformat(req.get("flood_date", "2026-08-26"))
            use_baseline = bool(req.get("use_baseline", False))

            # Run pipeline
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            results = run_pipeline(
                bbox=bbox,
                flood_date=flood_dt,
                output_dir=OUTPUTS_DIR,
                upstream_point=(bbox.west + 0.1, bbox.north - 0.05),
                use_baseline=use_baseline,
            )

            # Generate situation report
            generate_situation_report(results, OUTPUTS_DIR / "reports")

            res_json = results.model_dump_json(indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(res_json)
            return

        self.send_response(404)
        self.end_headers()


def start_server(port: int = 8000):
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    logger.info(f"SpaceTrack Flood Dashboard live at http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    start_server(8000)
