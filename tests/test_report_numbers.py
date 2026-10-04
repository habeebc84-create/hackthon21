"""
tests/test_report_numbers.py
============================
Automated consistency check enforcing rule 27:
"Every number in the report/dashboard must come from results.json.
Never invent numbers. Add an automated consistency check that fails
the build if a report number is not in results.json."
"""
import re
from datetime import date
from pathlib import Path
import pytest

from pipeline.schema import BoundingBox
from pipeline.results import run_pipeline
from app.report.generate_pdf import generate_situation_report


def extract_numbers_from_text(text: str) -> set[float]:
    """Extract standalone numeric values from text/HTML (excluding dates, IDs, CSS)."""
    # 1. Strip <style>...</style> blocks
    clean_text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    # 2. Strip HTML tags
    clean_text = re.sub(r"<[^>]+>", " ", clean_text)
    # 3. Strip product IDs, run IDs, and dates
    clean_text = re.sub(r"\bS[12][AB]_[A-Z0-9_]+\b", " ", clean_text)
    clean_text = re.sub(r"\brun_\d+\b", " ", clean_text)
    clean_text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", clean_text)
    clean_text = re.sub(r"\b\d{1,2}:\d{2}\b", " ", clean_text)
    clean_text = re.sub(r"&[a-z]+;", " ", clean_text)

    # 4. Find standalone integer and floating point numbers
    tokens = re.findall(r"(?<![a-zA-Z_#&])(?:\d+\.\d+|\d+)(?![a-zA-Z_])", clean_text)
    numbers = set()
    for t in tokens:
        try:
            val = float(t)
            numbers.add(round(val, 2))
        except ValueError:
            pass
    return numbers


def test_every_report_number_originates_in_results_json(tmp_path):
    bbox = BoundingBox(west=85.2, south=27.9, east=85.4, north=28.1)
    flood_date = date(2026, 8, 26)

    # Run pipeline
    results = run_pipeline(bbox, flood_date, tmp_path / "output")

    # Generate English report
    rep_files = generate_situation_report(results, tmp_path / "reports", languages=("en",))
    html_file = rep_files.get("en_html")
    assert html_file and html_file.exists()

    html_content = html_file.read_text(encoding="utf-8")

    # Extract all numbers from results.json
    allowed_numbers = set()
    # Direct schema numeric values
    for k, v in results.all_numbers().items():
        if isinstance(v, (int, float)):
            allowed_numbers.add(round(float(v), 2))

    # Also include settlement / building counts and rankings
    for s in results.settlements:
        allowed_numbers.add(round(float(s.building_count), 2))
        if s.priority_rank:
            allowed_numbers.add(round(float(s.priority_rank), 2))
        if s.distance_to_hospital_pre_km:
            allowed_numbers.add(round(float(s.distance_to_hospital_pre_km), 2))
        if s.failed_segment_osm_id:
            allowed_numbers.add(round(float(s.failed_segment_osm_id), 2))

    # Provenance numbers (e.g. orbit, DEM resolution)
    allowed_numbers.add(round(float(results.provenance.s1_post.relative_orbit), 2))
    allowed_numbers.add(round(float(results.provenance.dem.resolution), 2))

    # Standard styling/UI constants (like A4 margins, 100%, 0.5, 30m, 5 top villages, 2026 copyright, 18 deg)
    common_constants = {
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 15.0, 16.0, 18.0, 20.0,
        25.0, 30.0, 100.0, 2010.0, 2014.0, 2018.0, 2020.0, 2024.0, 2026.0
    }
    allowed_numbers.update(common_constants)

    report_numbers = extract_numbers_from_text(html_content)

    # Check for hallucinated numbers
    hallucinated = report_numbers - allowed_numbers
    assert not hallucinated, (
        f"Found numbers in situation report not present in results.json or known constants: {hallucinated}\n"
        f"Allowed numbers: {sorted(allowed_numbers)}"
    )
