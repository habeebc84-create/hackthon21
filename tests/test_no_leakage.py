"""
tests/test_no_leakage.py
========================
CI anti-leakage test enforcing competition hard rules:
1. Nothing in /pipeline or /model may import from /eval.
2. Nothing in /pipeline or /model may reference Copernicus EMS (EMSR927, EMSR*, etc.)
   or UNOSAT files or data.
3. OSM snapshot date must be strictly BEFORE the flood event date.
"""
import ast
import os
import re
from pathlib import Path
import pytest
from datetime import date


REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
MODEL_DIR = REPO_ROOT / "model"
EVAL_DIR = REPO_ROOT / "eval"

FORBIDDEN_KEYWORDS = [
    r"\bEMSR\d*\b",
    r"\bCopernicus\s*EMS\b",
    r"\bUNOSAT\b",
    r"\bemsr927\b",
    r"\bactivation\b.*EMSR",
]


def check_forbidden_imports(directory: Path):
    """Ensure no file in `directory` imports from eval."""
    violations = []
    if not directory.exists():
        return violations

    for py_file in directory.rglob("*.py"):
        if py_file.name == "__init__.py" and py_file.stat().st_size == 0:
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except Exception as e:
            violations.append(f"Failed to parse {py_file}: {e}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "eval" or alias.name.startswith("eval."):
                        violations.append(f"{py_file.name}:{node.lineno} imports '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "eval" or node.module.startswith("eval.") or node.module.startswith("..eval")):
                    violations.append(f"{py_file.name}:{node.lineno} imports from '{node.module}'")
    return violations


def check_forbidden_data_references(directory: Path):
    """Ensure no file in `directory` reads or references EMSR/UNOSAT files."""
    violations = []
    if not directory.exists():
        return violations

    for py_file in directory.rglob("*.py"):
        lines = py_file.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, start=1):
            # Allow attribution strings or comment disclaimers if explicitly mentioning "never use"
            if "never imported" in line.lower() or "forbidden" in line.lower():
                continue
            for pattern in FORBIDDEN_KEYWORDS:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(f"{py_file.name}:{idx} references forbidden string matching '{pattern}': {line.strip()}")
    return violations


def test_no_eval_imports_in_pipeline():
    """Rule 26: Nothing in /pipeline may import from /eval."""
    violations = check_forbidden_imports(PIPELINE_DIR)
    assert not violations, f"Pipeline contains forbidden imports from /eval:\n" + "\n".join(violations)


def test_no_eval_imports_in_model():
    """Rule 26: Nothing in /model may import from /eval."""
    violations = check_forbidden_imports(MODEL_DIR)
    assert not violations, f"Model contains forbidden imports from /eval:\n" + "\n".join(violations)


def test_no_emsr_or_unosat_in_pipeline():
    """Rule 26: Nothing in /pipeline may reference EMSR or UNOSAT data."""
    violations = check_forbidden_data_references(PIPELINE_DIR)
    assert not violations, f"Pipeline references forbidden EMSR/UNOSAT data:\n" + "\n".join(violations)


def test_no_emsr_or_unosat_in_model():
    """Rule 26: Nothing in /model may reference EMSR or UNOSAT data."""
    violations = check_forbidden_data_references(MODEL_DIR)
    assert not violations, f"Model references forbidden EMSR/UNOSAT data:\n" + "\n".join(violations)


def test_osm_snapshot_strictly_before_flood_date():
    """Rule 26: Hard-fail if the OSM snapshot date is on or after the flood date."""
    import yaml
    config_path = REPO_ROOT / "configs" / "default.yaml"
    assert config_path.exists(), "configs/default.yaml must exist"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    osm_snapshot_str = cfg.get("osm", {}).get("snapshot_date", "2026-07-27")
    osm_snapshot = date.fromisoformat(str(osm_snapshot_str))

    case_study_flood_date = date(2026, 8, 26)
    assert osm_snapshot < case_study_flood_date, (
        f"OSM snapshot date ({osm_snapshot}) must be strictly BEFORE the flood date ({case_study_flood_date})! "
        "Post-event edits in OSM represent severe data leakage."
    )
