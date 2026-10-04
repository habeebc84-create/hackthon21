"""
pipeline/config.py
==================
Configuration loader and validator for SpaceTrack Flood.
Loads configs/default.yaml, calculates its SHA-256 hash for provenance,
and provides strongly typed access across all stages.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for exact provenance tracking."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class Config:
    def __init__(self, raw: Dict[str, Any], config_path: Path):
        self._raw = raw
        self.config_path = config_path
        self.config_hash = compute_file_hash(config_path)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found at: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(raw=raw, config_path=cfg_path)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Access nested keys using dot notation, e.g. 'detection.thresholds.conservative'"""
        parts = key_path.split(".")
        current = self._raw
        for p in parts:
            if isinstance(current, dict) and p in current:
                current = current[p]
            else:
                return default
        return current

    @property
    def raw(self) -> Dict[str, Any]:
        return self._raw
