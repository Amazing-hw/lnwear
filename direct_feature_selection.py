"""Validated direct Stage2 feature selection shared by extraction and training."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from stage2_feature_catalog import FEATURE_POOL_VERSION, is_model_candidate


DIRECT_CSV_COLUMNS = ["feature"]


def _validate_feature_names(feature_names):
    selected = [str(name).strip() for name in feature_names if str(name).strip()]
    if not selected:
        raise ValueError("direct feature selection is empty")
    duplicates = []
    seen = set()
    for name in selected:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError("direct feature selection contains duplicate features: " + ", ".join(duplicates))
    unknown = [name for name in selected if not is_model_candidate(name)]
    if unknown:
        raise ValueError("direct feature selection contains unknown features: " + ", ".join(unknown))
    return selected


def parse_direct_features(value):
    """Parse a comma-separated CLI selection while preserving user order."""
    return _validate_feature_names(str(value or "").split(","))


def write_direct_feature_csv(path, feature_names):
    """Write the canonical one-column direct-selection CSV."""
    path = Path(path)
    selected = _validate_feature_names(feature_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(DIRECT_CSV_COLUMNS)
        writer.writerows([[name] for name in selected])
    return path


def load_direct_feature_csv(path):
    """Load a canonical direct-selection CSV and preserve its exact row order."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("direct feature selection file must be CSV")
    if not path.exists():
        raise ValueError(f"direct feature selection file missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != DIRECT_CSV_COLUMNS:
            raise ValueError(
                "direct feature selection CSV must contain exactly one column: feature"
            )
        rows = list(reader)
    return _validate_feature_names(row.get("feature", "") for row in rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direct_selection_provenance(path, feature_names=None):
    """Return an auditable provenance record for a frozen direct selection."""
    path = Path(path).resolve()
    selected = load_direct_feature_csv(path) if feature_names is None else _validate_feature_names(feature_names)
    return {
        "feature_selection_mode": "direct",
        "selection_source_type": "csv",
        "feature_pool_version": FEATURE_POOL_VERSION,
        "direct_feature_file": str(path),
        "direct_feature_file_sha256": sha256_file(path),
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "feature_count_search": False,
        "local_swap_search": False,
    }
