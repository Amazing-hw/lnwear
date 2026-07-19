"""CSV-only manual Stage2 feature-selection contract."""

from __future__ import annotations

import csv
import hashlib
import math
import numbers
from pathlib import Path

from stage2_feature_catalog import FEATURE_POOL_VERSION, feature_record, is_model_candidate


CSV_SCHEMA_VERSION = 1
CONTRACT_COLUMNS = [
    "csv_schema_version", "feature_pool_version", "ranking_sha256",
]
SELECTION_COLUMNS = [
    "selected", *CONTRACT_COLUMNS, "rank", "feature", "eligible", "group",
    "commercial_8_member", "commercial_original_name", "ranking_score",
    "train_group_fold_auc_mean", "valid_auc", "fp_proxy_sample_fp_rate",
    "valid_psi", "deployment_cost", "signal_source", "preprocessing", "unit",
    "formula", "fft", "buffer_samples", "accumulator", "c_operators",
    "risk_flags", "ineligible_reasons",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ranking(path: Path) -> dict:
    import json

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("ranking"), list):
        raise ValueError(f"{path} must contain a ranking array")
    if payload.get("feature_pool_version") != FEATURE_POOL_VERSION:
        raise ValueError(
            f"ranking feature_pool_version must be {FEATURE_POOL_VERSION}; rerun s03-s04"
        )
    return payload


def _row_for_feature(item: dict, ranking_sha256: str) -> dict:
    name = str(item["feature"])
    record = feature_record(name)
    return {
        "selected": 0,
        "csv_schema_version": CSV_SCHEMA_VERSION,
        "feature_pool_version": FEATURE_POOL_VERSION,
        "ranking_sha256": ranking_sha256,
        "rank": int(item.get("rank", 0)),
        "feature": name,
        "eligible": int(bool(item.get("eligible_for_manual_selection", False))),
        "group": str(record["group"]),
        "commercial_8_member": int(bool(record["commercial_8_member"])),
        "commercial_original_name": record["commercial_original_name"] or "",
        "ranking_score": item.get("ranking_score"),
        "train_group_fold_auc_mean": item.get("train_group_fold_auc_mean"),
        "valid_auc": item.get("valid_auc"),
        "fp_proxy_sample_fp_rate": item.get("fp_proxy_sample_fp_rate"),
        "valid_psi": item.get("valid_psi"),
        "deployment_cost": float(record["deployment_cost"]),
        "signal_source": str(record["signal_source"]),
        "preprocessing": str(record["preprocessing"]),
        "unit": str(record["unit"]),
        "formula": str(record["formula"]),
        "fft": int(bool(record["fft"])),
        "buffer_samples": int(record["buffer_samples"]),
        "accumulator": str(record["accumulator"]),
        "c_operators": ", ".join(map(str, record["c_operators"])),
        "risk_flags": ", ".join(map(str, item.get("risk_flags") if item.get("risk_flags") is not None else record.get("risk_flags", []))),
        "ineligible_reasons": ", ".join(map(str, item.get("ineligible_reasons") or [])),
    }


def export_manual_selection_csv(ranking_path, output_dir) -> Path:
    """Write the only user-editable manual-selection artifact."""
    ranking_path = Path(ranking_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_ranking(ranking_path)
    ranking_sha = _sha256(ranking_path)
    rows = [_row_for_feature(item, ranking_sha) for item in payload["ranking"]]
    csv_path = output_dir / "manual_feature_selection.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _contract_cells_equal(actual, expected) -> bool:
    if expected is None or expected == "":
        return actual in (None, "")
    if isinstance(expected, bool):
        expected = int(expected)
    if isinstance(expected, numbers.Real):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def load_manual_selection_csv(path, ranking_path, train_columns, valid_columns):
    """Validate a CSV where only the ``selected`` column may be edited."""
    path = Path(path).resolve()
    ranking_path = Path(ranking_path).resolve()
    if path.suffix.lower() != ".csv":
        raise ValueError("manual feature selection is CSV-only; rerun s04 and edit manual_feature_selection.csv")

    payload = _load_ranking(ranking_path)
    ranking_sha = _sha256(ranking_path)
    expected_rows = [_row_for_feature(item, ranking_sha) for item in payload["ranking"]]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SELECTION_COLUMNS:
            raise ValueError(
                f"manual selection CSV columns changed; expected immutable order {SELECTION_COLUMNS}"
            )
        rows = list(reader)
    if len(rows) != len(expected_rows):
        raise ValueError(
            f"manual selection CSV row count changed; expected {len(expected_rows)} governed features"
        )
    if any(row.get("ranking_sha256") != ranking_sha for row in rows):
        raise ValueError("manual selection CSV ranking SHA256 mismatch; rerun s04")

    selected = []
    errors = []
    train_columns = set(train_columns)
    valid_columns = set(valid_columns)
    for row_number, (row, expected) in enumerate(zip(rows, expected_rows), start=2):
        for field in SELECTION_COLUMNS:
            if field == "selected":
                continue
            if not _contract_cells_equal(row.get(field), expected[field]):
                errors.append(
                    f"CSV row {row_number} field {field} for {expected['feature']} "
                    "changed immutable contract value"
                )
        selected_value = row.get("selected")
        if selected_value not in ("0", "1"):
            errors.append(f"CSV row {row_number} selected must be 0 or 1")
            continue
        if selected_value == "0":
            continue
        name = str(row.get("feature") or "").strip()
        if name in selected:
            errors.append(f"CSV row {row_number} duplicate feature {name}")
            continue
        item = payload["ranking"][row_number - 2]
        if not is_model_candidate(name):
            errors.append(f"CSV row {row_number} unknown feature {name}")
            continue
        if not bool(item.get("eligible_for_manual_selection", False)):
            errors.append(f"CSV row {row_number} ineligible feature {name}")
        if name not in train_columns or name not in valid_columns:
            errors.append(f"CSV row {row_number} feature {name} missing from train/valid")
        selected.append(name)
    if errors:
        raise ValueError("manual CSV validation failed: " + "; ".join(errors))
    if not selected:
        raise ValueError("manual CSV has empty selection")

    selected_records = [feature_record(name) for name in selected]
    fft_sources = sorted({str(record["signal_source"]) for record in selected_records if record["fft"]})
    warnings = [
        f"selected_feature_count={len(selected)} (no selection limit)",
        f"fft_sources={fft_sources or []} (engineering warning only)",
        f"max_buffer_samples={max(int(record['buffer_samples']) for record in selected_records)}",
    ]
    if "mode" in selected:
        warnings.append("mode_selected=true (audit subject/device/session/mode generalization)")
    provenance = {
        "feature_selection_mode": "manual",
        "selection_source_type": "csv",
        "csv_schema_version": CSV_SCHEMA_VERSION,
        "feature_pool_version": FEATURE_POOL_VERSION,
        "manual_feature_file": str(path),
        "manual_feature_file_sha256": _sha256(path),
        "ranking_source": str(ranking_path),
        "ranking_source_sha256": ranking_sha,
        "engineering_warnings": warnings,
    }
    return selected, provenance
