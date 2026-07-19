"""Shared publication-grade PNG, source-data, manifest, and QA export."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


SCIENTIFIC_DPI = 600
PALETTE = {
    "negative": "#4C78A8",
    "positive": "#E07B53",
    "selected": "#2A9D8F",
    "neutral": "#7A7A7A",
    "warning": "#C89B3C",
    "danger": "#B64C4C",
}


def apply_scientific_theme():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "legend.frameon": False,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qa_png(path: Path, expected_size, dpi):
    with Image.open(path) as image:
        width, height = image.size
        actual_dpi = image.info.get("dpi", (0.0, 0.0))
        grey = np.asarray(image.convert("L"), dtype=np.uint8)
    nonblank = bool(grey.size and int(grey.max()) > int(grey.min()))
    size_pass = (width, height) == tuple(expected_size)
    dpi_pass = all(abs(float(value) - float(dpi)) <= max(2.0, 0.01 * dpi) for value in actual_dpi[:2])
    return {
        "passed": bool(nonblank and size_pass and dpi_pass),
        "nonblank": nonblank,
        "size_pass": size_pass,
        "dpi_pass": dpi_pass,
        "pixel_width": int(width),
        "pixel_height": int(height),
        "dpi": [float(value) for value in actual_dpi[:2]],
    }


def save_scientific_figure(
        fig,
        output_path,
        *,
        source_data,
        source_data_path=None,
        core_conclusion,
        panel_map,
        inputs=(),
        split,
        n_definition,
        statistics=None,
        reviewer_risks=(),
        test_read_only=False,
        dpi=SCIENTIFIC_DPI):
    """Write one mandatory PNG/CSV/manifest/QA artifact quartet."""
    if source_data is None:
        raise ValueError("source_data is required for every scientific figure")
    if str(split) == "test" and test_read_only is not True:
        raise ValueError("test_read_only must be true for test figures")
    if not str(core_conclusion).strip():
        raise ValueError("core_conclusion is required")
    if not isinstance(panel_map, dict) or not panel_map:
        raise ValueError("panel_map is required")

    apply_scientific_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = (
        Path(source_data_path)
        if source_data_path is not None
        else output_path.with_name(output_path.stem + "_source_data.csv")
    )
    source_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_name(output_path.stem + "_figure_manifest.json")
    qa_path = output_path.with_name(output_path.stem + "_figure_qa.json")
    source_frame = source_data.copy() if isinstance(source_data, pd.DataFrame) else pd.DataFrame(source_data)
    source_frame.to_csv(source_path, index=False)

    width_inches, height_inches = fig.get_size_inches()
    expected_size = (
        int(round(float(width_inches) * int(dpi))),
        int(round(float(height_inches) * int(dpi))),
    )
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.1}):
        fig.savefig(
            output_path,
            dpi=int(dpi),
            facecolor="white",
            edgecolor="none",
            bbox_inches=None,
            metadata={"Software": "wearing_liveness scientific_figures", "dpi": str(int(dpi))},
        )
    input_records = []
    for input_path in inputs:
        path = Path(input_path).resolve()
        input_records.append({
            "path": str(path),
            "sha256": _sha256(path) if path.is_file() else None,
        })
    manifest = {
        "schema_version": 1,
        "core_conclusion": str(core_conclusion),
        "panel_map": {str(key): str(value) for key, value in panel_map.items()},
        "split": str(split),
        "test_read_only": bool(test_read_only),
        "n_definition": str(n_definition),
        "statistics": dict(statistics or {}),
        "reviewer_risks": [str(value) for value in reviewer_risks],
        "inputs": input_records,
        "source_data": str(source_path.resolve()),
        "png": str(output_path.resolve()),
        "dpi": int(dpi),
        "figure_size_inches": [float(width_inches), float(height_inches)],
        "expected_pixel_size": list(expected_size),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    qa = _qa_png(output_path, expected_size, int(dpi))
    qa["source_data_exists"] = source_path.exists()
    qa["manifest_exists"] = manifest_path.exists()
    qa["passed"] = bool(qa["passed"] and qa["source_data_exists"] and qa["manifest_exists"])
    with qa_path.open("w", encoding="utf-8") as handle:
        json.dump(qa, handle, indent=2, ensure_ascii=False)
    if not qa["passed"]:
        raise RuntimeError(f"scientific figure QA failed for {output_path}: {qa}")
    return {
        "png": output_path,
        "source_data": source_path,
        "manifest": manifest_path,
        "qa": qa_path,
    }


apply_scientific_theme()


def _read_json_if_present(path: Path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def export_pipeline_scientific_overview(artifact_dir):
    """Export an asymmetric overview of the frozen pipeline evidence."""
    artifact_dir = Path(artifact_dir)
    report_dir = artifact_dir / "report_plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    completeness_path = artifact_dir / "feature_pool_completeness.json"
    selection_path = artifact_dir / "manual_selected_features.json"
    model_path = artifact_dir / "model_candidate_leaderboard.json"
    hard_negative_path = artifact_dir / "hard_negative_decision.json"
    postprocess_path = artifact_dir / "postprocess_opt" / "postprocess_optimized.json"
    test_paths = sorted(artifact_dir.glob("end_to_end_eval_test_*.json"))
    completeness = _read_json_if_present(completeness_path) or {}
    selection = _read_json_if_present(selection_path) or {}
    model = _read_json_if_present(model_path) or {}
    hard_negative = _read_json_if_present(hard_negative_path) or {}
    postprocess = _read_json_if_present(postprocess_path) or {}
    test_payload = _read_json_if_present(test_paths[0]) if test_paths else {}
    test_payload = test_payload or {}

    selected_model_name = model.get("selected_candidate")
    selected_model = next(
        (
            item for item in model.get("leaderboard", [])
            if item.get("candidate") == selected_model_name
        ),
        {},
    )
    post_metrics = postprocess.get("metrics", {}) or {}
    rows = [
        {"section": "feature_pool", "metric": "catalog_count", "value": completeness.get("catalog_count", np.nan), "status": "available" if completeness else "unavailable"},
        {"section": "feature_pool", "metric": "ranked_count", "value": completeness.get("ranked_count", np.nan), "status": "available" if completeness else "unavailable"},
        {"section": "selection", "metric": "selected_count", "value": len(selection.get("selected_features", [])) if selection else np.nan, "status": "available" if selection else "unavailable"},
        {"section": "model", "metric": "valid_accuracy", "value": selected_model.get("valid_accuracy", np.nan), "status": "available" if selected_model else "unavailable"},
        {"section": "model", "metric": "valid_fp_rate", "value": selected_model.get("valid_fp_rate", np.nan), "status": "available" if selected_model else "unavailable"},
        {"section": "model", "metric": "total_nodes", "value": selected_model.get("total_nodes", np.nan), "status": "available" if selected_model else "unavailable"},
        {"section": "hard_negative", "metric": "accepted", "value": int(bool(hard_negative.get("accepted"))) if hard_negative else np.nan, "status": "available" if hard_negative else "unavailable"},
        {"section": "hard_negative", "metric": "valid_accuracy_delta", "value": hard_negative.get("valid_accuracy_delta", np.nan), "status": "available" if hard_negative else "unavailable"},
        {"section": "hard_negative", "metric": "valid_fp_rate_delta", "value": hard_negative.get("valid_fp_rate_delta", np.nan), "status": "available" if hard_negative else "unavailable"},
        {"section": "postprocess", "metric": "window_accuracy", "value": post_metrics.get("window_accuracy", np.nan), "status": "available" if post_metrics else "unavailable"},
        {"section": "postprocess", "metric": "window_fp_rate", "value": post_metrics.get("window_fp_rate", np.nan), "status": "available" if post_metrics else "unavailable"},
        {"section": "postprocess", "metric": "added_latency_p95_sec", "value": post_metrics.get("first_worn_output_p95_sec", np.nan), "status": "available" if post_metrics else "unavailable"},
        {"section": "test", "metric": "sample_accuracy", "value": test_payload.get("summary", {}).get("accuracy", np.nan), "status": "read_only" if test_payload else "unavailable"},
        {"section": "test", "metric": "window_model_accuracy", "value": test_payload.get("window_model_summary", {}).get("accuracy", np.nan), "status": "read_only" if test_payload else "unavailable"},
        {"section": "test", "metric": "window_stream_accuracy", "value": test_payload.get("window_stream_summary", {}).get("accuracy", np.nan), "status": "read_only" if test_payload else "unavailable"},
    ]
    source = pd.DataFrame(rows)

    fig = plt.figure(figsize=(11.0, 6.5), facecolor="white")
    grid = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0], hspace=0.36, wspace=0.35)
    ax_hero = fig.add_subplot(grid[:, 0])
    ax_pool = fig.add_subplot(grid[0, 1])
    ax_model = fig.add_subplot(grid[0, 2])
    ax_post = fig.add_subplot(grid[1, 1:])

    hero_metrics = [
        ("Valid\naccuracy", selected_model.get("valid_accuracy", np.nan), 1.0),
        ("Valid\nFPR", selected_model.get("valid_fp_rate", np.nan), 0.01),
        ("Added P95\nlatency (s)", post_metrics.get("first_worn_output_p95_sec", np.nan), 3.0),
        ("Read-only test\naccuracy", test_payload.get("summary", {}).get("accuracy", np.nan), 1.0),
    ]
    for idx, (label, value, target) in enumerate(hero_metrics):
        y = len(hero_metrics) - 1 - idx
        ax_hero.text(0.0, y, label, fontsize=9, va="center")
        rendered = "unavailable" if not np.isfinite(value) else f"{float(value):.3f}"
        color = PALETTE["neutral"]
        if np.isfinite(value):
            passed = float(value) >= target if "accuracy" in label.lower() else float(value) <= target
            color = PALETTE["selected"] if passed else PALETTE["danger"]
        ax_hero.text(0.98, y, rendered, fontsize=18, weight="bold", ha="right", va="center", color=color)
    ax_hero.set_xlim(0, 1)
    ax_hero.set_ylim(-0.6, len(hero_metrics) - 0.4)
    ax_hero.set_title("Frozen pipeline evidence", loc="left", weight="bold")
    ax_hero.axis("off")

    _safe_float = lambda v, d=0.0: float(d) if v is None or (isinstance(v, float) and np.isnan(v)) else float(v or d)

    pool_values = [
        _safe_float(completeness.get("catalog_count", 0)),
        _safe_float(completeness.get("ranked_count", 0)),
        float(len(selection.get("selected_features", [])) if selection else 0),
    ]
    ax_pool.bar(["catalog", "ranked", "selected"], pool_values, color=[PALETTE["negative"], PALETTE["positive"], PALETTE["selected"]])
    ax_pool.set_title("Feature evidence", weight="bold")
    ax_pool.set_ylabel("feature count")
    ax_pool.tick_params(axis="x", rotation=20)

    model_values = [
        _safe_float(selected_model.get("valid_accuracy", 0.0)),
        _safe_float(selected_model.get("valid_fp_rate", 0.0)),
        _safe_float(hard_negative.get("valid_accuracy_delta", 0.0)),
    ]
    ax_model.bar(["accuracy", "FPR", "HN Δacc"], model_values, color=[PALETTE["selected"], PALETTE["warning"], PALETTE["negative"]])
    ax_model.axhline(0.01, color=PALETTE["danger"], linestyle="--", linewidth=1, label="FPR target")
    ax_model.set_ylim(min(-0.05, min(model_values, default=0.0) * 1.1), max(1.0, max(model_values, default=0.0) * 1.1))
    ax_model.set_title("Model validation", weight="bold")
    ax_model.legend(loc="upper right")

    post_labels = ["window accuracy", "window FPR", "added latency / 3 s"]
    post_values = [
        _safe_float(post_metrics.get("window_accuracy", 0.0)),
        _safe_float(post_metrics.get("window_fp_rate", 0.0)),
        _safe_float(post_metrics.get("first_worn_output_p95_sec", 0.0)) / 3.0,
    ]
    ax_post.barh(post_labels, post_values, color=[PALETTE["selected"], PALETTE["warning"], PALETTE["negative"]])
    ax_post.axvline(1.0, color=PALETTE["neutral"], linestyle=":", linewidth=1)
    ax_post.set_title("Causal postprocessing", loc="left", weight="bold")
    ax_post.set_xlabel("metric value (latency normalized to 3 s target)")

    fig.suptitle("Manual feature selection to frozen deployment evidence", fontsize=13, weight="bold", x=0.04, ha="left")
    fig.subplots_adjust(top=0.88, left=0.07, right=0.97, bottom=0.12)
    inputs = [
        path for path in [
            completeness_path, selection_path, model_path, hard_negative_path,
            postprocess_path, *(test_paths[:1]),
        ] if path.is_file()
    ]
    return save_scientific_figure(
        fig,
        report_dir / "pipeline_scientific_overview.png",
        source_data=source,
        core_conclusion=(
            "The frozen pipeline links complete feature ranking and explicit manual selection "
            "to deployable model and causal postprocessing evidence."
        ),
        panel_map={
            "a": "Hero validation metrics, deployment targets, and final read-only test accuracy.",
            "b": "Catalog, ranking, and manual-selection counts.",
            "c": "Validation accuracy, false-positive rate, and hard-negative accuracy impact.",
            "d": "Streaming accuracy, false-positive rate, and added latency.",
        },
        inputs=inputs,
        split="frozen_pipeline",
        n_definition="one frozen pipeline configuration with available stage contracts",
        statistics={"metrics": "point estimates from frozen artifacts", "interval": "reported by stage-specific figures"},
        reviewer_risks=["Unavailable stages are displayed as missing and must not be interpreted as zero."],
        test_read_only=False,
    )
