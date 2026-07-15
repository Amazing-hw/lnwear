"""Sectioned acceptance reporting for frozen liveness artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stage2_feature_catalog import FEATURE_POOL_VERSION, model_candidate_names


SECTION_ORDER = [
    "Feature pool", "Selection", "Model", "Postprocess",
    "Test", "C readiness", "Figures",
]

MANDATORY_FIGURES = [
    Path("report_plots/s01_split_analysis.png"),
    Path("stage1_scatter.png"),
    Path("report_plots/s03_feature_pool_analysis.png"),
    Path("report_plots/s04_feature_selection_report.png"),
    Path("report_plots/s05_threshold_fp_recall_tradeoff.png"),
    Path("report_plots/s06_deploy_report.png"),
    Path("postprocess_opt/postprocess_search_summary.png"),
    Path("report_plots/pipeline_scientific_overview.png"),
]
def _read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _section(passed, summary, evidence):
    return {
        "passed": bool(passed),
        "summary": str(summary),
        "evidence": [str(path) for path in evidence],
    }


def _figure_quartet(png_path: Path):
    manifest_path = png_path.with_name(png_path.stem + "_figure_manifest.json")
    qa_path = png_path.with_name(png_path.stem + "_figure_qa.json")
    manifest = _read_json(manifest_path) or {}
    source_path = Path(manifest.get("source_data", "")) if manifest.get("source_data") else png_path.with_name(
        png_path.stem + "_source_data.csv"
    )
    paths = [png_path, source_path, manifest_path, qa_path]
    present = all(path.is_file() and path.stat().st_size > 0 for path in paths)
    qa = _read_json(qa_path) or {}
    inputs = manifest.get("inputs") if isinstance(manifest, dict) else None
    hashed_inputs = bool(
        isinstance(inputs, list)
        and inputs
        and all(item.get("path") and item.get("sha256") for item in inputs if isinstance(item, dict))
        and all(isinstance(item, dict) for item in inputs)
    )
    passed = bool(
        present
        and qa.get("passed") is True
        and manifest.get("core_conclusion")
        and manifest.get("panel_map")
        and manifest.get("split")
        and manifest.get("n_definition")
        and hashed_inputs
    )
    return passed, paths


def _test_evaluation_passed(payload):
    contract = payload.get("evaluation_contract", {}) if isinstance(payload, dict) else {}
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    return bool(
        contract.get("split") == "test"
        and contract.get("test_read_only") is True
        and contract.get("configuration_frozen") is True
        and contract.get("selection_performed") is False
        and isinstance(summary, dict)
        and summary.get("parallel_semantics_version") == "stage1_mask_stage2_continuous_v1"
        and isinstance(summary.get("stage1_only"), dict)
        and isinstance(summary.get("stage2_independent"), dict)
        and isinstance(summary.get("fused_output"), dict)
        and isinstance(payload.get("window_model_summary"), dict)
        and isinstance(payload.get("window_stream_summary"), dict)
    )


def build_pipeline_acceptance_report(artifact_dir):
    artifact_dir = Path(artifact_dir)
    completeness_path = artifact_dir / "feature_pool_completeness.json"
    selection_path = artifact_dir / "manual_selected_features.json"
    model_path = artifact_dir / "model_candidate_leaderboard.json"
    hard_negative_path = artifact_dir / "hard_negative_decision.json"
    post_path = artifact_dir / "postprocess_opt" / "postprocess_optimized.json"
    test_paths = sorted({
        *artifact_dir.glob("end_to_end_eval_test_*.json"),
        *artifact_dir.glob("end_to_end_test_*.json"),
    })
    c_path = artifact_dir / "stage2_c_contract.json"
    completeness = _read_json(completeness_path) or {}
    expected_feature_count = len(model_candidate_names())
    feature_pass = bool(
        int(completeness.get("catalog_count", 0)) == expected_feature_count
        and int(completeness.get("ranked_count", 0)) == expected_feature_count
        and int(completeness.get("unique_ranked_count", 0)) == expected_feature_count
        and not completeness.get("missing_from_ranking")
        and not completeness.get("extra_in_ranking")
    )
    selection = _read_json(selection_path) or {}
    selection_provenance = selection.get("selection_provenance", {})
    selection_pass = bool(
        selection.get("feature_pool_version") == FEATURE_POOL_VERSION
        and selection.get("selected_features")
        and isinstance(selection_provenance, dict)
        and selection_provenance.get("selection_source_type") == "csv"
        and int(selection_provenance.get("csv_schema_version", -1)) == 1
        and selection_provenance.get("manual_feature_file_sha256")
        and selection_provenance.get("ranking_source_sha256")
    )
    model = _read_json(model_path) or {}
    hard_negative = _read_json(hard_negative_path) or {}
    selected_model_name = model.get("selected_candidate")
    selected_model = next((
        item for item in model.get("leaderboard", [])
        if item.get("candidate") == selected_model_name
    ), {})
    model_pass = bool(
        model.get("deployment_acceptance") is True
        and selected_model
        and selected_model.get("finite_predictions") is True
        and int(selected_model.get("total_nodes", 10**9)) <= int(model.get("max_nodes", 500))
        and float(selected_model.get("valid_fp_rate", 1.0))
        <= float(model.get("max_valid_fp_rate", 0.01))
        and isinstance(hard_negative.get("accepted"), bool)
        and hard_negative.get("reason")
        and hard_negative.get("selected_candidate") == selected_model_name
        and hard_negative.get("reference_candidate")
        and hard_negative.get("hard_negative_candidate")
    )
    post = _read_json(post_path) or {}
    post_metrics = post.get("metrics", {}) if isinstance(post, dict) else {}
    post_decision = post.get("selection_decision", {}) if isinstance(post, dict) else {}
    post_pass = bool(
        post_metrics
        and post_decision.get("deployment_acceptance") is True
        and float(post_metrics.get("window_fp_rate", 1.0)) <= 0.01
        and float(post_metrics.get("first_worn_output_p95_sec", float("inf"))) <= 3.0
    )
    test_pass = bool(
        test_paths
        and all(_test_evaluation_passed(_read_json(path) or {}) for path in test_paths)
    )
    c_contract = _read_json(c_path) or {}
    selected_features = list(selection.get("selected_features", []))
    c_features = c_contract.get("features", {}) if isinstance(c_contract, dict) else {}
    c_pass = bool(
        c_contract.get("feature_pool_version") == FEATURE_POOL_VERSION
        and selected_features
        and list(c_contract.get("feature_order", [])) == selected_features
        and list(c_features) == selected_features
        and c_contract.get("operator_inventory")
        and all(
            c_features.get(name, {}).get("formula")
            and c_features.get(name, {}).get("preprocessing")
            and c_features.get(name, {}).get("c_operators")
            and c_features.get(name, {}).get("accumulator") in {"float32", "float64"}
            and float(c_features.get(name, {}).get("c_abs_tolerance", 0.0)) > 0.0
            and float(c_features.get(name, {}).get("c_rel_tolerance", 0.0)) > 0.0
            for name in selected_features
        )
    )
    mandatory_figures = list(MANDATORY_FIGURES)
    figure_results = [_figure_quartet(artifact_dir / relative) for relative in mandatory_figures]
    figure_paths = [path for _passed, paths in figure_results for path in paths]
    figures_pass = bool(figure_results and all(passed for passed, _paths in figure_results))

    sections = {
        "Feature pool": _section(feature_pass, f"{expected_feature_count} governed candidates are uniquely and completely ranked." if feature_pass else "Feature-pool completeness failed.", [completeness_path]),
        "Selection": _section(selection_pass, f"{len(selected_features)} manually selected features and source hashes are frozen." if selection_pass else "Manual selection is missing, stale, or lacks source hashes.", [selection_path]),
        "Model": _section(model_pass, f"Selected model: {model.get('selected_candidate', 'unavailable')}" if model_pass else "Selected model or hard-negative decision is missing, inconsistent, or violates finite-prediction, FPR, or node constraints.", [model_path, hard_negative_path]),
        "Postprocess": _section(post_pass, "Causal postprocessing meets FPR and added-latency targets." if post_pass else "Postprocessing is missing or does not meet FPR/latency targets.", [post_path]),
        "Test": _section(test_pass, "Frozen read-only test acceptance passed." if test_pass else "Frozen read-only test acceptance is missing or failed.", test_paths),
        "C readiness": _section(c_pass, "C feature order and implementation metadata match the frozen manual selection." if c_pass else "C contract is missing, reordered, or lacks required implementation metadata.", [c_path]),
        "Figures": _section(figures_pass, "All mandatory scientific PNG/source/manifest/QA quartets are complete and source-hashed." if figures_pass else "One or more mandatory scientific figure quartets are incomplete or unhashed.", figure_paths),
    }
    report = {
        "feature_pool_version": FEATURE_POOL_VERSION,
        "overall_passed": all(section["passed"] for section in sections.values()),
        "sections": sections,
    }
    json_path = artifact_dir / "pipeline_acceptance.json"
    csv_path = artifact_dir / "pipeline_acceptance.csv"
    md_path = artifact_dir / "pipeline_acceptance.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([
        {"section": name, "passed": section["passed"], "summary": section["summary"]}
        for name, section in sections.items()
    ]).to_csv(csv_path, index=False)
    lines = ["# Pipeline Acceptance", "", f"Overall passed: **{report['overall_passed']}**", ""]
    for name in SECTION_ORDER:
        section = sections[name]
        lines.extend([f"## {name}", "", f"Passed: **{section['passed']}**", "", section["summary"], ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return report
