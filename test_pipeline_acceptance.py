import json

import pandas as pd


def _write_figure_quartet(png_path):
    png_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.write_bytes(b"png")
    png_path.with_name(png_path.stem + "_source_data.csv").write_text("value\n1\n", encoding="utf-8")
    png_path.with_name(png_path.stem + "_figure_manifest.json").write_text(
        json.dumps({"inputs": [{"path": "source", "sha256": "abc"}]}), encoding="utf-8"
    )
    png_path.with_name(png_path.stem + "_figure_qa.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )


def test_pipeline_acceptance_reports_sections_without_hiding_failures(tmp_path):
    from pipeline_acceptance import build_pipeline_acceptance_report

    (tmp_path / "feature_pool_completeness.json").write_text(json.dumps({
        "catalog_count": 111, "ranked_count": 111, "unique_ranked_count": 111,
        "missing_from_ranking": [], "extra_in_ranking": [],
    }), encoding="utf-8")
    (tmp_path / "manual_selected_features.json").write_text(json.dumps({
        "feature_pool_version": "stage2_interpretable_v7",
        "selected_features": ["COMM_GREEN_AC"],
    }), encoding="utf-8")
    (tmp_path / "model_candidate_leaderboard.json").write_text(json.dumps({
        "deployment_acceptance": True,
        "selected_candidate": "reference",
        "leaderboard": [{"candidate": "reference", "total_nodes": 120}],
    }), encoding="utf-8")
    (tmp_path / "stage2_c_contract.json").write_text(json.dumps({
        "feature_order": ["COMM_GREEN_AC"],
    }), encoding="utf-8")
    report_dir = tmp_path / "report_plots"
    report_dir.mkdir()
    for suffix in [".png", "_source_data.csv", "_figure_manifest.json", "_figure_qa.json"]:
        (report_dir / f"pipeline_scientific_overview{suffix}").write_text("{}", encoding="utf-8")

    report = build_pipeline_acceptance_report(tmp_path)

    assert list(report["sections"]) == [
        "Feature pool", "Selection", "Model", "Postprocess",
        "Test", "C readiness", "Figures",
    ]
    assert report["sections"]["Feature pool"]["passed"] is True
    assert report["sections"]["Postprocess"]["passed"] is False
    assert report["overall_passed"] is False
    assert (tmp_path / "pipeline_acceptance.json").exists()
    assert (tmp_path / "pipeline_acceptance.md").exists()
    csv = pd.read_csv(tmp_path / "pipeline_acceptance.csv")
    assert set(csv["section"]) == set(report["sections"])


def test_pipeline_acceptance_rejects_analysis_only_postprocess(tmp_path):
    from pipeline_acceptance import build_pipeline_acceptance_report

    post_dir = tmp_path / "postprocess_opt"
    post_dir.mkdir()
    (post_dir / "postprocess_optimized.json").write_text(json.dumps({
        "metrics": {"window_fp_rate": 0.001, "first_worn_output_p95_sec": 1.0},
        "selection_decision": {"deployment_acceptance": False, "status": "analysis_only"},
    }), encoding="utf-8")

    report = build_pipeline_acceptance_report(tmp_path)

    assert report["sections"]["Postprocess"]["passed"] is False


def test_pipeline_acceptance_rejects_over_budget_selected_model_even_if_flagged(tmp_path):
    from pipeline_acceptance import build_pipeline_acceptance_report

    (tmp_path / "model_candidate_leaderboard.json").write_text(json.dumps({
        "deployment_acceptance": True,
        "selected_candidate": "oversized",
        "max_nodes": 500,
        "max_valid_fp_rate": 0.01,
        "leaderboard": [{
            "candidate": "oversized", "total_nodes": 501,
            "valid_fp_rate": 0.001, "finite_predictions": True,
        }],
    }), encoding="utf-8")

    report = build_pipeline_acceptance_report(tmp_path)

    assert report["sections"]["Model"]["passed"] is False


def test_pipeline_acceptance_requires_matching_hard_negative_decision(tmp_path):
    from pipeline_acceptance import build_pipeline_acceptance_report

    (tmp_path / "model_candidate_leaderboard.json").write_text(json.dumps({
        "deployment_acceptance": True,
        "selected_candidate": "reference",
        "max_nodes": 500,
        "max_valid_fp_rate": 0.01,
        "leaderboard": [{
            "candidate": "reference", "total_nodes": 120,
            "valid_fp_rate": 0.005, "finite_predictions": True,
        }],
    }), encoding="utf-8")

    report = build_pipeline_acceptance_report(tmp_path)
    assert report["sections"]["Model"]["passed"] is False

    (tmp_path / "hard_negative_decision.json").write_text(json.dumps({
        "accepted": False,
        "reason": "valid_accuracy_decreased",
        "selected_candidate": "reference",
        "reference_candidate": "reference",
        "hard_negative_candidate": "hard_negative",
    }), encoding="utf-8")
    report = build_pipeline_acceptance_report(tmp_path)
    assert report["sections"]["Model"]["passed"] is True


def test_pipeline_acceptance_rejects_selection_without_frozen_provenance(tmp_path):
    from pipeline_acceptance import build_pipeline_acceptance_report

    (tmp_path / "manual_selected_features.json").write_text(json.dumps({
        "feature_pool_version": "stage2_interpretable_v7",
        "selected_features": ["COMM_GREEN_AC"],
    }), encoding="utf-8")

    report = build_pipeline_acceptance_report(tmp_path)

    assert report["sections"]["Selection"]["passed"] is False


def test_pipeline_acceptance_rejects_non_csv_selection_provenance(tmp_path):
    from pipeline_acceptance import build_pipeline_acceptance_report

    (tmp_path / "manual_selected_features.json").write_text(json.dumps({
        "feature_pool_version": "stage2_interpretable_v7",
        "selected_features": ["COMM_GREEN_AC"],
        "selection_provenance": {
            "selection_source_type": "xlsx",
            "csv_schema_version": 1,
            "manual_feature_file_sha256": "abc",
            "ranking_source_sha256": "def",
        },
    }), encoding="utf-8")

    report = build_pipeline_acceptance_report(tmp_path)

    assert report["sections"]["Selection"]["passed"] is False


def test_pipeline_acceptance_requires_complete_c_metadata(tmp_path):
    from pipeline_acceptance import build_pipeline_acceptance_report

    (tmp_path / "manual_selected_features.json").write_text(json.dumps({
        "feature_pool_version": "stage2_interpretable_v7",
        "selected_features": ["COMM_GREEN_AC"],
    }), encoding="utf-8")
    (tmp_path / "stage2_c_contract.json").write_text(json.dumps({
        "feature_order": ["COMM_GREEN_AC"],
    }), encoding="utf-8")

    report = build_pipeline_acceptance_report(tmp_path)

    assert report["sections"]["C readiness"]["passed"] is False


def test_pipeline_acceptance_requires_every_mandatory_figure_quartet(tmp_path):
    from pipeline_acceptance import build_pipeline_acceptance_report

    _write_figure_quartet(tmp_path / "report_plots" / "pipeline_scientific_overview.png")

    report = build_pipeline_acceptance_report(tmp_path)

    assert report["sections"]["Figures"]["passed"] is False


def test_completed_pipeline_returns_nonzero_when_acceptance_fails():
    import s08_run_pipeline as s08

    assert s08.pipeline_acceptance_exit_code({"overall_passed": False}, stopped_at=None) == 2
    assert s08.pipeline_acceptance_exit_code({"overall_passed": True}, stopped_at=None) == 0
    assert s08.pipeline_acceptance_exit_code({"overall_passed": False}, stopped_at="s04") == 0


def test_pipeline_acceptance_recognizes_frozen_read_only_test_evaluation(tmp_path):
    from pipeline_acceptance import build_pipeline_acceptance_report

    (tmp_path / "end_to_end_eval_test_state_machine.json").write_text(json.dumps({
        "evaluation_contract": {
            "split": "test", "test_read_only": True,
            "configuration_frozen": True, "selection_performed": False,
        },
        "summary": {
            "accuracy": 0.9,
            "parallel_semantics_version": "stage1_mask_stage2_continuous_v1",
            "stage1_only": {"accuracy": 0.8},
            "stage2_independent": {"accuracy": 0.9},
            "fused_output": {"accuracy": 0.9},
        },
        "window_model_summary": {"accuracy": 0.9},
        "window_stream_summary": {"accuracy": 0.9},
    }), encoding="utf-8")

    report = build_pipeline_acceptance_report(tmp_path)

    assert report["sections"]["Test"]["passed"] is True


def test_pipeline_acceptance_rejects_evaluation_without_parallel_metric_scopes(tmp_path):
    from pipeline_acceptance import build_pipeline_acceptance_report

    (tmp_path / "end_to_end_eval_test_state_machine.json").write_text(json.dumps({
        "evaluation_contract": {
            "split": "test", "test_read_only": True,
            "configuration_frozen": True, "selection_performed": False,
        },
        "summary": {"accuracy": 0.9},
        "window_model_summary": {"accuracy": 0.9},
        "window_stream_summary": {"accuracy": 0.9},
    }), encoding="utf-8")

    report = build_pipeline_acceptance_report(tmp_path)

    assert report["sections"]["Test"]["passed"] is False


def test_s06_builds_explicit_read_only_test_contract():
    import s06_deploy_eval as s06

    contract = s06.build_evaluation_contract("test")

    assert contract == {
        "split": "test",
        "test_read_only": True,
        "configuration_frozen": True,
        "selection_performed": False,
    }
