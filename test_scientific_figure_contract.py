import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pytest
from PIL import Image


def test_scientific_figure_export_writes_600dpi_png_csv_manifest_and_qa(tmp_path):
    from scientific_figures import save_scientific_figure

    input_path = tmp_path / "input.json"
    input_path.write_text('{"source":"synthetic"}', encoding="utf-8")
    source = pd.DataFrame({"threshold": [0.2, 0.5, 0.8], "accuracy": [0.7, 0.9, 0.8]})
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot(source["threshold"], source["accuracy"], marker="o")
    ax.set(xlabel="Threshold", ylabel="Accuracy", title="Validation threshold response")

    outputs = save_scientific_figure(
        fig,
        tmp_path / "threshold_response.png",
        source_data=source,
        core_conclusion="Validation accuracy peaks near threshold 0.5.",
        panel_map={"a": "Threshold versus validation accuracy."},
        inputs=[input_path],
        split="valid",
        n_definition="three threshold candidates",
        statistics={"metric": "window accuracy", "interval": "none"},
        reviewer_risks=["synthetic fixture only"],
        test_read_only=False,
    )
    plt.close(fig)

    assert outputs["png"].exists()
    assert outputs["source_data"].exists()
    assert outputs["manifest"].exists()
    assert outputs["qa"].exists()
    with Image.open(outputs["png"]) as image:
        assert image.size == (2400, 1800)
        dpi = image.info["dpi"]
        assert dpi[0] == pytest.approx(600, rel=0.01)
        extrema = image.convert("L").getextrema()
        assert extrema[0] < extrema[1]
    exported = pd.read_csv(outputs["source_data"])
    pd.testing.assert_frame_equal(exported, source)
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["core_conclusion"].startswith("Validation accuracy")
    assert manifest["panel_map"] == {"a": "Threshold versus validation accuracy."}
    assert manifest["split"] == "valid"
    assert manifest["dpi"] == 600
    assert manifest["inputs"][0]["sha256"]
    assert manifest["test_read_only"] is False
    qa = json.loads(outputs["qa"].read_text(encoding="utf-8"))
    assert qa["passed"] is True


def test_scientific_figure_requires_source_data_and_valid_split(tmp_path):
    from scientific_figures import save_scientific_figure
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with pytest.raises(ValueError, match="source_data"):
        save_scientific_figure(
            fig,
            tmp_path / "invalid.png",
            source_data=None,
            core_conclusion="A result.",
            panel_map={"a": "Evidence."},
            split="valid",
            n_definition="two points",
        )
    with pytest.raises(ValueError, match="test_read_only"):
        save_scientific_figure(
            fig,
            tmp_path / "invalid_test.png",
            source_data=pd.DataFrame({"x": [0, 1], "y": [0, 1]}),
            core_conclusion="A result.",
            panel_map={"a": "Evidence."},
            split="test",
            n_definition="two points",
            test_read_only=False,
        )
    plt.close(fig)


def test_scientific_figure_can_preserve_existing_source_csv_name(tmp_path):
    from scientific_figures import save_scientific_figure

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    outputs = save_scientific_figure(
        fig,
        tmp_path / "figure.png",
        source_data=pd.DataFrame({"x": [0, 1], "y": [0, 1]}),
        source_data_path=tmp_path / "legacy.csv",
        core_conclusion="The trend increases.",
        panel_map={"a": "Increasing trend."},
        split="valid",
        n_definition="two points",
    )
    plt.close(fig)

    assert outputs["source_data"] == tmp_path / "legacy.csv"
    assert outputs["source_data"].exists()


def test_scientific_export_ignores_global_tight_bbox_state(tmp_path):
    from scientific_figures import save_scientific_figure

    previous = mpl.rcParams["savefig.bbox"]
    try:
        mpl.rcParams["savefig.bbox"] = "tight"
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot([0, 1], [0, 1])
        outputs = save_scientific_figure(
            fig,
            tmp_path / "stable_canvas.png",
            source_data=pd.DataFrame({"x": [0, 1], "y": [0, 1]}),
            core_conclusion="The canvas remains stable.",
            panel_map={"a": "Line."},
            split="valid",
            n_definition="two points",
        )
        plt.close(fig)
    finally:
        mpl.rcParams["savefig.bbox"] = previous

    with Image.open(outputs["png"]) as image:
        assert image.size == (1800, 1200)


def test_pipeline_overview_figure_summarizes_available_contracts(tmp_path):
    from scientific_figures import export_pipeline_scientific_overview

    (tmp_path / "feature_pool_completeness.json").write_text(json.dumps({
        "catalog_count": 91,
        "ranked_count": 91,
        "eligible_count": 80,
        "ineligible_count": 3,
    }), encoding="utf-8")
    (tmp_path / "manual_selected_features.json").write_text(json.dumps({
        "selected_features": ["COMM_GREEN_AC", "GREEN_CORR"],
    }), encoding="utf-8")
    (tmp_path / "model_candidate_leaderboard.json").write_text(json.dumps({
        "selected_candidate": "hard_negative",
        "deployment_acceptance": True,
        "leaderboard": [{
            "candidate": "hard_negative", "valid_accuracy": 0.98,
            "valid_fp_rate": 0.008, "total_nodes": 220,
        }],
    }), encoding="utf-8")
    (tmp_path / "hard_negative_decision.json").write_text(json.dumps({
        "accepted": True,
        "valid_accuracy_delta": 0.01,
        "valid_fp_rate_delta": -0.002,
    }), encoding="utf-8")
    (tmp_path / "end_to_end_eval_test_state_machine.json").write_text(json.dumps({
        "evaluation_contract": {
            "split": "test", "test_read_only": True,
            "configuration_frozen": True, "selection_performed": False,
        },
        "summary": {"accuracy": 0.96},
        "window_model_summary": {"accuracy": 0.95},
        "window_stream_summary": {"accuracy": 0.94},
    }), encoding="utf-8")
    post_dir = tmp_path / "postprocess_opt"
    post_dir.mkdir()
    (post_dir / "postprocess_optimized.json").write_text(json.dumps({
        "metrics": {"window_accuracy": 0.97, "window_fp_rate": 0.009,
                    "first_worn_output_p95_sec": 2.0},
    }), encoding="utf-8")

    outputs = export_pipeline_scientific_overview(tmp_path)

    assert outputs["png"].exists()
    assert outputs["source_data"].exists()
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["core_conclusion"].startswith("The frozen pipeline")
    source = pd.read_csv(outputs["source_data"])
    assert set(source["section"]) >= {
        "feature_pool", "selection", "model", "hard_negative", "postprocess", "test"
    }
