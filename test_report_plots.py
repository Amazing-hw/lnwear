import json
from pathlib import Path


def test_s05_exports_threshold_fp_recall_tradeoff_plot_and_csv(tmp_path):
    import s05_train_final_model as train

    assert hasattr(train, "export_threshold_tradeoff_plot")

    plot_data = {
        "threshold_search": {"threshold": 0.72},
        "threshold_curve": [
            {
                "threshold": 0.10,
                "precision": 0.55,
                "recall": 1.00,
                "fpr": 0.45,
                "fp": 9,
                "fn": 0,
                "tp": 20,
                "tn": 11,
            },
            {
                "threshold": 0.50,
                "precision": 0.90,
                "recall": 0.90,
                "fpr": 0.05,
                "fp": 1,
                "fn": 2,
                "tp": 18,
                "tn": 19,
            },
            {
                "threshold": 0.72,
                "precision": 1.00,
                "recall": 0.75,
                "fpr": 0.00,
                "fp": 0,
                "fn": 5,
                "tp": 15,
                "tn": 20,
            },
        ],
    }

    paths = train.export_threshold_tradeoff_plot(plot_data, tmp_path)

    assert Path(paths["figure"]).name == "s05_threshold_fp_recall_tradeoff.png"
    assert Path(paths["source_data"]).name == "s05_threshold_fp_recall_tradeoff.csv"
    assert Path(paths["manifest"]).exists()
    assert Path(paths["qa"]).exists()
    assert (tmp_path / "report_plots" / "s05_threshold_fp_recall_tradeoff.png").exists()
    csv_text = (tmp_path / "report_plots" / "s05_threshold_fp_recall_tradeoff.csv").read_text(
        encoding="utf-8"
    )
    assert "false_positive_rate" in csv_text
    assert "0.72" in csv_text
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["dpi"] == 600
    assert manifest["split"] == "valid"
    assert manifest["inputs"]
    assert all(item["sha256"] for item in manifest["inputs"])


def test_s08_uses_shell_free_subprocess_execution():
    source = Path("s08_run_pipeline.py").read_text(encoding="utf-8")

    assert "shell=True" not in source
    assert "subprocess.call(" not in source


def test_s09_commercial_plots_use_scientific_figure_quartets(tmp_path):
    import s09_commercial_compare as s09

    metrics = {
        "precision": 0.9, "recall": 0.8, "f1": 0.85, "accuracy": 0.88,
        "confusion_matrix": {"TN": 8, "FP": 1, "FN": 2, "TP": 9},
    }
    report = {
        "commercial": {
            "summary": metrics,
            "window_stream_summary": metrics,
            "details": [
                {"sample_name": "c0", "target": 0, "window_probs": [0.1, 0.2]},
                {"sample_name": "c1", "target": 1, "window_probs": [0.7, 0.9]},
            ],
        },
        "project": {
            "summary": metrics,
            "window_stream_summary": metrics,
            "details": [
                {"sample_name": "p0", "target": 0, "window_probs": [0.05, 0.15]},
                {"sample_name": "p1", "target": 1, "window_probs": [0.8, 0.95]},
            ],
        },
        "paired_comparison": {"categories": {"both_correct": 18, "both_wrong": 2}},
    }
    input_path = tmp_path / "window_level_compare.csv"
    input_path.write_text("metric,commercial,project\naccuracy,0.88,0.88\n", encoding="utf-8")

    paths = s09.export_comparison_plots(report, tmp_path, inputs=[input_path])

    for png_value in paths.values():
        png_path = Path(png_value)
        assert png_path.exists()
        assert png_path.with_name(png_path.stem + "_source_data.csv").exists()
        assert png_path.with_name(png_path.stem + "_figure_manifest.json").exists()
        assert png_path.with_name(png_path.stem + "_figure_qa.json").exists()
