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
    assert (tmp_path / "report_plots" / "s05_threshold_fp_recall_tradeoff.png").exists()
    csv_text = (tmp_path / "report_plots" / "s05_threshold_fp_recall_tradeoff.csv").read_text(
        encoding="utf-8"
    )
    assert "false_positive_rate" in csv_text
    assert "0.72" in csv_text


def test_s08_uses_shell_free_subprocess_execution():
    source = Path("s08_run_pipeline.py").read_text(encoding="utf-8")

    assert "shell=True" not in source
    assert "subprocess.call(" not in source
