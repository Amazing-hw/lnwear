import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_feature_pool(path: Path, split: str, offset: float) -> None:
    rows = []
    for idx in range(8):
        target = idx % 2
        rows.append(
            {
                "sample_name": f"{split}_sample_{idx // 2}",
                "h5_file": f"{split}_{idx}.h5",
                "target": target,
                "start_100hz": idx * 100,
                "start_sec": float(idx),
                "window_index": idx,
                "mode": "stage2",
                "feat_linear": offset + target * 2.0 + idx * 0.01,
                "feat_curve": offset + np.sin(idx) + target,
                "feat_energy": offset + idx * 0.25,
                "feat_ratio": offset + (idx + 1) / 10.0,
            }
        )
    pd.DataFrame(rows).to_csv(path / f"feature_pool_{split}.csv", index=False)


def test_embedding_report_exports_pca_2d_3d_figures_and_report(tmp_path):
    import s04_feature_selection as report

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    _write_feature_pool(artifact_dir, "train", 0.0)
    _write_feature_pool(artifact_dir, "valid", 0.5)
    _write_feature_pool(artifact_dir, "test", 1.0)

    output = report.run_embedding_report(
        artifact_dir=artifact_dir,
        methods=("pca",),
        dims=(2, 3),
        formats=("png",),
        max_points=100,
        random_state=7,
    )

    out_dir = artifact_dir / "feature_embedding_report"
    assert output["report_path"] == out_dir / "embedding_report.md"
    assert (out_dir / "embedding_source_data.csv").exists()
    assert (out_dir / "pca_2d.png").exists()
    assert (out_dir / "pca_3d.png").exists()
    assert (out_dir / "embedding_panel_2d.png").exists()
    assert (out_dir / "embedding_panel_3d.png").exists()

    summary = json.loads((out_dir / "embedding_summary.json").read_text(encoding="utf-8"))
    assert summary["n_rows"] == 24
    assert summary["n_features"] == 4
    assert summary["label_counts"] == {"0": 12, "1": 12}
    assert summary["methods"]["pca"]["status"] == "ok"

    md = (out_dir / "embedding_report.md").read_text(encoding="utf-8")
    assert "PCA 2D/3D" in md
    assert "feature_embedding_report" in md


def test_embedding_report_exports_one_distribution_figure_per_selected_feature(tmp_path):
    import s04_feature_selection as report

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    _write_feature_pool(artifact_dir, "train", 0.0)
    _write_feature_pool(artifact_dir, "valid", 0.5)
    _write_feature_pool(artifact_dir, "test", 1.0)
    (artifact_dir / "selected_features.json").write_text(
        json.dumps(
            {
                "selected_features": [
                    "feat_linear",
                    "feat_curve",
                    "feat_energy",
                ]
            }
        ),
        encoding="utf-8",
    )

    output = report.run_embedding_report(
        artifact_dir=artifact_dir,
        methods=("pca",),
        dims=(2,),
        formats=("png",),
        max_points=100,
        random_state=7,
    )

    out_dir = artifact_dir / "feature_embedding_report"
    expected = [
        out_dir / "feature_distribution_01_feat_linear.png",
        out_dir / "feature_distribution_02_feat_curve.png",
        out_dir / "feature_distribution_03_feat_energy.png",
    ]
    for path in expected:
        assert path.exists()
    assert not (out_dir / "feature_distribution_04_feat_ratio.png").exists()
    assert (out_dir / "selected_feature_distribution_source_data.csv").exists()
    assert output["summary"]["selected_feature_distributions"]["n_features"] == 3
    assert output["summary"]["selected_feature_distributions"]["statistics"]["feat_linear"]["auc"] > 0.9

    md = (out_dir / "embedding_report.md").read_text(encoding="utf-8")
    assert "Selected Feature Distributions" in md
    assert "feature_distribution_01_feat_linear.png" in md


def test_embedding_report_uses_selected_features_for_embedding_and_exports_explainers(tmp_path):
    import s04_feature_selection as report

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    _write_feature_pool(artifact_dir, "train", 0.0)
    _write_feature_pool(artifact_dir, "valid", 0.5)
    _write_feature_pool(artifact_dir, "test", 1.0)
    (artifact_dir / "selected_features.json").write_text(
        json.dumps({"selected_features": ["feat_linear", "feat_curve"]}),
        encoding="utf-8",
    )

    output = report.run_embedding_report(
        artifact_dir=artifact_dir,
        methods=("pca",),
        dims=(2,),
        formats=("png",),
        max_points=100,
        random_state=7,
    )

    out_dir = artifact_dir / "feature_embedding_report"
    summary = output["summary"]
    assert summary["embedding_feature_source"] == "selected_features"
    assert summary["n_features"] == 2
    assert summary["feature_columns"] == ["feat_linear", "feat_curve"]
    assert "PC1 (" in summary["methods"]["pca"]["axis_labels"][0]
    assert (out_dir / "selected_feature_correlation_heatmap.png").exists()
    assert (out_dir / "pca_loading_top_features.png").exists()
    assert (out_dir / "selected_feature_split_auc_heatmap.png").exists()

    md = (out_dir / "embedding_report.md").read_text(encoding="utf-8")
    assert "Embedding feature source: selected_features" in md
    assert "selected_feature_correlation_heatmap.png" in md
    assert "pca_loading_top_features.png" in md
    assert "selected_feature_split_auc_heatmap.png" in md


def test_s08_dry_run_can_add_feature_embedding_report_step():
    import subprocess
    import sys

    root = Path(__file__).resolve().parent
    result = subprocess.run(
        [
            sys.executable,
            str(root / "s08_run_pipeline.py"),
            "--dry_run",
            "--stop_after",
            "s04_embed",
            "--skip",
            "s01,s02,s03,s04,s04_search",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "s11_feature_embedding_report.py" not in result.stdout
    assert "__feature_embedding_report__" in result.stdout
    assert '--methods "pca,tsne"' in result.stdout
    assert "--max_points" in result.stdout


def test_embedding_report_exports_umap_when_dependency_is_available(tmp_path, monkeypatch):
    class FakeUMAP:
        def __init__(self, n_components, **_kwargs):
            self.n_components = int(n_components)

        def fit_transform(self, x):
            x = np.asarray(x, dtype=float)
            if x.shape[1] >= self.n_components:
                return x[:, : self.n_components]
            padded = np.zeros((x.shape[0], self.n_components), dtype=float)
            padded[:, : x.shape[1]] = x
            return padded

    fake_umap = types.ModuleType("umap")
    fake_umap.UMAP = FakeUMAP
    monkeypatch.setitem(sys.modules, "umap", fake_umap)

    import s04_feature_selection as report

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    _write_feature_pool(artifact_dir, "train", 0.0)
    _write_feature_pool(artifact_dir, "valid", 0.5)
    _write_feature_pool(artifact_dir, "test", 1.0)

    output = report.run_embedding_report(
        artifact_dir=artifact_dir,
        methods=("umap",),
        dims=(2, 3),
        formats=("png",),
        max_points=100,
        random_state=7,
    )

    out_dir = artifact_dir / "feature_embedding_report"
    assert (out_dir / "umap_2d.png").exists()
    assert (out_dir / "umap_3d.png").exists()
    assert output["summary"]["methods"]["umap"]["status"] == "ok"


def test_load_feature_pools_reports_bad_csv_file(tmp_path):
    import s04_feature_selection as report

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    _write_feature_pool(artifact_dir, "train", 0.0)
    (artifact_dir / "feature_pool_valid.csv").write_text(
        "sample_name,target,feat_a\n"
        "ok,1,0.5\n"
        "bad,0,0.1,extra\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="feature_pool_valid.csv"):
        report.load_feature_pools(artifact_dir)


def test_s08_embedding_report_warning_does_not_abort_pipeline(tmp_path, capsys):
    import s08_run_pipeline as pipeline

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    _write_feature_pool(artifact_dir, "train", 0.0)
    (artifact_dir / "feature_pool_valid.csv").write_text(
        "sample_name,target,feat_a\n"
        "ok,1,0.5\n"
        "bad,0,0.1,extra\n",
        encoding="utf-8",
    )
    args = Namespace(artifact_dir=str(artifact_dir))

    pipeline.run_embedded_feature_embedding_report(args)

    output = capsys.readouterr().out
    assert "[WARN] skip feature embedding report" in output
    assert "feature_pool_valid.csv" in output
