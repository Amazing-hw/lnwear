import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import s03_extract_feature_pool as s03
from direct_feature_selection import load_direct_feature_csv, write_direct_feature_csv
from stage2_feature_catalog import FEATURE_POOL_VERSION, model_candidate_names


def _synthetic_window(n=125):
    t = np.arange(n, dtype=np.float64) / 25.0
    ambient = 120.0 + 2.0 * np.sin(2.0 * np.pi * 0.7 * t)
    greens = [
        1000.0 + offset + amp * np.sin(2.0 * np.pi * 1.2 * t + phase)
        for offset, amp, phase in ((0.0, 25.0, 0.0), (20.0, 22.0, 0.1), (-15.0, 18.0, -0.1))
    ]
    ir = np.zeros(n, dtype=np.float64)
    acc = np.column_stack([
        5.0 * np.sin(2.0 * np.pi * 0.8 * t),
        4.0 * np.cos(2.0 * np.pi * 0.8 * t),
        1000.0 + 3.0 * np.sin(2.0 * np.pi * 1.1 * t),
    ])
    return ir, ambient, *greens, acc


def test_direct_feature_csv_preserves_exact_order(tmp_path):
    path = tmp_path / "features.csv"
    expected = ["mode", "GTOP2_CORR", "ACC_MAG_MEAN", "GREEN_CORR"]
    write_direct_feature_csv(path, expected)
    assert load_direct_feature_csv(path) == expected


@pytest.mark.parametrize(
    "bad_rows, message",
    [
        (["GREEN_CORR", "GREEN_CORR"], "duplicate"),
        (["NOT_A_FEATURE"], "unknown"),
        ([], "empty"),
    ],
)
def test_direct_feature_csv_rejects_invalid_selection(tmp_path, bad_rows, message):
    path = tmp_path / "features.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature"])
        writer.writerows([[name] for name in bad_rows])
    with pytest.raises(ValueError, match=message):
        load_direct_feature_csv(path)


def test_selective_extraction_matches_full_pool_for_mixed_families():
    ir, ambient, g1, g2, g3, acc = _synthetic_window()
    selected = [
        "mode",
        "GREEN_CORR",
        "GTOP2_CORR",
        "G_2OF3_PERIODICITY",
        "G_PAIR_SPECTRAL_CONSENSUS",
        "GZONE2_AC_DC_RATIO",
        "ACC_GREEN_BP_CORR",
    ]
    full, _ = s03.assemble_stage2_feature_candidates(
        ir, ambient, g1, g2, g3, mode=1, fs=25.0, acc_window=acc
    )
    direct, _ = s03.assemble_stage2_feature_candidates(
        ir,
        ambient,
        g1,
        g2,
        g3,
        mode=1,
        fs=25.0,
        acc_window=acc,
        selected_features=selected,
    )
    assert list(direct) == selected
    np.testing.assert_allclose(
        [direct[name] for name in selected],
        [full[name] for name in selected],
        rtol=1e-12,
        atol=1e-12,
    )


def test_selective_extraction_matches_all_governed_candidates():
    ir, ambient, g1, g2, g3, acc = _synthetic_window()
    selected = model_candidate_names()
    full, _ = s03.assemble_stage2_feature_candidates(
        ir, ambient, g1, g2, g3, mode=1, fs=25.0, acc_window=acc
    )
    direct, _ = s03.assemble_stage2_feature_candidates(
        ir,
        ambient,
        g1,
        g2,
        g3,
        mode=1,
        fs=25.0,
        acc_window=acc,
        selected_features=selected,
    )
    assert list(direct) == selected
    np.testing.assert_allclose(
        list(direct.values()), list(full.values()), rtol=1e-12, atol=1e-12
    )


def test_every_candidate_can_be_extracted_individually_with_full_pool_parity():
    ir, ambient, g1, g2, g3, acc = _synthetic_window()
    full, _ = s03.assemble_stage2_feature_candidates(
        ir, ambient, g1, g2, g3, mode=1, fs=25.0, acc_window=acc
    )
    for name in model_candidate_names():
        direct, _ = s03.assemble_stage2_feature_candidates(
            ir,
            ambient,
            g1,
            g2,
            g3,
            mode=1,
            fs=25.0,
            acc_window=acc,
            selected_features=[name],
        )
        assert list(direct) == [name]
        np.testing.assert_allclose(
            direct[name], full[name], rtol=1e-12, atol=1e-12,
            err_msg=name,
        )


def test_s05_direct_selection_accepts_sparse_feature_pool(tmp_path):
    import pandas as pd
    from s05_train_final_model import load_direct_feature_selection

    selected = ["mode", "GTOP2_CORR", "ACC_MAG_MEAN"]
    path = write_direct_feature_csv(tmp_path / "features.csv", selected)
    frame = pd.DataFrame({
        "feature_pool_version": [FEATURE_POOL_VERSION, FEATURE_POOL_VERSION],
        "sample_name": ["a", "b"],
        "target": [0, 1],
        "mode": [0.0, 1.0],
        "GTOP2_CORR": [0.1, 0.8],
        "ACC_MAG_MEAN": [1000.0, 990.0],
    })
    loaded, provenance = load_direct_feature_selection(path, frame, frame.copy())
    assert loaded == selected
    assert provenance["feature_selection_mode"] == "direct"
    assert provenance["feature_count_search"] is False


def test_mode_only_skips_optical_and_acc_feature_blocks(monkeypatch):
    ir, ambient, g1, g2, g3, acc = _synthetic_window()

    def fail(*_args, **_kwargs):
        raise AssertionError("expensive feature block should not run")

    monkeypatch.setattr(s03, "extract_feature_pool_from_window", fail)
    monkeypatch.setattr(s03, "_acc_candidate_features", fail)
    features, preprocessed = s03.assemble_stage2_feature_candidates(
        ir,
        ambient,
        g1,
        g2,
        g3,
        mode=2,
        fs=25.0,
        acc_window=acc,
        selected_features=["mode"],
    )
    assert features == {"mode": 2.0}
    assert preprocessed == {}


def test_pipeline_direct_features_dry_run_skips_ranking(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    command = [
        sys.executable,
        str(Path(__file__).with_name("s08_run_pipeline.py")),
        "--dataset_dir",
        str(tmp_path / "dataset"),
        "--artifact_dir",
        str(artifact_dir),
        "--direct_features",
        "mode,GTOP2_CORR,ACC_MAG_MEAN",
        "--dry_run",
        "--stop_after",
        "s05",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    assert "--selected_feature_file" in output
    assert "--feature_selection_mode direct" in output
    assert "s04_feature_selection.py" not in output
    assert load_direct_feature_csv(artifact_dir / "direct_feature_selection.csv") == [
        "mode",
        "GTOP2_CORR",
        "ACC_MAG_MEAN",
    ]
