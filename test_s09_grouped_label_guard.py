import numpy as np

import s09_commercial_compare as s09


def _mixed_grouped_sample():
    return {
        "sample_name": "record_mixed",
        "h5_file": "not-read-before-label-validation.h5",
        "target": 1,
        "window_layout": "grouped_windows",
        "window_indices": [0, 1, 2],
        "window_labels": [0, 1, 1],
    }


def test_grouped_window_targets_preserve_mixed_labels():
    assert s09._prewindow_targets(_mixed_grouped_sample(), 3) == [0, 1, 1]


def test_commercial_training_uses_per_window_targets(monkeypatch):
    monkeypatch.setattr(
        s09,
        "infer_one_sample_commercial",
        lambda _sample, _threshold: {
            "target": 1,
            "fallback": False,
            "stage2_enabled_flags": [1, 1, 1],
            "features": [[1.0] * 8, [2.0] * 8, [3.0] * 8],
            "window_targets": [0, 1, 1],
        },
    )

    X, y = s09.collect_commercial_training_windows([_mixed_grouped_sample()], 0.1e6)

    assert X.shape == (3, 8)
    assert y.tolist() == [0, 1, 1]


def test_commercial_training_includes_stage1_closed_windows(monkeypatch):
    monkeypatch.setattr(
        s09,
        "infer_one_sample_commercial",
        lambda _sample, _threshold: {
            "target": 1,
            "fallback": False,
            "stage1_gate_flags": [0, 1, 0],
            "stage2_enabled_flags": [0, 1, 0],
            "features": [[1.0] * 8, [2.0] * 8, [3.0] * 8],
            "window_targets": [0, 1, 1],
        },
    )

    X, y = s09.collect_commercial_training_windows([_mixed_grouped_sample()], 0.1e6)

    assert X.shape == (3, 8)
    assert y.tolist() == [0, 1, 1]


def test_window_metrics_use_per_window_targets():
    metrics = s09.window_metrics_from_details([{
        "target": 1,
        "window_targets": [0, 1, 1],
        "window_preds": [0, 1, 1],
    }])

    assert metrics["accuracy"] == 1.0
    assert metrics["total_windows"] == 3
