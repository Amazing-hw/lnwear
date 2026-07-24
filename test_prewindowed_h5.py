import numpy as np
import h5py
import pytest

import s01_data_split as s01
import s03_extract_feature_pool as s03
import s04_feature_selection as s04
import s06_deploy_eval as s06


def test_s03_has_no_automatic_window_trim_contract():
    assert not hasattr(s03, "trim_ordered_windows")
    assert not hasattr(s03, "EDGE_WINDOW_TRIM")
    assert not hasattr(s03, "NoUsableWindowsAfterEdgeTrim")


def test_s03_prewindowed_sample_keeps_every_stored_window(monkeypatch, tmp_path):
    ppg_windows = np.zeros((10, 40, 300), dtype=float)
    ppg_windows[:, 0, :] = 4.0e6
    ppg_windows[:, 1, :] = 1.0e5
    ppg_windows[:, 2:, :] = 2.0e6
    h5_path = tmp_path / "trimmed.h5"
    with h5py.File(h5_path, "w") as f:
        group = f.create_group("sample_trim")
        group.create_dataset("ppg", data=ppg_windows)
    monkeypatch.setattr(s03, "load_acc", lambda sample: None)
    monkeypatch.setattr(
        s03,
        "extract_stage2_window",
        lambda *_args, **_kwargs: (
            {"GREEN_CORR": 1.0},
            {"mode": 0, "feature_pool_version": s03.STAGE2_FEATURE_POOL_VERSION},
            {},
        ),
    )

    rows = s03._extract_rows_for_sample(
        {
            "sample_name": "sample_trim", "h5_file": str(h5_path), "target": 1,
            "frequency": 100, "ppg_config": 0,
        },
        window_len=300,
        stride_len=100,
        fs=100,
        target_aware_stride=False,
        stride_neg=100,
        stride_pos=100,
        use_stage2_ir=False,
    )

    assert [row["window_index"] for row in rows] == list(range(10))


def test_s03_parallel_extraction_schedules_heaviest_first_and_preserves_input_order(monkeypatch):
    submitted = []

    class FakeFuture:
        def __init__(self, args):
            self.args = args

        def result(self, timeout=None):
            assert timeout is None
            sample = self.args[0]
            return [{"sample_name": sample["sample_name"], "target": sample["target"]}]

    class FakeExecutor:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, _fn, args):
            submitted.append(args[0]["sample_name"])
            return FakeFuture(args)

    monkeypatch.setattr(s03, "ProcessPoolExecutor", FakeExecutor)

    def complete_last_sample_first(futures):
        return sorted(
            futures,
            key=lambda future: future.args[0]["sample_name"],
            reverse=True,
        )

    monkeypatch.setattr(s03, "as_completed", complete_last_sample_first)
    samples = [
        {"sample_name": "sample_a", "h5_file": "a.h5", "target": 0, "ppg_shape": (100, 40)},
        {"sample_name": "sample_b", "h5_file": "b.h5", "target": 1, "ppg_shape": (1000, 40)},
        {"sample_name": "sample_c", "h5_file": "c.h5", "target": 0, "ppg_shape": (500, 40)},
    ]

    result = s03.extract_features_for_split(
        samples,
        n_workers=2,
    )

    assert submitted == ["sample_b", "sample_c", "sample_a"]
    assert result["sample_name"].tolist() == ["sample_a", "sample_b", "sample_c"]


def test_s03_parallel_extraction_attempts_every_sample_before_reporting_failures(monkeypatch):
    completed = []

    class FakeFuture:
        def __init__(self, args):
            self.args = args

        def result(self, timeout=None):
            assert timeout is None
            sample = self.args[0]
            completed.append(sample["sample_name"])
            if sample["sample_name"] == "sample_bad":
                raise ValueError("synthetic extraction failure")
            return [{"sample_name": sample["sample_name"], "target": sample["target"]}]

    class FakeExecutor:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, _fn, args):
            return FakeFuture(args)

    monkeypatch.setattr(s03, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(s03, "as_completed", lambda futures: list(futures))
    samples = [
        {"sample_name": "sample_bad", "h5_file": "bad.h5", "target": 0},
        {"sample_name": "sample_good", "h5_file": "good.h5", "target": 1},
        {"sample_name": "sample_good_2", "h5_file": "good_2.h5", "target": 0},
    ]

    with pytest.raises(RuntimeError, match="all samples were attempted.*sample_bad"):
        s03.extract_features_for_split(samples, n_workers=2)

    assert sorted(completed) == ["sample_bad", "sample_good", "sample_good_2"]


def test_s03_sample_read_failure_is_not_silently_converted_to_empty_rows(monkeypatch):
    monkeypatch.setattr(
        s03,
        "load_ppg",
        lambda _sample: (_ for _ in ()).throw(OSError("synthetic read failure")),
    )

    with pytest.raises(RuntimeError, match="sample_read.*synthetic read failure"):
        s03._extract_rows_for_sample(
            {
                "sample_name": "sample_read",
                "h5_file": "unreadable.h5",
                "target": 1,
                "frequency": 100,
                "ppg_config": 0,
            },
            window_len=500,
            stride_len=100,
            fs=100,
            target_aware_stride=False,
            stride_neg=100,
            stride_pos=100,
        )


def test_s03_grouped_sample_keeps_all_six_windows(tmp_path, capsys, monkeypatch):
    h5_path = tmp_path / "short_grouped.h5"
    sample_name = "short_grouped"
    with h5py.File(h5_path, "w") as f:
        sample_group = f.create_group(sample_name)
        for window_index in range(6):
            window_group = sample_group.create_group(f"record_w{window_index}_1")
            window_group.create_dataset("ppg", data=np.zeros((40, 500), dtype=float))

    monkeypatch.setattr(
        s03,
        "extract_stage2_window",
        lambda *_args, **_kwargs: (
            {"GREEN_CORR": 1.0},
            {"feature_pool_version": s03.STAGE2_FEATURE_POOL_VERSION},
            {},
        ),
    )
    result = s03.extract_features_for_split(
        [{
            "sample_name": sample_name,
            "h5_file": str(h5_path),
            "target": 1,
            "frequency": 100,
            "ppg_config": 0,
            "window_layout": "grouped_windows",
            "ppg_shape": [6, 40, 500],
        }],
        n_workers=1,
    )

    output = capsys.readouterr().out
    assert len(result) == 6
    assert result["window_index"].tolist() == list(range(6))
    assert "no_windows_after_edge_trim" not in output


def test_s03_parallel_workers_keep_all_grouped_windows(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.delenv("WL_FORCE_SERIAL", raising=False)
    h5_path = tmp_path / "short_grouped_parallel.h5"
    # Three items are required because resolve_n_workers intentionally uses the
    # serial path for n_items <= 2.
    sample_names = ["short_grouped_a", "short_grouped_b", "short_grouped_c"]
    assert s03.resolve_n_workers(2, n_items=len(sample_names)) == 2
    with h5py.File(h5_path, "w") as f:
        for sample_name in sample_names:
            sample_group = f.create_group(sample_name)
            for window_index in range(6):
                window_group = sample_group.create_group(f"record_w{window_index}_1")
                window_group.create_dataset(
                    "ppg", data=np.zeros((40, 500), dtype=float)
                )

    result = s03.extract_features_for_split(
        [{
            "sample_name": sample_name,
            "h5_file": str(h5_path),
            "target": 1,
            "frequency": 100,
            "ppg_config": 0,
            "window_layout": "grouped_windows",
            "ppg_shape": [6, 40, 500],
        } for sample_name in sample_names],
        n_workers=2,
    )

    output = capsys.readouterr().out
    assert len(result) == 18
    assert result.groupby("sample_name").size().to_dict() == {
        name: 6 for name in sample_names
    }
    assert "no_windows_after_edge_trim" not in output
    assert "[ERROR]" not in output


def test_s03_grouped_sample_with_no_recognized_ppg_windows_remains_fatal(tmp_path):
    h5_path = tmp_path / "broken_grouped.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_group("broken_grouped")

    with pytest.raises(RuntimeError, match="has no grouped PPG windows"):
        s03.extract_features_for_split(
            [{
                "sample_name": "broken_grouped",
                "h5_file": str(h5_path),
                "target": 1,
                "frequency": 100,
                "ppg_config": 0,
                "window_layout": "grouped_windows",
                "ppg_shape": [1, 40, 500],
            }],
            n_workers=1,
        )


def test_s03_array_prewindowed_sample_keeps_all_six_windows(
    tmp_path, capsys, monkeypatch
):
    h5_path = tmp_path / "short_array_prewindowed.h5"
    sample_name = "short_array_prewindowed"
    with h5py.File(h5_path, "w") as f:
        sample_group = f.create_group(sample_name)
        sample_group.create_dataset("ppg", data=np.zeros((6, 40, 500), dtype=float))

    monkeypatch.setattr(
        s03,
        "extract_stage2_window",
        lambda *_args, **_kwargs: (
            {"GREEN_CORR": 1.0},
            {"feature_pool_version": s03.STAGE2_FEATURE_POOL_VERSION},
            {},
        ),
    )
    result = s03.extract_features_for_split(
        [{
            "sample_name": sample_name,
            "h5_file": str(h5_path),
            "target": 1,
            "frequency": 100,
            "ppg_config": 0,
            "ppg_shape": [6, 40, 500],
        }],
        n_workers=1,
    )

    output = capsys.readouterr().out
    assert len(result) == 6
    assert result["window_index"].tolist() == list(range(6))
    assert "no_windows_after_edge_trim" not in output


def test_s03_continuous_sample_keeps_all_six_windows(tmp_path, capsys, monkeypatch):
    h5_path = tmp_path / "short_continuous.h5"
    sample_name = "short_continuous"
    # 1000 points at 100 Hz produce exactly six 5 s windows with a 1 s stride.
    with h5py.File(h5_path, "w") as f:
        sample_group = f.create_group(sample_name)
        sample_group.create_dataset("ppg", data=np.zeros((40, 1000), dtype=float))

    monkeypatch.setattr(
        s03,
        "extract_stage2_window",
        lambda *_args, **_kwargs: (
            {"GREEN_CORR": 1.0},
            {"feature_pool_version": s03.STAGE2_FEATURE_POOL_VERSION},
            {},
        ),
    )
    result = s03.extract_features_for_split(
        [{
            "sample_name": sample_name,
            "h5_file": str(h5_path),
            "target": 1,
            "frequency": 100,
            "ppg_config": 0,
            "ppg_shape": [40, 1000],
        }],
        n_workers=1,
    )

    output = capsys.readouterr().out
    assert len(result) == 6
    assert result["start_100hz"].tolist() == [0, 100, 200, 300, 400, 500]
    assert result["window_index"].tolist() == list(range(6))
    assert "no_windows_after_edge_trim" not in output


def test_s03_attempts_all_sample_windows_then_reports_feature_failures(monkeypatch):
    ppg = np.zeros((3, 500, 40), dtype=float)
    ppg[:, :, 0] = 4.0e6
    ppg[:, :, 1] = 1.0e5
    ppg[:, :, 3:6] = 2.0e6
    calls = []

    monkeypatch.setattr(s03, "load_ppg", lambda _sample: ppg)
    monkeypatch.setattr(s03, "load_acc", lambda _sample: None)

    def extract_with_one_failure(*_args, **_kwargs):
        calls.append(len(calls))
        if len(calls) == 1:
            raise ValueError("synthetic window failure")
        return (
            {"GREEN_CORR": 1.0},
            {"feature_pool_version": s03.STAGE2_FEATURE_POOL_VERSION},
            {},
        )

    monkeypatch.setattr(s03, "extract_stage2_window", extract_with_one_failure)

    with pytest.raises(RuntimeError, match="1/3 windows.*synthetic window failure"):
        s03._extract_rows_for_sample(
            {
                "sample_name": "sample_windows",
                "h5_file": "unused.h5",
                "target": 1,
                "frequency": 100,
                "ppg_config": 0,
            },
            window_len=500,
            stride_len=100,
            fs=100,
            target_aware_stride=False,
            stride_neg=100,
            stride_pos=100,
        )

    assert len(calls) == 3


def test_s01_accepts_any_2d_or_3d_ppg_shape():
    assert s01.is_supported_ppg_shape((40, 300))
    assert s01.is_supported_ppg_shape((5, 40, 300))
    assert s01.is_supported_ppg_shape((5, 300, 40))
    assert s01.is_supported_ppg_shape((5, 20, 300))
    assert s01.is_supported_ppg_shape((100, 6))
    assert not s01.is_supported_ppg_shape((40,))
    assert not s01.is_supported_ppg_shape((5, 40, 300, 1))


def test_s03_normalizes_h5_prewindowed_channel_points_layout():
    raw = np.zeros((4, 40, 300), dtype=float)
    raw[:, 0, :] = 4.0e6
    normalized = s03.normalize_ppg_array(raw)
    assert normalized.shape == (4, 300, 40)
    assert float(normalized[0, 0, 0]) == 4.0e6


def test_s03_prewindowed_sample_uses_all_windows(monkeypatch):
    ppg_windows = np.zeros((4, 40, 300), dtype=float)
    ppg_windows[:, 0, :] = 4.0e6
    ppg_windows[:, 1, :] = 1.0e5
    ppg_windows[:, 2:, :] = 2.0e6

    monkeypatch.setattr(s03, "load_ppg", lambda sample: s03.normalize_ppg_array(ppg_windows))
    monkeypatch.setattr(s03, "load_acc", lambda sample: None)
    monkeypatch.setattr(
        s03,
        "extract_stage2_window",
        lambda *_args, **_kwargs: (
            {"GREEN_CORR": 1.0},
            {"mode": 0, "feature_pool_version": s03.STAGE2_FEATURE_POOL_VERSION},
            {},
        ),
    )

    rows = s03._extract_rows_for_sample(
        {
            "sample_name": "sample_a", "h5_file": "x.h5", "target": 1,
            "frequency": 100, "ppg_config": 0,
        },
        window_len=300,
        stride_len=100,
        fs=100,
        target_aware_stride=False,
        stride_neg=100,
        stride_pos=100,
        use_stage2_ir=False,
    )

    assert len(rows) == 4
    assert [r["start_100hz"] for r in rows] == [0, 100, 200, 300]


def test_s03_prewindowed_window_length_must_match_requested_seconds(monkeypatch):
    ppg = np.zeros((2, 75, 6), dtype=float)
    ppg[:, :, 3:] = 2.0e6

    monkeypatch.setattr(s03, "load_ppg", lambda _sample: ppg)
    monkeypatch.setattr(s03, "load_acc", lambda _sample: None)

    with pytest.raises(RuntimeError, match="does not match requested window length"):
        s03._extract_rows_for_sample(
            {
                "sample_name": "mismatched_3s",
                "h5_file": "x.h5",
                "target": 1,
                "frequency": 25,
                "ppg_config": 0,
            },
            window_len=500,
            stride_len=100,
            fs=100,
            target_aware_stride=False,
            stride_neg=100,
            stride_pos=100,
            use_stage2_ir=False,
        )


def test_s06_prewindowed_inference_runs_xgboost_for_all_windows(monkeypatch):
    ppg_windows = np.zeros((4, 40, 300), dtype=float)
    ppg_windows[:, 0, :] = 4.0e6
    ppg_windows[:, 1, :] = 1.0e5
    ppg_windows[:, 2:, :] = 2.0e6
    normalized_ppg = s03.normalize_ppg_array(ppg_windows)

    base = {
        "sample_name": "sample_a",
        "target": 1,
        "frequency": 100,
        "ppg_config": 0,
        "mode": 0,
        "window_probs": [],
        "window_preds": [],
        "quality_metas": [],
        "window_ood_scores": [],
        "window_start_sec": [],
        "window_end_sec": [],
    }

    monkeypatch.setattr(
        s06,
        "get_channels_from_window",
        lambda window, mode: (window[:, 0], window[:, 1], window[:, 2], window[:, 3], window[:, 4]),
    )
    monkeypatch.setattr(
        s06,
        "extract_feature_pool_from_window",
        lambda **_kwargs: ({"GREEN_CORR": 1.0}, {"g1_bp": np.ones(75), "g_top2_bp": np.ones(75), "ir_bp": np.ones(75)}),
    )
    monkeypatch.setattr(
        s06,
        "predict_label_windows",
        lambda feats, bundle: (np.ones(len(feats), dtype=int), np.full(len(feats), 0.8)),
    )

    result = s06._infer_prewindowed_sample(
        base,
        normalized_ppg,
        acc=None,
        window_sec=3,
        stride_sec=1,
        bundle={"feature_quantiles": None, "feature_names": []},
        use_stage2_ir=False,
    )

    assert result["window_start_sec"] == [0.0, 1.0, 2.0, 3.0]
    assert result["window_end_sec"] == [3.0, 4.0, 5.0, 6.0]
    assert result["window_preds"] == [1, 1, 1, 1]
    assert result["window_probs"] == [0.8, 0.8, 0.8, 0.8]
    assert "stage1_gate_flags" not in result
    assert "stage2_enabled_flags" not in result


def test_s06_prewindowed_window_length_mismatch_is_explicit_fallback():
    ppg = np.zeros((2, 75, 6), dtype=float)
    base = {
        "sample_name": "mismatched_3s",
        "target": 1,
        "frequency": 25,
        "ppg_config": 0,
        "mode": 0,
        "fallback": False,
        "window_indices": [],
        "window_labels": [],
    }

    result = s06._infer_prewindowed_sample(
        base,
        ppg,
        acc=None,
        window_sec=5,
        stride_sec=1,
        bundle={"feature_quantiles": None, "feature_names": []},
        use_stage2_ir=False,
    )

    assert result["fallback"] is True
    assert result["fallback_reason"].startswith("all_window_feature_extraction_failed")
    assert "does not match requested window length" in result["fallback_reason"]
    assert result["window_feature_failure_count"] == 2


def test_s06_continuous_inference_runs_xgboost_for_all_windows(monkeypatch):
    ppg = np.zeros((1200, 40), dtype=float)
    ppg[:, 0] = 4.0e6
    ppg[:, 1] = 1.0e5
    ppg[:, 2:] = 2.0e6

    monkeypatch.setattr(s06, "validate_h5_file", lambda *_args, **_kwargs: (True, None))
    monkeypatch.setattr(s06, "load_ppg", lambda _sample: ppg)
    monkeypatch.setattr(s06, "load_acc", lambda _sample: None)
    monkeypatch.setattr(
        s06,
        "get_channels_from_window",
        lambda window, mode: (window[:, 0], window[:, 1], window[:, 2], window[:, 3], window[:, 4]),
    )
    monkeypatch.setattr(
        s06,
        "extract_feature_pool_from_window",
        lambda **_kwargs: ({"GREEN_CORR": 1.0}, {"g_top2_bp": np.ones(75)}),
    )
    monkeypatch.setattr(
        s06,
        "predict_label_windows",
        lambda feats, bundle: (np.ones(len(feats), dtype=int), np.full(len(feats), 0.8)),
    )

    result = s06._infer_one_sample(
        {
            "sample_name": "continuous", "h5_file": "x.h5", "target": 1,
            "frequency": 100, "ppg_config": 0,
        },
        window_sec=3,
        stride_sec=1,
        bundle={"feature_quantiles": None, "feature_names": []},
        use_stage2_ir=False,
    )

    assert result["window_probs"] == [0.8] * 10
    assert result["window_preds"] == [1] * 10
    assert "stage1_gate_flags" not in result
    assert "stage2_enabled_flags" not in result


def test_s06_prewindowed_all_feature_failures_are_explicit_fallback(monkeypatch):
    ppg = np.zeros((2, 125, 12), dtype=float)
    base = {
        "sample_name": "all_failed",
        "target": 1,
        "frequency": 25,
        "ppg_config": 1,
        "mode": 1,
        "fallback": False,
        "fallback_reason": None,
        "window_indices": [],
        "window_labels": [],
    }
    monkeypatch.setattr(
        s06,
        "_extract_model_window_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forced feature failure")),
    )

    result = s06._infer_prewindowed_sample(
        base,
        ppg,
        acc=None,
        window_sec=5,
        stride_sec=1,
        bundle={"feature_quantiles": None, "feature_names": []},
        use_stage2_ir=False
    )

    assert result["fallback"] is True
    assert result["fallback_reason"].startswith("all_window_feature_extraction_failed")
    assert "RuntimeError: forced feature failure" in result["fallback_reason"]
    assert result["window_probs"] == []
    assert result["window_preds"] == []
    assert result["window_feature_failure_count"] == 2


def test_s06_prewindowed_partial_failure_drops_invalid_window_and_keeps_alignment(monkeypatch):
    ppg = np.zeros((3, 125, 12), dtype=float)
    base = {
        "sample_name": "partial_failed",
        "target": 1,
        "frequency": 25,
        "ppg_config": 1,
        "mode": 1,
        "fallback": False,
        "fallback_reason": None,
        "window_indices": [10, 11, 12],
        "window_labels": [1, 0, 1],
        "window_layout": None,
    }
    calls = {"count": 0}

    def extract_or_fail(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("first window failed")
        return {"GREEN_CORR": 1.0}

    monkeypatch.setattr(s06, "_extract_model_window_features", extract_or_fail)
    monkeypatch.setattr(
        s06,
        "predict_label_windows",
        lambda feats, bundle: (np.ones(len(feats), dtype=int), np.full(len(feats), 0.8)),
    )

    result = s06._infer_prewindowed_sample(
        base,
        ppg,
        acc=None,
        window_sec=5,
        stride_sec=1,
        bundle={"feature_quantiles": None, "feature_names": []},
        use_stage2_ir=False
    )

    assert result["fallback"] is False
    assert result["window_feature_failure_count"] == 1
    assert result["window_probs"] == [0.8, 0.8]
    assert result["window_preds"] == [1, 1]
    assert result["window_start_sec"] == [1.0, 2.0]
    assert result["window_end_sec"] == [6.0, 7.0]


def test_s06_continuous_all_feature_failures_are_explicit_fallback(monkeypatch):
    ppg = np.zeros((1200, 40), dtype=float)
    monkeypatch.setattr(s06, "validate_h5_file", lambda *_args, **_kwargs: (True, None))
    monkeypatch.setattr(s06, "load_ppg", lambda _sample: ppg)
    monkeypatch.setattr(s06, "load_acc", lambda _sample: None)
    monkeypatch.setattr(
        s06,
        "_extract_model_window_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("continuous feature failure")),
    )

    result = s06._infer_one_sample(
        {
            "sample_name": "continuous_failed", "h5_file": "x.h5", "target": 1,
            "frequency": 100, "ppg_config": 0,
        },
        window_sec=3,
        stride_sec=1,
        bundle={"feature_quantiles": None, "feature_names": []},
        use_stage2_ir=False,
    )

    assert result["fallback"] is True
    assert result["fallback_reason"].startswith("all_window_feature_extraction_failed")
    assert "RuntimeError: continuous feature failure" in result["fallback_reason"]
    assert result["window_probs"] == []
    assert result["window_feature_failure_count"] == 10


def test_s01_scans_grouped_window_h5_layout(tmp_path):
    h5_path = tmp_path / "grouped.h5"
    with h5py.File(h5_path, "w") as f:
        record = f.create_group("record_a")
        record.create_dataset("frequency", data=100)
        record.create_dataset("ppg_config", data=0)
        for window_name in ["rec_a_w20_1", "rec_a_w0_1", "rec_a_w2_1", "rec_a_w1_1"]:
            win = record.create_group(window_name)
            win.create_dataset("ppg", data=np.zeros((40, 300), dtype=float))

    samples, filtered = s01._scan_one_h5(str(h5_path))

    assert all(value == 0 for value in filtered.values())
    assert len(samples) == 1
    sample = samples[0]
    assert sample["sample_name"] == "record_a"
    assert sample["window_layout"] == "grouped_windows"
    assert sample["target"] == 1
    assert sample["window_indices"] == [0, 1, 2, 20]
    assert sample["window_labels"] == [1, 1, 1, 1]


def test_s01_skips_grouped_window_h5_without_ppg_config(tmp_path):
    h5_path = tmp_path / "grouped_no_cfg.h5"
    with h5py.File(h5_path, "w") as f:
        record_a = f.create_group("原始数据A")
        record_a.create_dataset("frequency", data=100)
        for window_name in ["xxx_w20_1", "xxx_w18_1"]:
            win = record_a.create_group(window_name)
            win.create_dataset("ppg", data=np.zeros((40, 300), dtype=float))
            win.create_dataset("acc", data=np.zeros((3, 300), dtype=float))
        record_b = f.create_group("原始数据B")
        record_b.create_dataset("frequency", data=100)
        win = record_b.create_group("yyy_w0_0")
        win.create_dataset("ppg", data=np.zeros((40, 300), dtype=float))
        win.create_dataset("acc", data=np.zeros((3, 300), dtype=float))

    samples, filtered = s01._scan_one_h5(str(h5_path))

    assert samples == []
    assert filtered["missing_ppg_config"] == 2


def test_s03_grouped_window_h5_uses_w_order_for_skip_and_start(monkeypatch, tmp_path):
    h5_path = tmp_path / "grouped.h5"
    with h5py.File(h5_path, "w") as f:
        record = f.create_group("record_a")
        record.create_dataset("frequency", data=100)
        record.create_dataset("ppg_config", data=0)
        for window_name, value in [
            ("rec_a_w20_1", 20.0), ("rec_a_w0_1", 0.0),
            ("rec_a_w8_1", 8.0), ("rec_a_w2_1", 2.0),
            ("rec_a_w7_1", 7.0), ("rec_a_w1_1", 1.0),
            ("rec_a_w6_1", 6.0), ("rec_a_w3_1", 3.0),
            ("rec_a_w5_1", 5.0), ("rec_a_w4_1", 4.0),
        ]:
            win = record.create_group(window_name)
            ppg = np.zeros((40, 300), dtype=float)
            ppg[0, :] = 4.0e6 + value
            ppg[1, :] = 1.0e5
            ppg[2:, :] = 2.0e6
            win.create_dataset("ppg", data=ppg)

    sample = {
        "sample_name": "record_a",
        "h5_file": str(h5_path),
        "target": 1,
        "window_layout": "grouped_windows",
        "window_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 20],
        "window_labels": [1] * 10,
        "frequency": 100,
        "ppg_config": 0,
    }

    monkeypatch.setattr(
        s03,
        "extract_stage2_window",
        lambda *_args, **_kwargs: (
            {"GREEN_CORR": 1.0},
            {"mode": 0, "feature_pool_version": s03.STAGE2_FEATURE_POOL_VERSION},
            {},
        ),
    )

    rows = s03._extract_rows_for_sample(
        sample,
        window_len=300,
        stride_len=100,
        fs=100,
        target_aware_stride=False,
        stride_neg=100,
        stride_pos=100,
        use_stage2_ir=False,
    )

    assert [row["start_100hz"] for row in rows] == [
        0, 100, 200, 300, 400, 500, 600, 700, 800, 2000,
    ]
    assert [row["window_index"] for row in rows] == [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 20,
    ]
    assert all(row["target"] == 1 for row in rows)


def test_s06_grouped_window_inference_uses_w_order_for_timestamps(monkeypatch, tmp_path):
    h5_path = tmp_path / "grouped.h5"
    with h5py.File(h5_path, "w") as f:
        record = f.create_group("record_a")
        record.create_dataset("frequency", data=100)
        record.create_dataset("ppg_config", data=0)
        for window_name in [
            "rec_a_w20_1", "rec_a_w0_1", "rec_a_w8_1", "rec_a_w2_1",
            "rec_a_w7_1", "rec_a_w1_1", "rec_a_w6_1", "rec_a_w3_1",
            "rec_a_w5_1", "rec_a_w4_1",
        ]:
            win = record.create_group(window_name)
            ppg = np.zeros((40, 300), dtype=float)
            ppg[0, :] = 4.0e6
            ppg[1, :] = 1.0e5
            ppg[2:, :] = 2.0e6
            win.create_dataset("ppg", data=ppg)

    sample = {
        "sample_name": "record_a",
        "h5_file": str(h5_path),
        "target": 1,
        "window_layout": "grouped_windows",
        "window_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 20],
        "window_labels": [1] * 10,
        "frequency": 100,
        "ppg_config": 0,
    }

    monkeypatch.setattr(
        s06,
        "get_channels_from_window",
        lambda window, mode: (window[:, 0], window[:, 1], window[:, 2], window[:, 3], window[:, 4]),
    )
    monkeypatch.setattr(
        s06,
        "extract_feature_pool_from_window",
        lambda **_kwargs: ({"GREEN_CORR": 1.0}, {"g1_bp": np.ones(75), "g_top2_bp": np.ones(75), "ir_bp": np.ones(75)}),
    )
    monkeypatch.setattr(
        s06,
        "predict_label_windows",
        lambda feats, bundle: (np.ones(len(feats), dtype=int), np.full(len(feats), 0.8)),
    )

    result = s06._infer_one_sample(
        sample,
        window_sec=3,
        stride_sec=1,
        bundle={"feature_quantiles": None, "feature_names": []},
        use_stage2_ir=False,
    )

    assert result["window_start_sec"] == [
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 20.0,
    ]
    assert result["window_end_sec"] == [
        3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 23.0,
    ]
    assert result["window_indices"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 20]
    assert result["window_targets"] == [1] * 10


def test_s04_excludes_window_metadata_but_keeps_mode_candidate():
    import pandas as pd

    df = pd.DataFrame({
        "sample_name": ["a", "b"],
        "h5_file": ["a.h5", "b.h5"],
        "target": [0, 1],
        "start_100hz": [0, 100],
        "window_index": [0, 1],
        "mode": [1, 1],
        "GREEN_CORR": [0.1, 0.9],
    })

    assert set(s04.get_feature_cols(df)) == {"mode", "GREEN_CORR"}
