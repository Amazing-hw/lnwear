import numpy as np
import h5py

import s01_data_split as s01
import s03_extract_feature_pool as s03
import s04_feature_selection as s04
import s06_deploy_eval as s06


def test_s01_accepts_prewindowed_ppg_shape():
    assert s01.is_supported_ppg_shape((40, 300))
    assert s01.is_supported_ppg_shape((5, 40, 300))
    assert not s01.is_supported_ppg_shape((5, 300, 40))
    assert not s01.is_supported_ppg_shape((5, 20, 300))


def test_s03_normalizes_h5_prewindowed_channel_points_layout():
    raw = np.zeros((4, 40, 300), dtype=float)
    raw[:, 0, :] = 4.0e6
    normalized = s03.normalize_ppg_array(raw)
    assert normalized.shape == (4, 300, 40)
    assert float(normalized[0, 0, 0]) == 4.0e6


def test_s03_prewindowed_sample_uses_existing_windows(monkeypatch):
    ppg_windows = np.zeros((4, 40, 300), dtype=float)
    ppg_windows[:, 0, :] = 4.0e6
    ppg_windows[:, 1, :] = 1.0e5
    ppg_windows[:, 2:, :] = 2.0e6

    monkeypatch.setattr(s03, "load_ppg", lambda sample: s03.normalize_ppg_array(ppg_windows))
    monkeypatch.setattr(s03, "load_acc", lambda sample: None)
    monkeypatch.setattr(s03, "stage1_sample_pass", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(s03, "stage1_ambient_check", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(s03, "detect_green_mode", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        s03,
        "get_channels_from_window",
        lambda window, mode: (window[:, 0], window[:, 1], window[:, 2], window[:, 3], window[:, 4]),
    )
    monkeypatch.setattr(
        s03,
        "extract_feature_pool_from_window",
        lambda **_kwargs: ({"GREEN_CORR": 1.0}, {"g1_bp": np.ones(75), "g_top2_bp": np.ones(75), "ir_bp": np.ones(75)}),
    )

    rows = s03._extract_rows_for_sample(
        {"sample_name": "sample_a", "h5_file": "x.h5", "target": 1},
        dc_threshold=3.6e6,
        ac_dc_threshold=0.35,
        window_len=300,
        stride_len=100,
        fs=100,
        target_aware_stride=False,
        stride_neg=100,
        stride_pos=100,
        skip_initial_windows=1,
        use_stage2_ir=False,
    )

    assert len(rows) == 3
    assert [r["start_100hz"] for r in rows] == [100, 200, 300]


def test_s06_prewindowed_inference_emits_existing_windows(monkeypatch):
    ppg_windows = np.zeros((4, 40, 300), dtype=float)
    ppg_windows[:, 0, :] = 4.0e6
    ppg_windows[:, 1, :] = 1.0e5
    ppg_windows[:, 2:, :] = 2.0e6
    normalized_ppg = s03.normalize_ppg_array(ppg_windows)

    base = {
        "sample_name": "sample_a",
        "target": 1,
        "stage1_pass": True,
        "mode": 0,
        "window_probs": [],
        "window_preds": [],
        "quality_metas": [],
        "window_ood_scores": [],
        "stage2_enabled_flags": [],
        "window_start_sec": [],
        "window_end_sec": [],
    }

    monkeypatch.setattr(s06, "stage1_sample_pass", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(s06, "stage1_ambient_check", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(s06, "detect_green_mode", lambda *_args, **_kwargs: 0)
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
        dc_threshold=3.6e6,
        ac_dc_threshold=0.35,
        window_sec=3,
        stride_sec=1,
        bundle={"feature_quantiles": None, "feature_names": []},
        use_stage2_ir=False,
        skip_initial_windows=1,
    )

    assert result["window_start_sec"] == [1.0, 2.0, 3.0]
    assert result["window_end_sec"] == [4.0, 5.0, 6.0]
    assert result["window_preds"] == [1, 1, 1]


def test_s01_scans_grouped_window_h5_layout(tmp_path):
    h5_path = tmp_path / "grouped.h5"
    with h5py.File(h5_path, "w") as f:
        record = f.create_group("record_a")
        record.create_dataset("ppg_config", data=65)
        for window_name in ["rec_a_w20_1", "rec_a_w0_1", "rec_a_w2_1", "rec_a_w1_1"]:
            win = record.create_group(window_name)
            win.create_dataset("ppg", data=np.zeros((40, 300), dtype=float))

    samples, filtered = s01._scan_one_h5(str(h5_path))

    assert filtered == {"ppg_cfg": 0, "channel_count": 0}
    assert len(samples) == 1
    sample = samples[0]
    assert sample["sample_name"] == "record_a"
    assert sample["window_layout"] == "grouped_windows"
    assert sample["target"] == 1
    assert sample["window_indices"] == [0, 1, 2, 20]
    assert sample["window_labels"] == [1, 1, 1, 1]


def test_s01_scans_grouped_window_h5_without_ppg_config(tmp_path):
    h5_path = tmp_path / "grouped_no_cfg.h5"
    with h5py.File(h5_path, "w") as f:
        record_a = f.create_group("原始数据A")
        for window_name in ["xxx_w20_1", "xxx_w18_1"]:
            win = record_a.create_group(window_name)
            win.create_dataset("ppg", data=np.zeros((40, 300), dtype=float))
            win.create_dataset("acc", data=np.zeros((3, 300), dtype=float))
        record_b = f.create_group("原始数据B")
        win = record_b.create_group("yyy_w0_0")
        win.create_dataset("ppg", data=np.zeros((40, 300), dtype=float))
        win.create_dataset("acc", data=np.zeros((3, 300), dtype=float))

    samples, filtered = s01._scan_one_h5(str(h5_path))

    assert filtered == {"ppg_cfg": 0, "channel_count": 0}
    assert [s["sample_name"] for s in samples] == ["原始数据A", "原始数据B"]
    assert samples[0]["window_indices"] == [18, 20]
    assert samples[0]["window_labels"] == [1, 1]
    assert samples[0]["target"] == 1
    assert samples[1]["window_indices"] == [0]
    assert samples[1]["window_labels"] == [0]
    assert samples[1]["target"] == 0


def test_s03_grouped_window_h5_uses_w_order_for_skip_and_start(monkeypatch, tmp_path):
    h5_path = tmp_path / "grouped.h5"
    with h5py.File(h5_path, "w") as f:
        record = f.create_group("record_a")
        record.create_dataset("ppg_config", data=65)
        for window_name, value in [
            ("rec_a_w20_1", 20.0),
            ("rec_a_w0_1", 0.0),
            ("rec_a_w2_1", 2.0),
            ("rec_a_w1_1", 1.0),
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
        "window_indices": [0, 1, 2, 20],
        "window_labels": [1, 1, 1, 1],
    }

    monkeypatch.setattr(s03, "stage1_sample_pass", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(s03, "stage1_ambient_check", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(s03, "detect_green_mode", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        s03,
        "get_channels_from_window",
        lambda window, mode: (window[:, 0], window[:, 1], window[:, 2], window[:, 3], window[:, 4]),
    )
    monkeypatch.setattr(
        s03,
        "extract_feature_pool_from_window",
        lambda **_kwargs: ({"GREEN_CORR": 1.0}, {"g1_bp": np.ones(75), "g_top2_bp": np.ones(75), "ir_bp": np.ones(75)}),
    )

    rows = s03._extract_rows_for_sample(
        sample,
        dc_threshold=3.6e6,
        ac_dc_threshold=0.35,
        window_len=300,
        stride_len=100,
        fs=100,
        target_aware_stride=False,
        stride_neg=100,
        stride_pos=100,
        skip_initial_windows=3,
        use_stage2_ir=False,
    )

    assert len(rows) == 1
    assert rows[0]["start_100hz"] == 2000
    assert rows[0]["window_index"] == 20
    assert rows[0]["target"] == 1


def test_s06_grouped_window_inference_uses_w_order_for_timestamps(monkeypatch, tmp_path):
    h5_path = tmp_path / "grouped.h5"
    with h5py.File(h5_path, "w") as f:
        record = f.create_group("record_a")
        record.create_dataset("ppg_config", data=65)
        for window_name in ["rec_a_w20_1", "rec_a_w0_1", "rec_a_w2_1", "rec_a_w1_1"]:
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
        "window_indices": [0, 1, 2, 20],
        "window_labels": [1, 1, 1, 1],
    }

    monkeypatch.setattr(s06, "stage1_sample_pass", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(s06, "stage1_ambient_check", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(s06, "detect_green_mode", lambda *_args, **_kwargs: 0)
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
        dc_threshold=3.6e6,
        ac_dc_threshold=0.35,
        window_sec=3,
        stride_sec=1,
        bundle={"feature_quantiles": None, "feature_names": []},
        skip_initial_windows=3,
        use_stage2_ir=False,
    )

    assert result["window_start_sec"] == [20.0]
    assert result["window_end_sec"] == [23.0]
    assert result["window_indices"] == [20]
    assert result["window_targets"] == [1]


def test_s04_excludes_window_metadata_from_feature_candidates():
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

    assert s04.get_feature_cols(df) == ["GREEN_CORR"]
