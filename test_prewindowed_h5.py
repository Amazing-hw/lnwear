import numpy as np

import s01_data_split as s01
import s03_extract_feature_pool as s03
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
        lambda **_kwargs: ({"GREEN_CORR": 1.0}, {"g1_bp": np.ones(75), "ir_bp": np.ones(75)}),
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
        lambda **_kwargs: ({"GREEN_CORR": 1.0}, {"g1_bp": np.ones(75), "ir_bp": np.ones(75)}),
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
