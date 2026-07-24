from collections import OrderedDict
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import commercial_liveness_features as commercial
import s03_extract_feature_pool as s03
import s06_deploy_eval as s06


EXPECTED_STAGE2_FIELDS = (
    "GREEN_CORR",
    "COMM_GREEN_AC",
    "COMM_AMB_AC",
    "ACC_MAG_MEAN",
    "GREEN_DC_MEDIAN",
    "AMBX_DC_MEDIAN",
    "GREEN_AUTO_CORR_PEAK",
    "GREEN_FFT_PEAK_MEDIAN_RATIO",
)


def _raw_window(length):
    idx = np.arange(length, dtype=np.float32)
    ppg = np.zeros((length, 6), dtype=np.float32)
    ppg[:, 0] = 4_000_000 + idx
    ppg[:, 1] = 100_000 + 2 * idx
    ppg[:, 3] = 2_000_000 + idx
    ppg[:, 4] = 2_000_001 + 2 * idx
    ppg[:, 5] = 2_000_003 + 3 * idx
    acc = np.column_stack([
        100 + idx,
        -200 + 2 * idx,
        4096 + 3 * idx,
    ]).astype(np.float32)
    return ppg, acc


def test_exact_port_zero_inputs_return_eight_float32_zeros():
    ppg = np.zeros((125, 4), dtype=np.float32)
    acc = np.zeros((125, 3), dtype=np.float32)

    actual = commercial.main(ppg, acc)

    assert actual.shape == (8,)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.zeros(8, dtype=np.float32))


def test_peak_valley_port_handles_the_first_interior_peak():
    signal = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    peak_count, peak_times, peak_values, *_ = commercial._find_peak_valley(signal)

    assert peak_count == 1
    np.testing.assert_array_equal(peak_times, np.asarray([1], dtype=np.int16))
    np.testing.assert_array_equal(peak_values, np.asarray([1.0], dtype=np.float32))


def test_exact_port_dc_uses_absolute_raw_mean():
    ppg = np.zeros((125, 4), dtype=np.float32)
    ppg[:, 0] = -2_000_000
    ppg[:, 3] = 100_000
    acc = np.zeros((125, 3), dtype=np.float32)

    actual = commercial.main(ppg, acc)

    assert actual[4] == np.float32(2_000_000)
    assert actual[5] == np.float32(100_000)


def test_commercial_adapter_uses_raw_stride_and_float32_precision(monkeypatch):
    ppg, acc = _raw_window(500)
    captured = {}

    def fake_main(port_ppg, port_acc):
        captured["ppg"] = port_ppg.copy()
        captured["acc"] = port_acc.copy()
        return np.arange(8, dtype=np.float32)

    monkeypatch.setattr(s03, "_commercial_port_main", fake_main)

    actual = s03.extract_commercial_feature_overrides(
        ppg,
        acc,
        frequency=100,
        ppg_config=0,
    )

    expected_ppg = ppg[::4]
    expected_green = (
        (expected_ppg[:, 3] + expected_ppg[:, 4] + expected_ppg[:, 5]) / 3.0
    )
    assert captured["ppg"].shape == (125, 4)
    assert captured["ppg"].dtype == np.float32
    np.testing.assert_array_equal(captured["ppg"][:, 0], expected_green.astype(np.float32))
    np.testing.assert_array_equal(
        captured["ppg"][:, 3], expected_ppg[:, 1].astype(np.float32)
    )
    np.testing.assert_array_equal(captured["ppg"][:, 1:3], 0)
    assert captured["acc"].shape == (125, 3)
    assert captured["acc"].dtype == np.float32
    np.testing.assert_array_equal(captured["acc"], acc[::4].astype(np.float32))
    assert isinstance(actual, OrderedDict)
    assert tuple(actual) == EXPECTED_STAGE2_FIELDS
    assert list(actual.values()) == pytest.approx(np.arange(8, dtype=float))


def test_commercial_adapter_native_25hz_does_not_resample(monkeypatch):
    ppg, acc = _raw_window(125)
    captured = {}

    def fake_main(port_ppg, port_acc):
        captured["ppg"] = port_ppg.copy()
        captured["acc"] = port_acc.copy()
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(s03, "_commercial_port_main", fake_main)
    s03.extract_commercial_feature_overrides(ppg, acc, frequency=25, ppg_config=0)

    expected_green = ((ppg[:, 3] + ppg[:, 4] + ppg[:, 5]) / 3.0)
    np.testing.assert_array_equal(captured["ppg"][:, 0], expected_green.astype(np.float32))
    np.testing.assert_array_equal(captured["acc"], acc.astype(np.float32))


def test_commercial_adapter_preserves_fractional_green_zones(monkeypatch):
    ppg = np.zeros((125, 12), dtype=np.float32)
    acc = np.zeros((125, 3), dtype=np.float32)
    ppg[:, 1] = 100
    ppg[:, 3] = 0
    ppg[:, 9] = 1
    ppg[:, 4] = 0
    ppg[:, 10] = 1
    ppg[:, 5] = 2
    ppg[:, 11] = 3
    captured = {}

    def fake_main(port_ppg, _port_acc):
        captured["ppg"] = port_ppg.copy()
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(s03, "_commercial_port_main", fake_main)
    s03.extract_commercial_feature_overrides(ppg, acc, frequency=25, ppg_config=1)

    # Physical zones are 0.5, 0.5, and 2.5.  With float32 precision the mean is
    # (0.5+0.5+2.5)/3 = 7/6 ≈ 1.1666666, not truncated to 1.
    np.testing.assert_array_almost_equal(
        captured["ppg"][:, 0], np.full(125, np.float32(7.0 / 6.0)), decimal=4
    )


@pytest.mark.parametrize(
    ("frequency", "length"),
    [(25, 124), (25, 126), (100, 499), (100, 501)],
)
def test_commercial_adapter_rejects_wrong_window_length(frequency, length):
    ppg, acc = _raw_window(length)

    with pytest.raises(ValueError, match="commercial PPG window"):
        s03.extract_commercial_feature_overrides(
            ppg,
            acc,
            frequency=frequency,
            ppg_config=0,
        )


def test_commercial_adapter_rejects_missing_or_misaligned_acc():
    ppg, acc = _raw_window(125)

    with pytest.raises(ValueError, match="commercial ACC window"):
        s03.extract_commercial_feature_overrides(
            ppg,
            None,
            frequency=25,
            ppg_config=0,
        )
    with pytest.raises(ValueError, match="commercial ACC window"):
        s03.extract_commercial_feature_overrides(
            ppg,
            acc[:-1],
            frequency=25,
            ppg_config=0,
        )


def test_commercial_only_row_requires_acc_and_propagates_port_errors(monkeypatch):
    ppg, acc = _raw_window(125)

    with pytest.raises(ValueError, match="commercial ACC window"):
        s03._commercial_only_feature_row(ppg, None, mode=0, frequency=25)

    monkeypatch.setattr(
        s03,
        "extract_commercial_feature_overrides",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("port failed")),
    )
    with pytest.raises(RuntimeError, match="port failed"):
        s03._commercial_only_feature_row(ppg, acc, mode=0, frequency=25)


def test_commercial_only_row_3s_falls_back_to_governed_features():
    """The original commercial port is 5 s only; 3 s uses matching governed fields."""
    ppg, acc = _raw_window(75)

    actual = s03._commercial_only_feature_row(ppg, acc, mode=0, frequency=25)
    expected, diagnostics, _ = s03.extract_stage2_window(
        ppg,
        mode=0,
        fs=25,
        acc_window=acc,
        selected_features=s03.COMMERCIAL_STAGE2_FIELDS,
    )

    assert [actual[name] for name in EXPECTED_STAGE2_FIELDS] == pytest.approx(
        [expected[name] for name in EXPECTED_STAGE2_FIELDS]
    )
    assert actual["TOTAL_INVALID_COUNT"] == diagnostics["TOTAL_INVALID_COUNT"]
    assert actual["ACC_AVAILABLE"] == diagnostics["ACC_AVAILABLE"]


def test_stage2_existing_commercial_fields_are_overridden(monkeypatch):
    ppg, acc = _raw_window(125)
    expected = np.arange(8, dtype=np.float32) + np.float32(10)
    monkeypatch.setattr(s03, "_commercial_port_main", lambda _ppg, _acc: expected)

    features = s03.extract_window_features(
        ppg,
        fs=25,
        acc_window=acc,
        ppg_config=0,
    )

    assert tuple(s03.COMMERCIAL_STAGE2_FIELDS) == EXPECTED_STAGE2_FIELDS
    assert [features[name] for name in EXPECTED_STAGE2_FIELDS] == pytest.approx(expected)


def test_window_feature_extraction_propagates_commercial_override_errors(monkeypatch):
    ppg, acc = _raw_window(125)

    monkeypatch.setattr(
        s03,
        "extract_commercial_feature_overrides",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("port failed")),
    )

    with pytest.raises(RuntimeError, match="port failed"):
        s03.extract_window_features(
            ppg,
            fs=25,
            acc_window=acc,
            ppg_config=0,
        )


def test_s06_help_marks_stage2_ir_option_as_legacy():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("s06_deploy_eval.py")), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "legacy compatibility flag; Stage2 IR is always" in result.stdout


def test_batch_extraction_overrides_commercial_fields_from_raw_100hz_windows(monkeypatch):
    ppg, acc = _raw_window(1500)
    observed = []

    monkeypatch.setattr(s03, "load_ppg", lambda _sample: ppg)
    monkeypatch.setattr(s03, "load_acc", lambda _sample: acc)
    monkeypatch.setattr(
        s03,
        "extract_stage2_window",
        lambda *_args, **_kwargs: ({"GREEN_CORR": -1.0}, {}, {}),
    )

    def fake_overrides(raw_ppg, raw_acc, frequency, ppg_config):
        observed.append((raw_ppg.shape, raw_acc.shape, frequency, ppg_config))
        return OrderedDict((("GREEN_CORR", 10.0),))

    monkeypatch.setattr(s03, "extract_commercial_feature_overrides", fake_overrides)
    rows = s03._extract_rows_for_sample(
        {
            "sample_name": "sample",
            "h5_file": "sample.h5",
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

    assert rows
    assert all(row["GREEN_CORR"] == 10.0 for row in rows)
    assert observed


def test_three_second_windows_do_not_mix_in_five_second_commercial_overrides(monkeypatch):
    ppg, acc = _raw_window(900)
    observed = []

    monkeypatch.setattr(s03, "load_ppg", lambda _sample: ppg)
    monkeypatch.setattr(s03, "load_acc", lambda _sample: acc)
    monkeypatch.setattr(
        s03,
        "extract_stage2_window",
        lambda *_args, **_kwargs: ({"GREEN_CORR": -1.0}, {}, {}),
    )

    def fake_overrides(*args, **kwargs):
        observed.append((args, kwargs))
        return OrderedDict((("GREEN_CORR", 10.0),))

    monkeypatch.setattr(s03, "extract_commercial_feature_overrides", fake_overrides)
    rows = s03._extract_rows_for_sample(
        {
            "sample_name": "sample_3s",
            "h5_file": "sample.h5",
            "target": 1,
            "frequency": 100,
            "ppg_config": 0,
        },
        window_len=300,
        stride_len=100,
        fs=100,
        target_aware_stride=False,
        stride_neg=100,
        stride_pos=100,
    )

    assert rows
    assert all(row["GREEN_CORR"] == -1.0 for row in rows)
    assert observed == []


def test_s06_continuous_inference_applies_same_commercial_overrides_as_training(monkeypatch):
    ppg, acc = _raw_window(1500)
    observed = []
    predicted_features = []

    monkeypatch.setattr(s06, "validate_h5_file", lambda *_args: (True, None))
    monkeypatch.setattr(s06, "load_ppg", lambda _sample: ppg)
    monkeypatch.setattr(s06, "load_acc", lambda _sample: acc)
    def fake_overrides(raw_ppg, raw_acc, frequency, ppg_config):
        observed.append((raw_ppg.shape, raw_acc.shape, frequency, ppg_config))
        return OrderedDict((("GREEN_CORR", 10.0),))

    def fake_predict(features, bundle):
        predicted_features.extend(features)
        return np.ones(len(features), dtype=int), np.full(len(features), 0.8)

    monkeypatch.setattr(s06, "extract_commercial_feature_overrides", fake_overrides)
    monkeypatch.setattr(s06, "predict_label_windows", fake_predict)

    result = s06._infer_one_sample(
        {
            "sample_name": "sample",
            "h5_file": "sample.h5",
            "target": 1,
            "frequency": 100,
            "ppg_config": 0,
        },
        window_sec=5,
        stride_sec=1,
        bundle={"feature_names": ["GREEN_CORR"], "feature_quantiles": None},
    )

    assert result["fallback"] is False
    assert predicted_features
    assert all(feature["GREEN_CORR"] == 10.0 for feature in predicted_features)
    assert all(item == ((500, 6), (500, 3), 100, 0) for item in observed)


def test_s06_prewindowed_inference_applies_same_commercial_overrides_as_training(monkeypatch):
    raw_ppg, raw_acc = _raw_window(500)
    ppg = np.stack([raw_ppg, raw_ppg], axis=0)
    acc = np.stack([raw_acc, raw_acc], axis=0)
    predicted_features = []

    monkeypatch.setattr(
        s06,
        "extract_commercial_feature_overrides",
        lambda *_args, **_kwargs: OrderedDict((("GREEN_CORR", 10.0),)),
    )

    def fake_predict(features, bundle):
        predicted_features.extend(features)
        return np.ones(len(features), dtype=int), np.full(len(features), 0.8)

    monkeypatch.setattr(s06, "predict_label_windows", fake_predict)
    result = s06._infer_prewindowed_sample(
        {
            "sample_name": "sample",
            "target": 1,
            "frequency": 100,
            "ppg_config": 0,
            "window_indices": [],
            "window_labels": [],
            "fallback": False,
        },
        ppg,
        acc,
        window_sec=5,
        stride_sec=1,
        bundle={"feature_names": ["GREEN_CORR"], "feature_quantiles": None},
        use_stage2_ir=False,
    )

    assert result["fallback"] is False
    assert len(predicted_features) == 2
    assert all(feature["GREEN_CORR"] == 10.0 for feature in predicted_features)


def test_commercial_only_continuous_100hz_uses_raw_windows(monkeypatch):
    ppg, acc = _raw_window(1500)
    observed = []

    monkeypatch.setattr(s03, "load_ppg", lambda _sample: ppg)
    monkeypatch.setattr(s03, "load_acc", lambda _sample: acc)

    def fake_overrides(raw_ppg, raw_acc, frequency, ppg_config):
        observed.append((raw_ppg.copy(), raw_acc.copy(), frequency, ppg_config))
        return OrderedDict((name, float(i)) for i, name in enumerate(EXPECTED_STAGE2_FIELDS))

    monkeypatch.setattr(s03, "extract_commercial_feature_overrides", fake_overrides)
    rows = s03._extract_rows_for_sample(
        {
            "sample_name": "sample",
            "h5_file": "sample.h5",
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
        commercial_only=True,
    )

    assert rows
    first_ppg, first_acc, frequency, ppg_config = observed[0]
    assert frequency == 100
    assert ppg_config == 0
    np.testing.assert_array_equal(first_ppg, ppg[0:500])
    np.testing.assert_array_equal(first_acc, acc[0:500])


def test_commercial_only_continuous_3s_uses_actual_source_window(monkeypatch):
    ppg, acc = _raw_window(900)
    observed_lengths = []

    monkeypatch.setattr(s03, "load_ppg", lambda _sample: ppg)
    monkeypatch.setattr(s03, "load_acc", lambda _sample: acc)

    def fake_row(raw_ppg, raw_acc, mode, frequency):
        observed_lengths.append((len(raw_ppg), len(raw_acc), mode, frequency))
        return {
            **{name: 0.0 for name in s03.stage2_model_candidate_names()},
            "mode": int(mode),
            "feature_pool_version": s03.STAGE2_FEATURE_POOL_VERSION,
            "TOTAL_INVALID_COUNT": 0.0,
            "PPG_INVALID_COUNT": 0.0,
            "GREEN_INVALID_COUNT": 0.0,
            "ACC_AVAILABLE": 1.0,
        }

    monkeypatch.setattr(s03, "_commercial_only_feature_row", fake_row)
    rows = s03._extract_rows_for_sample(
        {
            "sample_name": "sample_3s",
            "h5_file": "sample.h5",
            "target": 1,
            "frequency": 100,
            "ppg_config": 0,
        },
        window_len=300,
        stride_len=100,
        fs=100,
        target_aware_stride=False,
        stride_neg=100,
        stride_pos=100,
        commercial_only=True,
    )

    assert rows
    assert observed_lengths
    assert all(item == (300, 300, 0, 100) for item in observed_lengths)


def test_commercial_only_prewindowed_100hz_uses_raw_windows(monkeypatch):
    first_ppg, first_acc = _raw_window(500)
    ppg = np.stack([first_ppg, first_ppg + 1000], axis=0)
    acc = np.stack([first_acc, first_acc + 10], axis=0)
    observed = []

    monkeypatch.setattr(s03, "load_ppg", lambda _sample: ppg)
    monkeypatch.setattr(s03, "load_acc", lambda _sample: acc)

    def fake_overrides(raw_ppg, raw_acc, frequency, ppg_config):
        observed.append((raw_ppg.copy(), raw_acc.copy(), frequency, ppg_config))
        return OrderedDict((name, float(i)) for i, name in enumerate(EXPECTED_STAGE2_FIELDS))

    monkeypatch.setattr(s03, "extract_commercial_feature_overrides", fake_overrides)
    rows = s03._extract_rows_for_sample(
        {
            "sample_name": "sample",
            "h5_file": "sample.h5",
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
        commercial_only=True,
    )

    assert len(rows) == 2
    assert len(observed) == 2
    for index, (raw_ppg, raw_acc, frequency, ppg_config) in enumerate(observed):
        assert frequency == 100
        assert ppg_config == 0
        np.testing.assert_array_equal(raw_ppg, ppg[index])
        np.testing.assert_array_equal(raw_acc, acc[index])


def test_regular_prewindowed_training_applies_exact_commercial_overrides(monkeypatch):
    raw_ppg, raw_acc = _raw_window(500)
    ppg = np.stack([raw_ppg, raw_ppg + 1000], axis=0)
    acc = np.stack([raw_acc, raw_acc + 10], axis=0)
    observed = []

    monkeypatch.setattr(s03, "load_ppg", lambda _sample: ppg)
    monkeypatch.setattr(s03, "load_acc", lambda _sample: acc)
    monkeypatch.setattr(
        s03,
        "extract_stage2_window",
        lambda *_args, **_kwargs: ({"GREEN_CORR": -1.0}, {}, {}),
    )

    def fake_overrides(window_ppg, window_acc, frequency, ppg_config):
        observed.append((window_ppg.copy(), window_acc.copy(), frequency, ppg_config))
        return OrderedDict((("GREEN_CORR", 10.0),))

    monkeypatch.setattr(s03, "extract_commercial_feature_overrides", fake_overrides)
    rows = s03._extract_rows_for_sample(
        {
            "sample_name": "sample",
            "h5_file": "sample.h5",
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

    assert len(rows) == 2
    assert all(row["GREEN_CORR"] == 10.0 for row in rows)
    assert len(observed) == 2
    for index, (window_ppg, window_acc, frequency, ppg_config) in enumerate(observed):
        assert frequency == 100
        assert ppg_config == 0
        np.testing.assert_array_equal(window_ppg, ppg[index])
        np.testing.assert_array_equal(window_acc, acc[index])
