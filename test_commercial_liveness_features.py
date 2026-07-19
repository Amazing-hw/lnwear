from collections import OrderedDict

import numpy as np
import pytest

import commercial_liveness_features as commercial
import s03_extract_feature_pool as s03


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
    idx = np.arange(length, dtype=np.int32)
    ppg = np.zeros((length, 6), dtype=np.int32)
    ppg[:, 0] = 4_000_000 + idx
    ppg[:, 1] = 100_000 + 2 * idx
    ppg[:, 3] = 2_000_000 + idx
    ppg[:, 4] = 2_000_001 + 2 * idx
    ppg[:, 5] = 2_000_003 + 3 * idx
    acc = np.column_stack([
        100 + idx,
        -200 + 2 * idx,
        4096 + 3 * idx,
    ]).astype(np.int16)
    return ppg, acc


def test_exact_port_zero_inputs_return_eight_float32_zeros():
    ppg = np.zeros((125, 4), dtype=np.int32)
    acc = np.zeros((125, 3), dtype=np.int16)

    actual = commercial.main(ppg, acc)

    assert actual.shape == (8,)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.zeros(8, dtype=np.float32))


def test_exact_port_dc_uses_absolute_raw_mean():
    ppg = np.zeros((125, 4), dtype=np.int32)
    ppg[:, 0] = -2_000_000
    ppg[:, 3] = 100_000
    acc = np.zeros((125, 3), dtype=np.int16)

    actual = commercial.main(ppg, acc)

    assert actual[4] == np.float32(2_000_000)
    assert actual[5] == np.float32(100_000)


def test_commercial_adapter_uses_raw_stride_and_truncated_three_zone_mean(monkeypatch):
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
    ).astype(np.int32)
    assert captured["ppg"].shape == (125, 4)
    assert captured["ppg"].dtype == np.int32
    np.testing.assert_array_equal(captured["ppg"][:, 0], expected_green)
    np.testing.assert_array_equal(
        captured["ppg"][:, 3], expected_ppg[:, 1].astype(np.int32)
    )
    np.testing.assert_array_equal(captured["ppg"][:, 1:3], 0)
    assert captured["acc"].shape == (125, 3)
    assert captured["acc"].dtype == np.int16
    np.testing.assert_array_equal(captured["acc"], acc[::4])
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

    expected_green = ((ppg[:, 3] + ppg[:, 4] + ppg[:, 5]) / 3.0).astype(np.int32)
    np.testing.assert_array_equal(captured["ppg"][:, 0], expected_green)
    np.testing.assert_array_equal(captured["acc"], acc)


def test_commercial_adapter_truncates_after_combining_fractional_green_zones(monkeypatch):
    ppg = np.zeros((125, 12), dtype=np.int32)
    acc = np.zeros((125, 3), dtype=np.int16)
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

    # Physical zones are 0.5, 0.5, and 2.5. C receives int32(mean(...)) == 1.
    np.testing.assert_array_equal(captured["ppg"][:, 0], np.ones(125, dtype=np.int32))


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
    np.testing.assert_array_equal(first_ppg, ppg[300:800])
    np.testing.assert_array_equal(first_acc, acc[300:800])


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
