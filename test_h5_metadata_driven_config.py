import h5py
import numpy as np
import pytest

import s01_data_split as s01
import s03_extract_feature_pool as s03


def _write_standard_sample(root, name, *, frequency=None, ppg_config=None):
    group = root.create_group(name)
    group.create_dataset("target", data=np.int32(1))
    group.create_dataset("ppg", data=np.zeros((40, 1000), dtype=np.float32))
    if frequency is not None:
        group.create_dataset("frequency", data=np.int32(frequency))
    if ppg_config is not None:
        group.create_dataset("ppg_config", data=np.int32(ppg_config))
    return group


def test_s01_scans_frequency_and_ppg_config_from_standard_h5(tmp_path):
    h5_path = tmp_path / "standard.h5"
    with h5py.File(h5_path, "w") as root:
        _write_standard_sample(root, "sample", frequency=25, ppg_config=1)

    samples, filtered = s01._scan_one_h5(str(h5_path))

    assert len(samples) == 1
    assert samples[0]["frequency"] == 25
    assert samples[0]["ppg_config"] == 1
    assert all(value == 0 for value in filtered.values())


def test_s01_skips_missing_and_invalid_metadata_with_reason_counts(tmp_path, capsys):
    h5_path = tmp_path / "invalid.h5"
    with h5py.File(h5_path, "w") as root:
        _write_standard_sample(root, "missing_frequency", ppg_config=0)
        _write_standard_sample(root, "invalid_frequency", frequency=50, ppg_config=0)
        _write_standard_sample(root, "missing_ppg_config", frequency=25)
        _write_standard_sample(root, "invalid_ppg_config", frequency=100, ppg_config=9)

    samples, filtered = s01._scan_one_h5(str(h5_path))
    output = capsys.readouterr().out

    assert samples == []
    assert filtered["missing_frequency"] == 1
    assert filtered["invalid_frequency"] == 1
    assert filtered["missing_ppg_config"] == 1
    assert filtered["invalid_ppg_config"] == 1
    for reason in (
        "missing_frequency",
        "invalid_frequency",
        "missing_ppg_config",
        "invalid_ppg_config",
    ):
        assert reason in output


def test_s01_accepts_consistent_grouped_child_metadata(tmp_path):
    h5_path = tmp_path / "grouped.h5"
    with h5py.File(h5_path, "w") as root:
        record = root.create_group("record")
        for index in range(8):
            window = record.create_group(f"record_w{index}_1")
            window.create_dataset("ppg", data=np.zeros((40, 125), dtype=np.float32))
            window.create_dataset("frequency", data=np.int32(25))
            window.create_dataset("ppg_config", data=np.int32(2))

    samples, filtered = s01._scan_one_h5(str(h5_path))

    assert len(samples) == 1
    assert samples[0]["frequency"] == 25
    assert samples[0]["ppg_config"] == 2
    assert filtered["inconsistent_metadata"] == 0


def test_s01_skips_inconsistent_grouped_metadata(tmp_path, capsys):
    h5_path = tmp_path / "grouped_inconsistent.h5"
    with h5py.File(h5_path, "w") as root:
        record = root.create_group("record")
        record.create_dataset("frequency", data=np.int32(100))
        for index, config in enumerate((0, 1)):
            window = record.create_group(f"record_w{index}_1")
            window.create_dataset("ppg", data=np.zeros((40, 300), dtype=np.float32))
            window.create_dataset("frequency", data=np.int32(100))
            window.create_dataset("ppg_config", data=np.int32(config))

    samples, filtered = s01._scan_one_h5(str(h5_path))

    assert samples == []
    assert filtered["inconsistent_metadata"] == 1
    assert "inconsistent_metadata" in capsys.readouterr().out


@pytest.mark.parametrize("frequency", [25, 100])
def test_sample_frequency_comes_only_from_metadata(frequency):
    assert s03.get_sample_frequency({"sample_name": "misleading_sleep_25hz", "frequency": frequency}) == frequency


def test_sample_frequency_rejects_missing_or_invalid_metadata():
    with pytest.raises(ValueError, match="frequency"):
        s03.get_sample_frequency({"sample_name": "sleep_25hz"})
    with pytest.raises(ValueError, match="frequency"):
        s03.get_sample_frequency({"frequency": 50})


def test_ppg_config_maps_exact_fixed_green_zones():
    window = np.tile(np.arange(40, dtype=float), (5, 1))

    config0 = s03.get_channels_from_window(window, ppg_config=0)[2:]
    config1 = s03.get_channels_from_window(window, ppg_config=1)[2:]
    config2 = s03.get_channels_from_window(window, ppg_config=2)[2:]

    assert [float(zone[0]) for zone in config0] == [3.0, 4.0, 5.0]
    assert [float(zone[0]) for zone in config1] == [6.0, 7.0, 8.0]
    assert [float(zone[0]) for zone in config2] == [9.0, 10.0, 11.0]


def test_window_feature_extraction_requires_explicit_ppg_config():
    window = np.zeros((125, 40), dtype=float)
    with pytest.raises(ValueError, match="ppg_config"):
        s03.extract_window_features(window, fs=25)


@pytest.mark.parametrize(
    ("frequency", "sample_count", "expected_downsample_calls"),
    [(25, 275, []), (100, 1100, [(100, 25)])],
)
def test_stage2_downsampling_is_controlled_only_by_frequency_metadata(
    monkeypatch, frequency, sample_count, expected_downsample_calls
):
    ppg = np.zeros((sample_count, 40), dtype=float)
    ppg[:, 0] = 4.0e6
    ppg[:, 1] = 1.0e5
    ppg[:, 3:6] = 2.0e6
    downsample_calls = []

    monkeypatch.setattr(s03, "load_ppg", lambda _sample: ppg)
    monkeypatch.setattr(s03, "load_acc", lambda _sample: None)

    def fake_downsample(signal, src_fs=100, tgt_fs=25):
        downsample_calls.append((src_fs, tgt_fs))
        return signal[::4]

    monkeypatch.setattr(s03, "_downsample_ppg", fake_downsample)
    monkeypatch.setattr(
        s03,
        "extract_stage2_window",
        lambda *_args, **_kwargs: (
            {"GREEN_CORR": 1.0},
            {"feature_pool_version": s03.STAGE2_FEATURE_POOL_VERSION},
            {},
        ),
    )

    rows = s03._extract_rows_for_sample(
        {
            "sample_name": f"sample_{frequency}",
            "h5_file": "unused.h5",
            "target": 1,
            "frequency": frequency,
            "ppg_config": 0,
        },
        dc_threshold=0.1e6,
        ac_dc_threshold=1.0,
        window_len=500,
        stride_len=100,
        fs=100,
        target_aware_stride=False,
        stride_neg=100,
        stride_pos=100,
        skip_initial_windows=0,
        use_stage2_ir=False,
    )

    assert len(rows) == 1
    assert downsample_calls == expected_downsample_calls
