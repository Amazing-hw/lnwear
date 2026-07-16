import json
from itertools import permutations
import warnings

import numpy as np
import pandas as pd
import pytest

import s03_extract_feature_pool as s03
import s04_feature_selection as s04
import s05_train_final_model as s05


def _signals(n=125, fs=25.0):
    t = np.arange(n, dtype=float) / fs
    ir = 4.0e6 + 1.0e4 * np.sin(2 * np.pi * 1.2 * t)
    ambient = 1.0e5 + 400.0 * np.sin(2 * np.pi * 0.35 * t)
    g1 = 2.0e6 + 8.0e3 * np.sin(2 * np.pi * 1.2 * t + 0.01)
    g2 = 2.1e6 + 7.5e3 * np.sin(2 * np.pi * 1.2 * t + 0.03)
    g3 = 1.9e6 + 8.5e3 * np.sin(2 * np.pi * 1.2 * t - 0.02)
    ppg = np.column_stack([ir, ambient, np.zeros(n), g1, g2, g3])
    acc = np.column_stack([
        0.02 * np.sin(2 * np.pi * 0.8 * t),
        0.01 * np.cos(2 * np.pi * 0.5 * t),
        1.0 + 0.015 * np.sin(2 * np.pi * 1.1 * t),
    ])
    return ir, ambient, g1, g2, g3, ppg, acc


def test_catalog_exactly_matches_generated_window_candidates():
    import stage2_feature_catalog as catalog

    *_, ppg, acc = _signals()
    features = s03.extract_window_features(
        ppg, fs=25.0, acc_window=acc, ppg_config=0
    )
    expected = catalog.model_candidate_names()

    assert s03.STAGE2_FEATURE_POOL_VERSION == catalog.FEATURE_POOL_VERSION
    assert catalog.FEATURE_POOL_VERSION == "stage2_interpretable_v8"
    assert len(expected) == 126
    assert "mode" in expected
    assert list(features) == expected
    assert all(np.isfinite(value) for value in features.values())
    assert not [name for name in expected if s04.feature_to_group(name) == "other"]


def test_fixed_position_zone_feature_catalog_contract():
    import stage2_feature_catalog as catalog

    expected = [
        f"GZONE{zone}_{suffix}"
        for suffix in (
            "DC_CONTRAST",
            "AC_CONTRAST",
            "AC_DC_RATIO",
            "PERIODICITY",
            "AMB_ABS_CORR",
        )
        for zone in (1, 2, 3)
    ]

    assert [name for name in catalog.model_candidate_names() if name.startswith("GZONE")] == expected
    for name in expected:
        record = catalog.feature_record(name)
        assert record["group"] == "green_position"
        assert record["signal_source"] in {"green_zone1", "green_zone2", "green_zone3"}
        assert record["formula"]
        assert record["unit"]
        assert record["c_operators"]
        assert record["deployment_cost"] > 0.0
        assert {
            "fixed_position",
            "device_shortcut",
            "cross_mode_generalization",
        } <= set(record["risk_flags"])
        if name.endswith(("DC_CONTRAST", "AC_CONTRAST", "PERIODICITY", "AMB_ABS_CORR")):
            assert record["bounded_range"] is not None


def _fixed_position_feature_names():
    return [
        f"GZONE{zone}_{suffix}"
        for suffix in (
            "DC_CONTRAST",
            "AC_CONTRAST",
            "AC_DC_RATIO",
            "PERIODICITY",
            "AMB_ABS_CORR",
        )
        for zone in (1, 2, 3)
    ]


def test_fixed_position_zone_features_match_governed_formulas():
    ir, ambient, g1, g2, g3, _, _ = _signals()
    features = s03.extract_feature_pool_from_window(ir, ambient, g1, g2, g3, fs=25.0)

    raw_zones = [s03._contact_raw_signal(zone) for zone in (g1, g2, g3)]
    pulse_zones = [s03._pulse_signal(zone, 25.0) for zone in raw_zones]
    ambient_pulse = s03._pulse_signal(s03._contact_raw_signal(ambient), 25.0)
    zone_dc = np.asarray([float(np.median(zone)) for zone in raw_zones])
    zone_ac = np.asarray([float(np.sqrt(np.mean(zone ** 2))) for zone in pulse_zones])
    median_dc = float(np.median(zone_dc))
    median_ac = float(np.median(zone_ac))

    for index, zone in enumerate((1, 2, 3)):
        expected_dc_contrast = s03.guarded_ratio(
            zone_dc[index] - median_dc,
            abs(zone_dc[index]) + abs(median_dc),
            scale=zone_dc,
        )
        expected_ac_contrast = s03.guarded_ratio(
            zone_ac[index] - median_ac,
            zone_ac[index] + median_ac,
            scale=zone_ac,
        )
        expected_ac_dc = s03.guarded_ratio(
            zone_ac[index], abs(zone_dc[index]), scale=raw_zones[index]
        )
        expected_periodicity = s03.autocorr_periodicity_features(
            pulse_zones[index], 25.0
        )[0]
        expected_ambient_corr = abs(s03.guarded_corr(pulse_zones[index], ambient_pulse))

        assert features[f"GZONE{zone}_DC_CONTRAST"] == pytest.approx(expected_dc_contrast)
        assert features[f"GZONE{zone}_AC_CONTRAST"] == pytest.approx(expected_ac_contrast)
        assert features[f"GZONE{zone}_AC_DC_RATIO"] == pytest.approx(expected_ac_dc)
        assert features[f"GZONE{zone}_PERIODICITY"] == pytest.approx(expected_periodicity)
        assert features[f"GZONE{zone}_AMB_ABS_CORR"] == pytest.approx(expected_ambient_corr)


def test_fixed_position_zone_features_follow_stable_zone_order():
    ir, ambient, g1, g2, g3, _, _ = _signals()
    baseline = s03.extract_feature_pool_from_window(ir, ambient, g1, g2, g3)
    swapped = s03.extract_feature_pool_from_window(ir, ambient, g2, g1, g3)

    for suffix in (
        "DC_CONTRAST",
        "AC_CONTRAST",
        "AC_DC_RATIO",
        "PERIODICITY",
        "AMB_ABS_CORR",
    ):
        assert swapped[f"GZONE1_{suffix}"] == pytest.approx(baseline[f"GZONE2_{suffix}"])
        assert swapped[f"GZONE2_{suffix}"] == pytest.approx(baseline[f"GZONE1_{suffix}"])
        assert swapped[f"GZONE3_{suffix}"] == pytest.approx(baseline[f"GZONE3_{suffix}"])

    for invariant_name in ("GREEN_AC_DC_RATIO", "GMEDIAN_AC_DC_RATIO", "G_2OF3_PERIODICITY"):
        assert swapped[invariant_name] == pytest.approx(baseline[invariant_name])


@pytest.mark.parametrize("case", ["zero", "constant", "tiny", "one_bad_zone"])
def test_fixed_position_zone_features_are_finite_for_edge_cases(case):
    n = 125
    t = np.arange(n, dtype=float) / 25.0
    ambient = np.zeros(n, dtype=float)
    if case == "zero":
        zones = [np.zeros(n, dtype=float) for _ in range(3)]
    elif case == "constant":
        zones = [np.full(n, value, dtype=float) for value in (1.0e6, 2.0e6, 3.0e6)]
    elif case == "tiny":
        zones = [1.0e-12 * np.sin(2.0 * np.pi * frequency * t) for frequency in (1.0, 1.2, 1.4)]
    else:
        pulse = np.sin(2.0 * np.pi * 1.2 * t)
        zones = [2.0e6 + 8.0e3 * pulse, 2.0e6 + 7.0e3 * pulse, np.full(n, np.nan)]

    features = s03.extract_feature_pool_from_window(
        np.zeros(n), ambient, zones[0], zones[1], zones[2], fs=25.0
    )
    assert all(np.isfinite(features[name]) for name in _fixed_position_feature_names())


def test_three_zone_expansion_is_governed_and_complete():
    import stage2_feature_catalog as catalog

    expected_new = {
        "GTOP2_ROBUST_SKEWNESS",
        "GTOP2_SPECTRAL_ENTROPY",
        "ACC_JERK_TAIL_MEAN_REL",
        "ACC_GREEN_MAX_LAG_CORR",
        "ACC_GREEN_PSD_SIMILARITY",
        "G_2OF3_PERIODICITY",
        "G_ZONE_LAG_RMS_SEC",
    }

    assert len(catalog.model_candidate_names()) == 126
    assert expected_new <= set(catalog.model_candidate_names())
    for name in expected_new:
        record = catalog.feature_record(name)
        assert record["formula"]
        assert record["c_operators"]
        assert record["deployment_cost"] > 0.0
    assert "partial_sort" in catalog.feature_record("ACC_JERK_TAIL_MEAN_REL")["c_operators"]
    assert "bounded_lag_loop" in catalog.feature_record("ACC_GREEN_MAX_LAG_CORR")["c_operators"]
    assert "cosine_similarity" in catalog.feature_record("ACC_GREEN_PSD_SIMILARITY")["c_operators"]


def test_three_zone_robust_expansion_is_governed():
    import stage2_feature_catalog as catalog

    expected = {
        "GMEDIAN_AC_DC_RATIO",
        "GMEDIAN_CORR",
        "GMEDIAN_AUTO_CORR_PEAK",
        "GMEDIAN_FFT_PEAK_MEDIAN_RATIO",
        "GTOP2_CORR",
        "G_TOP2_ALL_CORR",
        "G_WEAK_TO_TOP2_CORR",
        "G_ZONE_DOM_FREQ_MAD_HZ",
        "G_ZONE_HR_SUPPORT_RATIO",
        "G_PAIR_PERIODICITY_MAX",
        "G_PAIR_PERIODICITY_MEDIAN",
        "G_PAIR_FREQ_GAP_MIN_HZ",
        "G_PAIR_FREQ_GAP_MEDIAN_HZ",
        "G_PAIR_ACDC_MEDIAN",
        "G_PAIR_AMB_ABS_CORR_MIN",
        "G_PAIR_AMB_ABS_CORR_MEDIAN",
        "G_AMB_RESIDUAL_2OF3_PERIODICITY",
        "G_AMB_RESIDUAL_PAIR_CORR_MAX",
        "G_ZONE_PHASE_CONCENTRATION",
        "G_PAIR_SPECTRAL_CONSENSUS",
    }

    assert catalog.FEATURE_POOL_VERSION == "stage2_interpretable_v8"
    assert len(catalog.model_candidate_names()) == 126
    assert expected <= set(catalog.model_candidate_names())
    for name in expected:
        record = catalog.feature_record(name)
        assert record["formula"]
        assert record["c_operators"]
        assert record["deployment_cost"] > 0.0
    for name in {"G_ZONE_PHASE_CONCENTRATION", "G_PAIR_SPECTRAL_CONSENSUS"}:
        assert "experimental_high_cost" in catalog.feature_record(name)["risk_flags"]


def test_three_zone_guarded_helpers_remove_ambient_and_reuse_fft_phase():
    fs = 25.0
    t = np.arange(125, dtype=float) / fs
    pulse = np.sin(2.0 * np.pi * 1.2 * t)
    ambient = 0.8 * np.sin(2.0 * np.pi * 0.35 * t + 0.2)
    contaminated = pulse + 2.5 * ambient

    residual = s03._ambient_projection_residual(contaminated, ambient)
    assert np.isfinite(residual).all()
    assert abs(s03.guarded_corr(residual, ambient)) < abs(
        s03.guarded_corr(contaminated, ambient)
    )
    assert s03.guarded_corr(residual, pulse) > 0.95

    caches = [
        s03.compute_fft_cache(np.sin(2.0 * np.pi * 1.2 * t + phase), fs)
        for phase in (0.0, 0.04, -0.03)
    ]
    assert s03._phase_concentration(caches, 1.2) > 0.99
    assert s03._spectral_power_cosine_from_cache(caches[0], caches[1]) > 0.99

    flat_cache = s03.compute_fft_cache(np.zeros_like(t), fs)
    assert s03._phase_concentration([flat_cache] * 3, 1.2) == 0.0
    assert s03._spectral_power_cosine_from_cache(flat_cache, flat_cache) == 0.0


def test_three_zone_robust_features_are_finite_and_permutation_invariant():
    import stage2_feature_catalog as catalog

    ir, ambient, g1, g2, g3, _, _ = _signals()
    names = [
        name for name in catalog.model_candidate_names()
        if name.startswith("GMEDIAN_")
        or name == "GTOP2_CORR"
        or name.startswith("G_TOP2_ALL")
        or name.startswith("G_WEAK_TO_TOP2")
        or name.startswith("G_ZONE_DOM_FREQ")
        or name.startswith("G_ZONE_HR_SUPPORT")
        or name.startswith("G_PAIR_")
        or name.startswith("G_AMB_RESIDUAL_")
        or name == "G_ZONE_PHASE_CONCENTRATION"
    ]
    baseline = s03.extract_feature_pool_from_window(ir, ambient, g1, g2, g3)

    assert len(names) == 20
    assert all(np.isfinite(baseline[name]) for name in names)
    for permuted in permutations([g1, g2, g3]):
        actual = s03.extract_feature_pool_from_window(ir, ambient, *permuted)
        for name in names:
            assert actual[name] == pytest.approx(baseline[name], abs=1e-12)


def test_three_zone_robust_features_handle_one_bad_zone_and_frequency_disagreement():
    n = 125
    fs = 25.0
    t = np.arange(n, dtype=float) / fs
    rng = np.random.default_rng(71)
    pulse = np.sin(2.0 * np.pi * 1.2 * t)
    other_frequency = np.sin(2.0 * np.pi * 2.0 * t)
    ambient_wave = np.sin(2.0 * np.pi * 0.35 * t + 0.3)
    dc = 2.0e6
    ir = np.zeros(n)
    ambient = 1.0e5 + 6000.0 * ambient_wave
    clean_zones = [dc + 8000.0 * pulse, dc + 7600.0 * pulse, dc + 8400.0 * pulse]

    clean = s03.extract_feature_pool_from_window(ir, ambient, *clean_zones, fs=fs)
    bad_zone = dc + 26000.0 * rng.normal(size=n) + 18000.0 * ambient_wave
    unilateral = s03.extract_feature_pool_from_window(
        ir, ambient, clean_zones[0], clean_zones[1], bad_zone, fs=fs
    )
    disagreement = s03.extract_feature_pool_from_window(
        ir, ambient, clean_zones[0], clean_zones[1], dc + 8000.0 * other_frequency, fs=fs
    )

    assert unilateral["GMEDIAN_AUTO_CORR_PEAK"] > unilateral["GREEN_AUTO_CORR_PEAK"]
    assert unilateral["G_PAIR_PERIODICITY_MAX"] > 0.5
    assert unilateral["G_AMB_RESIDUAL_2OF3_PERIODICITY"] > 0.5
    assert disagreement["G_ZONE_DOM_FREQ_MAD_HZ"] > clean["G_ZONE_DOM_FREQ_MAD_HZ"]
    assert disagreement["G_ZONE_HR_SUPPORT_RATIO"] < clean["G_ZONE_HR_SUPPORT_RATIO"]


def test_three_zone_phase_concentration_detects_asynchronous_zones():
    n = 125
    fs = 25.0
    t = np.arange(n, dtype=float) / fs
    dc = 2.0e6
    ir = np.zeros(n)
    ambient = np.zeros(n)
    aligned = [dc + 8000.0 * np.sin(2.0 * np.pi * 1.2 * t + phase)
               for phase in (0.0, 0.03, -0.02)]
    spread = [dc + 8000.0 * np.sin(2.0 * np.pi * 1.2 * t + phase)
              for phase in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)]

    aligned_features = s03.extract_feature_pool_from_window(ir, ambient, *aligned, fs=fs)
    spread_features = s03.extract_feature_pool_from_window(ir, ambient, *spread, fs=fs)

    assert aligned_features["G_ZONE_PHASE_CONCENTRATION"] > 0.95
    assert spread_features["G_ZONE_PHASE_CONCENTRATION"] < 0.2


def test_exact_rms_ties_do_not_make_top2_features_depend_on_zone_order():
    n = 124
    pulse_a = np.tile([1.0, -1.0], n // 2)
    pulse_b = pulse_a.copy()
    pulse_c = np.tile([1.0, 1.0, -1.0, -1.0], n // 4)
    pulses = [pulse_a, pulse_b, pulse_c]
    raw = [2.0e6 + pulse for pulse in pulses]

    values = []
    for order in permutations(range(3)):
        spatial, _, _ = s03._green_spatial_candidates(
            [raw[index] for index in order],
            [pulses[index] for index in order],
            fs=25.0,
        )
        values.append(spatial["G_TOP2_CORR_MIN"])

    assert max(values) - min(values) < 1e-12


def test_invalid_zone_frequency_is_excluded_but_two_zone_consensus_survives():
    fs = 25.0
    n = 125
    t = np.arange(n, dtype=float) / fs
    pulse = np.sin(2.0 * np.pi * 1.2 * t)
    dc = 2.0e6
    flat = np.full(n, dc)
    features = s03.extract_feature_pool_from_window(
        np.zeros(n),
        np.zeros(n),
        dc + 8000.0 * pulse,
        dc + 7600.0 * pulse,
        flat,
        fs=fs,
    )

    assert s03.compute_fft_cache(np.zeros(n), fs)["dom_freq"] == 0.0
    assert features["G_ZONE_HR_SUPPORT_RATIO"] == pytest.approx(2.0 / 3.0)
    assert features["G_PAIR_FREQ_GAP_MIN_HZ"] == pytest.approx(0.0)
    assert features["G_PAIR_FREQ_GAP_MEDIAN_HZ"] == pytest.approx(0.0)
    assert features["G_ZONE_PHASE_CONCENTRATION"] > 0.95
    assert features["G_PAIR_SPECTRAL_CONSENSUS"] > 0.95


def test_tiny_aperiodic_noise_has_no_valid_zone_frequency_support():
    fs = 25.0
    n = 125
    rng = np.random.default_rng(197)
    dc = 2.0e6
    noise = 1.0e-6 * rng.normal(size=n)
    features = s03.extract_feature_pool_from_window(
        np.zeros(n), np.zeros(n), dc + noise, dc + noise, dc + noise, fs=fs
    )

    assert features["G_ZONE_HR_SUPPORT_RATIO"] == 0.0
    assert features["G_ZONE_DOM_FREQ_MAD_HZ"] == 0.0
    assert features["G_ZONE_PHASE_CONCENTRATION"] == 0.0
    assert features["G_PAIR_SPECTRAL_CONSENSUS"] == 0.0


def test_tiny_periodic_zone_is_excluded_from_frequency_consensus():
    fs = 25.0
    n = 125
    t = np.arange(n, dtype=float) / fs
    dc = 2.0e6
    clean_1 = dc + 8000.0 * np.sin(2.0 * np.pi * 1.2 * t)
    clean_2 = dc + 7600.0 * np.sin(2.0 * np.pi * 1.2 * t + 0.02)
    tiny_opposite = dc + 1.0e-6 * np.sin(2.0 * np.pi * 1.2 * t + np.pi)

    features = s03.extract_feature_pool_from_window(
        np.zeros(n), np.zeros(n), clean_1, clean_2, tiny_opposite, fs=fs
    )

    assert features["G_ZONE_HR_SUPPORT_RATIO"] == pytest.approx(2.0 / 3.0)
    assert features["G_PAIR_FREQ_GAP_MIN_HZ"] == pytest.approx(0.0)
    assert features["G_PAIR_FREQ_GAP_MEDIAN_HZ"] == pytest.approx(0.0)
    assert features["G_ZONE_PHASE_CONCENTRATION"] > 0.95
    assert features["G_PAIR_SPECTRAL_CONSENSUS"] > 0.95


def test_raw_green_layouts_normalize_to_same_three_symmetric_zones():
    _, _, g1, g2, g3, _, _ = _signals()
    n = len(g1)
    layout_3 = np.zeros((n, 6), dtype=float)
    layout_3[:, 3:6] = np.column_stack([g1, g2, g3])
    layout_grouped = np.zeros((n, 16), dtype=float)
    for column in (6, 9, 12):
        layout_grouped[:, column] = g1
    for column in (7, 10, 13):
        layout_grouped[:, column] = g2
    for column in (8, 11, 14):
        layout_grouped[:, column] = g3

    zones_3 = s03.get_channels_from_window(layout_3, ppg_config=0)[2:]
    zones_grouped = s03.get_channels_from_window(layout_grouped, ppg_config=2)[2:]

    for actual, expected in zip(zones_grouped, zones_3):
        assert np.allclose(actual, expected)

    assert not np.allclose(zones_3[0], zones_3[1])
    ambient = np.zeros(n, dtype=float)
    ir = np.zeros(n, dtype=float)
    features_3 = s03.extract_feature_pool_from_window(ir, ambient, *zones_3)
    features_9 = s03.extract_feature_pool_from_window(ir, ambient, *zones_grouped)
    for name in _fixed_position_feature_names():
        assert features_9[name] == pytest.approx(features_3[name], abs=1e-12)


def test_new_three_zone_features_are_permutation_invariant():
    import stage2_feature_catalog as catalog

    ir, ambient, g1, g2, g3, _, _ = _signals()
    names = [
        name for name in catalog.model_candidate_names()
        if catalog.feature_record(name)["group"] in {"green_spatial", "spatial_coupling"}
    ]
    baseline = s03.extract_feature_pool_from_window(ir, ambient, g1, g2, g3)

    for permuted in permutations([g1, g2, g3]):
        actual = s03.extract_feature_pool_from_window(ir, ambient, *permuted)
        for name in names:
            assert actual[name] == pytest.approx(baseline[name], abs=1e-12)


def test_new_optical_features_separate_periodic_shape_noise_and_zone_delay():
    n = 125
    fs = 25.0
    t = np.arange(n, dtype=float) / fs
    rng = np.random.default_rng(17)
    clean = np.sin(2.0 * np.pi * 1.2 * t)
    delayed = np.sin(2.0 * np.pi * 1.2 * (t - 0.16))
    noise = rng.normal(size=n)
    dc = 2.0e6
    ir = np.zeros(n)
    ambient = np.zeros(n)

    periodic = s03.extract_feature_pool_from_window(
        ir, ambient, dc + 8000 * clean, dc + 7500 * clean, dc + 8500 * clean, fs=fs
    )
    noisy = s03.extract_feature_pool_from_window(
        ir, ambient, dc + 8000 * noise, dc + 7500 * rng.normal(size=n),
        dc + 8500 * rng.normal(size=n), fs=fs
    )
    asynchronous = s03.extract_feature_pool_from_window(
        ir, ambient, dc + 8000 * clean, dc + 7500 * delayed, dc + 8500 * clean, fs=fs
    )

    assert 0.0 <= periodic["GTOP2_SPECTRAL_ENTROPY"] <= 1.0
    assert noisy["GTOP2_SPECTRAL_ENTROPY"] > periodic["GTOP2_SPECTRAL_ENTROPY"]
    assert periodic["G_2OF3_PERIODICITY"] > noisy["G_2OF3_PERIODICITY"]
    assert asynchronous["G_ZONE_LAG_RMS_SEC"] > periodic["G_ZONE_LAG_RMS_SEC"]
    assert -1.0 <= periodic["GTOP2_ROBUST_SKEWNESS"] <= 1.0


def test_new_acc_features_capture_impulse_delay_and_spectral_motion():
    n = 125
    fs = 25.0
    t = np.arange(n, dtype=float) / fs
    green = np.sin(2.0 * np.pi * 1.2 * t)
    green_raw = 2.0e6 + 8000.0 * green
    same_motion = np.sin(2.0 * np.pi * 1.2 * (t - 0.16))
    other_motion = np.sin(2.0 * np.pi * 2.7 * t)

    def acc_from_motion(motion):
        return np.column_stack([np.zeros(n), np.zeros(n), 1.0 + 0.08 * motion])

    same = s03._acc_candidate_features(acc_from_motion(same_motion), green, green_raw, fs)
    other = s03._acc_candidate_features(acc_from_motion(other_motion), green, green_raw, fs)
    impulsive_acc = acc_from_motion(np.zeros(n))
    impulsive_acc[n // 2, 2] += 2.0
    impulse = s03._acc_candidate_features(impulsive_acc, green, green_raw, fs)
    calm = s03._acc_candidate_features(acc_from_motion(np.zeros(n)), green, green_raw, fs)

    assert same["ACC_GREEN_MAX_LAG_CORR"] > same["ACC_GREEN_BP_CORR"]
    assert same["ACC_GREEN_PSD_SIMILARITY"] > other["ACC_GREEN_PSD_SIMILARITY"]
    assert impulse["ACC_JERK_TAIL_MEAN_REL"] > calm["ACC_JERK_TAIL_MEAN_REL"]
    missing = s03._acc_candidate_features(None, green, green_raw, fs)
    assert all(missing[name] == 0.0 for name in (
        "ACC_JERK_TAIL_MEAN_REL", "ACC_GREEN_MAX_LAG_CORR", "ACC_GREEN_PSD_SIMILARITY"
    ))


def test_commercial_eight_are_mapped_to_independent_governed_formulas():
    import stage2_feature_catalog as catalog

    expected_mapping = {
        "GREEN_CORR": "GREEN_CORR",
        "GREEN_AC": "COMM_GREEN_AC",
        "AMB_AC": "COMM_AMB_AC",
        "ACC_YSUM": "ACC_MAG_MEAN",
        "GREEN_DC": "GREEN_DC_MEDIAN",
        "AMB_DC": "AMBX_DC_MEDIAN",
        "GREEN_XCORR": "GREEN_AUTO_CORR_PEAK",
        "FFT_PEAK_MEDIAN_RATIO": "GREEN_FFT_PEAK_MEDIAN_RATIO",
    }

    assert catalog.COMMERCIAL_8_FEATURE_MAPPING == expected_mapping
    assert len(set(expected_mapping.values())) == 8
    for original_name, governed_name in expected_mapping.items():
        record = catalog.feature_record(governed_name)
        assert record["commercial_8_member"] is True
        assert record["commercial_original_name"] == original_name

    non_member = catalog.feature_record("GREEN_AC_RMS")
    assert non_member["commercial_8_member"] is False
    assert non_member["commercial_original_name"] is None


def test_commercial_ac_candidates_match_governed_pulse_formula():
    ir, ambient, g1, g2, g3, _, _ = _signals()

    features, preprocessed = s03.extract_feature_pool_from_window(
        ir, ambient, g1, g2, g3, fs=25.0, return_preprocessed=True
    )
    green_pulse = preprocessed["g_mean_bp"]
    ambient_pulse = preprocessed["amb_bp"]
    expected_green = (
        0.5 * np.sqrt(np.mean(green_pulse ** 2))
        + 0.5 * 1.4826 * s03.robust_mad(green_pulse)
    )
    expected_ambient = (
        0.5 * np.sqrt(np.mean(ambient_pulse ** 2))
        + 0.5 * 1.4826 * s03.robust_mad(ambient_pulse)
    )

    assert features["COMM_GREEN_AC"] == pytest.approx(expected_green)
    assert features["COMM_AMB_AC"] == pytest.approx(expected_ambient)


def test_catalog_excludes_shortcuts_aliases_and_unstable_formulas():
    import stage2_feature_catalog as catalog

    candidates = set(catalog.model_candidate_names())
    forbidden = {
        "SIG_LEN",
        "SIG_SEC",
        "TOTAL_INVALID_COUNT",
        "PPG_INVALID_COUNT",
        "GREEN_INVALID_COUNT",
        "G_TOP2_CHANNEL_COUNT",
        "G_TOP2_WORST_IDX",
        "G_MIN_CHANNEL_ID",
        "G_DROPOUT_ANGLE",
        "ACC_DOM_AXIS",
        "ACC_YSUM",
        "GREEN_DC",
        "AMB_DC",
        "GREEN_XCORR",
        "FFT_PEAK_MEDIAN_RATIO",
        "G_mean_acdc",
        "G_TOP2_SWITCH_RATE",
        "GREEN_SAT_FRAC",
        "ACC_ENERGY_TO_GREEN_AC",
        "ACC_STILL_GREEN_MISMATCH",
        "SQI_FLAT_RATIO",
        "SQI_SPIKE_RATIO",
        "GREEN_FFT_peak_width_Hz",
        "GREEN_FFT_SNR",
        "G_SPATIAL_STABILITY_SCORE",
        "GREEN_AMB_LEAK",
        "ACC_STILL_SCORE",
        "ACC_STILL_GREEN_STABILITY",
    }

    assert not (candidates & forbidden)
    assert not [name for name in candidates if name.startswith(("G1_", "G2_", "G3_"))]
    assert not [name for name in candidates if name.startswith("G_consensus_")]
    assert not [name for name in candidates if "_X_" in name or "_Y_" in name or "_Z_" in name]


def test_mode_is_model_candidate_while_quality_diagnostics_are_not():
    import stage2_feature_catalog as catalog

    df = pd.DataFrame({
        "sample_name": ["a", "b"],
        "h5_file": ["a.h5", "b.h5"],
        "target": [0, 1],
        "start_100hz": [0, 100],
        "start_sec": [0.0, 1.0],
        "window_index": [0, 1],
        "mode": [1, 2],
        "feature_pool_version": [catalog.FEATURE_POOL_VERSION] * 2,
        "TOTAL_INVALID_COUNT": [0.0, 1.0],
        "PPG_INVALID_COUNT": [0.0, 1.0],
        "GREEN_INVALID_COUNT": [0.0, 1.0],
        "ACC_AVAILABLE": [1.0, 0.0],
        "GREEN_AC_MAD": [1.0, 2.0],
    })

    assert s04.get_feature_cols(df) == ["mode", "GREEN_AC_MAD"]
    assert "mode" not in catalog.DIAGNOSTIC_ONLY_FIELDS
    assert catalog.is_model_candidate("mode")
    assert catalog.feature_record("mode")["group"] == "acquisition_context"
    assert "ACC_AVAILABLE" in catalog.DIAGNOSTIC_ONLY_FIELDS


def test_quality_features_see_spikes_before_pulse_preprocessing_repairs_them():
    ir, ambient, g1, g2, g3, _, _ = _signals()
    clean = s03.extract_feature_pool_from_window(ir, ambient, g1, g2, g3, fs=25.0)
    g1_spike = g1.copy()
    g2_spike = g2.copy()
    g3_spike = g3.copy()
    g1_spike[60] += 2.0e6
    g2_spike[60] += 2.0e6
    g3_spike[60] += 2.0e6
    spiked = s03.extract_feature_pool_from_window(
        ir, ambient, g1_spike, g2_spike, g3_spike, fs=25.0
    )

    assert spiked["GREEN_SPIKE_RATIO"] > clean["GREEN_SPIKE_RATIO"]
    assert abs(spiked["GREEN_DOM_FREQ"] - clean["GREEN_DOM_FREQ"]) <= 0.2


def test_contact_level_path_preserves_real_step_transition():
    ir, ambient, g1, g2, g3, _, _ = _signals()
    baseline = s03.extract_feature_pool_from_window(ir, ambient, g1, g2, g3, fs=25.0)
    step = np.zeros_like(g1)
    step[len(step) // 2:] = 8.0e5
    stepped = s03.extract_feature_pool_from_window(
        ir, ambient, g1 + step, g2 + step, g3 + step, fs=25.0
    )

    assert stepped["GREEN_DC_IQR"] > baseline["GREEN_DC_IQR"] * 10.0


def test_candidate_features_are_finite_on_degenerate_inputs():
    import stage2_feature_catalog as catalog

    n = 125
    ppg = np.zeros((n, 6), dtype=float)
    ppg[:, 1:5] = 1000.0
    ppg[5, 2] = np.nan
    ppg[6, 3] = np.inf
    acc = np.zeros((n, 3), dtype=float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        features = s03.extract_window_features(
            ppg, fs=25.0, acc_window=acc, ppg_config=0
        )

    assert list(features) == catalog.model_candidate_names()
    assert all(np.isfinite(value) for value in features.values())
    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]


def test_every_candidate_has_preprocessing_and_c_contract_metadata():
    import stage2_feature_catalog as catalog

    for name in catalog.model_candidate_names():
        record = catalog.feature_record(name)
        assert record["group"] != "other", name
        assert record["preprocessing"] in {
            "quality_raw",
            "contact_raw",
            "pulse_detrended",
            "acc_motion",
            "cross_signal",
            "acquisition_context",
        }, name
        assert record["formula"], name
        assert record["numerical_guard"], name
        assert record["c_operators"], name
        assert record["accumulator"] in {"int32", "float32", "float64"}, name
        assert record["c_abs_tolerance"] > 0.0, name
        assert record["c_rel_tolerance"] > 0.0, name
        assert 0.0 < record["deployment_cost"] <= 4.0, name


def test_catalog_exports_selected_only_c_contract():
    import stage2_feature_catalog as catalog

    selected = ["GREEN_AC_DC_RATIO", "G_TOP2_CORR_MIN", "ACC_REL_MOTION"]
    payload = catalog.build_selected_feature_contract(selected, fs=25.0, window_samples=125)

    assert payload["feature_pool_version"] == catalog.FEATURE_POOL_VERSION
    assert payload["feature_order"] == selected
    assert list(payload["features"]) == selected
    assert payload["window_samples"] == 125
    assert "pulse_detrended" in payload["shared_preprocessing"]
    assert "safe_ratio" in payload["operator_inventory"]


def test_s08_deployment_formulas_are_catalog_backed_for_every_candidate():
    import s08_run_pipeline as s08
    import stage2_feature_catalog as catalog

    selected = catalog.model_candidate_names()
    formulas = s08.build_selected_feature_formulas(selected)

    assert list(formulas) == selected
    for name in selected:
        record = catalog.feature_record(name)
        assert formulas[name]["formula"] == record["formula"]
        assert formulas[name]["preprocessing"] == record["preprocessing"]
        assert formulas[name]["c_operators"] == record["c_operators"]
        assert formulas[name]["c_abs_tolerance"] == record["c_abs_tolerance"]
        assert formulas[name]["c_rel_tolerance"] == record["c_rel_tolerance"]


def test_s06_deployment_formulas_are_catalog_backed_for_every_candidate():
    import s06_deploy_eval as s06
    import stage2_feature_catalog as catalog

    selected = catalog.model_candidate_names()
    formulas = s06.build_feature_formula_map(selected)

    assert list(formulas) == selected
    for name in selected:
        record = catalog.feature_record(name)
        assert formulas[name]["formula"] == record["formula"]
        assert formulas[name]["preprocessing"] == record["preprocessing"]
        assert formulas[name]["c_operators"] == record["c_operators"]


def test_shared_window_interface_returns_candidates_diagnostics_and_preprocessed():
    import stage2_feature_catalog as catalog

    *_, ppg, acc = _signals()
    features, diagnostics, preprocessed = s03.extract_stage2_window(
        ppg,
        mode=0,
        fs=25.0,
        acc_window=acc,
        use_stage2_ir=False,
    )

    assert list(features) == catalog.model_candidate_names()
    assert diagnostics["feature_pool_version"] == catalog.FEATURE_POOL_VERSION
    assert features["mode"] == 0.0
    assert diagnostics["ACC_AVAILABLE"] == 1.0
    assert "AMB_STAGE1_RATIO" in diagnostics
    assert {"g_top2_bp", "g_top2_raw", "g_mean_bp"} <= set(preprocessed)


def test_mode_zero_preserves_explicit_three_green_channel_layout():
    *_, ppg, acc = _signals()

    features, _, _ = s03.extract_stage2_window(
        ppg,
        mode=0,
        fs=25.0,
        acc_window=acc,
        use_stage2_ir=False,
    )

    assert features["G_ch_dc_cv"] > 0.01
    assert features["G_imbalance_mean"] > 0.01


def test_s03_batch_extraction_calls_shared_window_interface(monkeypatch):
    import stage2_feature_catalog as catalog

    # Seven candidate windows become one retained center window after [3:-3].
    *_, ppg, acc = _signals(n=900, fs=100.0)
    calls = []

    monkeypatch.setattr(s03, "load_ppg", lambda _sample: ppg)
    monkeypatch.setattr(s03, "load_acc", lambda _sample: acc)
    monkeypatch.setattr(s03, "stage1_sample_pass", lambda *_args, **_kwargs: True)

    def fake_shared(window, mode, fs, acc_window, use_stage2_ir):
        calls.append((window.shape, mode, fs, acc_window is not None, use_stage2_ir))
        features = {name: 0.0 for name in catalog.model_candidate_names()}
        diagnostics = {
            "feature_pool_version": catalog.FEATURE_POOL_VERSION,
            "TOTAL_INVALID_COUNT": 0.0,
            "PPG_INVALID_COUNT": 0.0,
            "GREEN_INVALID_COUNT": 0.0,
            "ACC_AVAILABLE": float(acc_window is not None),
            "AMB_STAGE1_RATIO": 0.0,
            "AMB_STAGE1_PASS": 1.0,
            "IR_DC_LEVEL": 0.0,
        }
        return features, diagnostics, {}

    monkeypatch.setattr(s03, "extract_stage2_window", fake_shared)
    rows = s03._extract_rows_for_sample(
        {
            "sample_name": "sample", "h5_file": "x.h5", "target": 1,
            "frequency": 100, "ppg_config": 0,
        },
        dc_threshold=1.0,
        ac_dc_threshold=1.0,
        window_len=300,
        stride_len=100,
        fs=100,
        target_aware_stride=False,
        stride_neg=100,
        stride_pos=100,
        skip_initial_windows=0,
        use_stage2_ir=False,
    )

    assert len(rows) == 1
    assert len(calls) == 1
    assert rows[0]["feature_pool_version"] == catalog.FEATURE_POOL_VERSION
    assert set(catalog.model_candidate_names()) <= set(rows[0])


def test_s04_rejects_missing_or_mismatched_feature_pool_versions():
    import stage2_feature_catalog as catalog

    base = {
        "sample_name": ["a", "b"],
        "target": [0, 1],
        "GREEN_AC_MAD": [1.0, 2.0],
    }
    missing = pd.DataFrame(base)
    with pytest.raises(ValueError, match="rerun s03"):
        s04.validate_feature_pool_frames(missing, missing.copy())

    stale = pd.DataFrame({**base, "feature_pool_version": ["old", "old"]})
    with pytest.raises(ValueError, match=catalog.FEATURE_POOL_VERSION):
        s04.validate_feature_pool_frames(stale, stale.copy())

    current = pd.DataFrame({
        **base,
        "feature_pool_version": [catalog.FEATURE_POOL_VERSION] * 2,
    })
    s04.validate_feature_pool_frames(current, current.copy())


def _ranking_frames():
    import stage2_feature_catalog as catalog

    rows = []
    for sample_idx in range(8):
        target = sample_idx % 2
        for window_idx in range(3):
            row = {
                "sample_name": f"sample_{sample_idx}",
                "target": target,
                "feature_pool_version": catalog.FEATURE_POOL_VERSION,
            }
            for feature_idx, name in enumerate(catalog.model_candidate_names()):
                row[name] = (
                    target * (1.0 + feature_idx * 0.001)
                    + sample_idx * 0.01
                    + window_idx * 0.001
                )
            rows.append(row)
    train = pd.DataFrame(rows)
    valid = train.copy()
    valid["sample_name"] = "valid_" + valid["sample_name"].astype(str)
    return train, valid


def test_full_ranking_covers_catalog_and_does_not_use_valid_labels_for_score(tmp_path):
    import stage2_feature_catalog as catalog

    train, valid = _ranking_frames()
    ranking_a = s04.build_full_feature_ranking(
        train,
        valid,
        catalog.model_candidate_names(),
        removed_map={"high_corr": ["GREEN_AC_RMS"]},
        n_splits=4,
    )
    valid_flipped = valid.copy()
    valid_flipped["target"] = 1 - valid_flipped["target"]
    ranking_b = s04.build_full_feature_ranking(
        train,
        valid_flipped,
        catalog.model_candidate_names(),
        removed_map={"high_corr": ["GREEN_AC_RMS"]},
        n_splits=4,
    )

    assert {item["feature"] for item in ranking_a} == set(catalog.model_candidate_names())
    assert len({item["feature"] for item in ranking_a}) == len(ranking_a)
    assert [item["rank"] for item in ranking_a] == list(range(1, len(ranking_a) + 1))
    assert {
        item["feature"]: item["ranking_score"] for item in ranking_a
    } == {
        item["feature"]: item["ranking_score"] for item in ranking_b
    }
    green_rms = next(item for item in ranking_a if item["feature"] == "GREEN_AC_RMS")
    assert "high_corr" in green_rms["risk_flags"]
    assert green_rms["eligible_for_manual_selection"] is True
    mode_row = next(item for item in ranking_a if item["feature"] == "mode")
    assert {"hardware_shortcut", "cross_mode_generalization"} <= set(mode_row["risk_flags"])

    outputs = s04.export_full_feature_ranking(tmp_path, ranking_a)
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    csv_df = pd.read_csv(outputs["csv"])
    assert payload["feature_pool_version"] == catalog.FEATURE_POOL_VERSION
    assert len(payload["ranking"]) == len(catalog.model_candidate_names())
    assert len(csv_df) == len(catalog.model_candidate_names())


def test_full_ranking_exports_catalog_and_c_completeness_audit(tmp_path):
    import stage2_feature_catalog as catalog

    train, valid = _ranking_frames()
    ranking = s04.build_full_feature_ranking(
        train, valid, catalog.model_candidate_names(), n_splits=4
    )
    outputs = s04.export_full_feature_ranking(tmp_path, ranking)

    assert len(ranking) == 126
    for item in ranking:
        record = catalog.feature_record(item["feature"])
        assert item["commercial_8_member"] == record["commercial_8_member"]
        assert item["commercial_original_name"] == record["commercial_original_name"]
        assert item["signal_source"] == record["signal_source"]
        assert item["buffer_samples"] == record["buffer_samples"]
        assert item["c_operators"] == record["c_operators"]

    completeness = json.loads(outputs["completeness"].read_text(encoding="utf-8"))
    assert completeness["feature_pool_version"] == catalog.FEATURE_POOL_VERSION
    assert completeness["catalog_count"] == 126
    assert completeness["ranked_count"] == 126
    assert completeness["unique_ranked_count"] == 126
    assert completeness["missing_from_ranking"] == []
    assert completeness["extra_in_ranking"] == []
    assert completeness["commercial_8_mapping"] == dict(catalog.COMMERCIAL_8_FEATURE_MAPPING)
