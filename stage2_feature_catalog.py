"""Governed Stage2 feature catalog for ranking, training, and C deployment."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Mapping


FEATURE_POOL_VERSION = "stage2_interpretable_v8"

COMMERCIAL_8_FEATURE_MAPPING = OrderedDict([
    ("GREEN_CORR", "GREEN_CORR"),
    ("GREEN_AC", "COMM_GREEN_AC"),
    ("AMB_AC", "COMM_AMB_AC"),
    ("ACC_YSUM", "ACC_MAG_MEAN"),
    ("GREEN_DC", "GREEN_DC_MEDIAN"),
    ("AMB_DC", "AMBX_DC_MEDIAN"),
    ("GREEN_XCORR", "GREEN_AUTO_CORR_PEAK"),
    ("FFT_PEAK_MEDIAN_RATIO", "GREEN_FFT_PEAK_MEDIAN_RATIO"),
])

ROW_METADATA_FIELDS = {
    "sample_name",
    "h5_file",
    "target",
    "start_100hz",
    "start_sec",
    "window_index",
}

DIAGNOSTIC_ONLY_FIELDS = {
    "feature_pool_version",
    "TOTAL_INVALID_COUNT",
    "PPG_INVALID_COUNT",
    "GREEN_INVALID_COUNT",
    "ACC_AVAILABLE",
    "AMB_STAGE1_RATIO",
    "AMB_STAGE1_PASS",
    "IR_DC_LEVEL",
}

REMOVED_FEATURES = {
    "SIG_LEN": "constant for a fixed window configuration",
    "SIG_SEC": "constant for a fixed window configuration",
    "G_TOP2_CHANNEL_COUNT": "constant by construction",
    "TOTAL_INVALID_COUNT": "data-quality diagnostic; not a liveness signal",
    "PPG_INVALID_COUNT": "data-quality diagnostic; not a liveness signal",
    "GREEN_INVALID_COUNT": "data-quality diagnostic; not a liveness signal",
    "G_TOP2_WORST_IDX": "channel identity shortcut",
    "G_MIN_CHANNEL_ID": "channel identity shortcut",
    "G_DROPOUT_ANGLE": "absolute sensor direction shortcut",
    "ACC_DOM_AXIS": "absolute orientation shortcut",
    "G_mean_acdc": "exact alias of GREEN_AC_DC_RATIO",
    "GREEN_DC": "exact alias of GREEN_DC_MEDIAN",
    "GREEN_XCORR": "exact alias of GREEN_AUTO_CORR_PEAK",
    "FFT_PEAK_MEDIAN_RATIO": "exact alias of GREEN_FFT_PEAK_MEDIAN_RATIO",
    "AMB_DC": "exact alias of AMBX_DC_MEDIAN",
    "AMB_FFT_PEAK_MEDIAN_RATIO": "exact alias of AMBX_FFT_PEAK_MEDIAN_RATIO",
    "AMB_DOM_FREQ": "exact alias of AMBX_DOM_FREQ",
    "ACC_YSUM": "exact alias of ACC_MAG_MEAN",
    "G_TOP2_SWITCH_RATE": "deterministic complement of G_TOP2_RANK_STABILITY",
    "GREEN_SAT_FRAC": "window-relative maximum is not a hardware saturation limit",
    "ACC_ENERGY_TO_GREEN_AC": "window-length dependent and dimensionally unstable",
    "ACC_STILL_GREEN_MISMATCH": "unbounded near zero ACC derivative energy",
    "SQI_FLAT_RATIO": "deterministic mean of GREEN_FLAT_RATIO and AMB_FLAT_RATIO",
    "SQI_SPIKE_RATIO": "deterministic mean of GREEN_SPIKE_RATIO and AMB_SPIKE_RATIO",
    "GREEN_FFT_peak_width_Hz": "half-height width is unstable at 5-second spectral resolution",
    "GREEN_FFT_SNR": "short-window leakage makes the outside-band denominator unstable",
    "G_SPATIAL_STABILITY_SCORE": "deterministic composite of retained support, correlation, and imbalance features",
    "GREEN_AMB_LEAK": "deterministic product of retained ambient-green correlation and AC ratio",
    "ACC_STILL_SCORE": "deterministic monotonic transform of ACC_REL_MOTION",
    "ACC_STILL_GREEN_STABILITY": "deterministic composite of retained motion and green stability features",
}


def _ops(*values: str) -> list[str]:
    return list(values)


def _record(
    group: str,
    preprocessing: str,
    formula: str,
    c_operators: Iterable[str],
    *,
    unit: str = "dimensionless",
    signal_source: str = "green",
    buffer_samples: int = 125,
    fft: bool = False,
    bounded_range=None,
    accumulator: str = "float64",
    scale_dependent: bool = False,
    numerical_guard: str = "finite replacement; guarded denominator/correlation; non-finite output becomes zero",
    c_abs_tolerance: float = 1e-6,
    c_rel_tolerance: float = 1e-5,
    deployment_cost: float | None = None,
    risk_flags: Iterable[str] = (),
    commercial_8_member: bool = False,
    commercial_original_name: str | None = None,
) -> dict:
    if deployment_cost is None:
        deployment_cost = {
            "signal_quality": 0.8,
            "green_contact": 1.0,
            "ambient_contact": 1.0,
            "green_top2_contact": 1.1,
            "pulse_shape": 1.2,
            "green_spatial": 1.4,
            "ambient_cross": 1.4,
            "spatial_coupling": 1.6,
            "acc_motion": 1.6,
            "acc_green_coupling": 1.8,
            "frequency": 2.8,
            "acquisition_context": 0.1,
        }.get(str(group), 2.0)
    return {
        "group": str(group),
        "preprocessing": str(preprocessing),
        "formula": str(formula),
        "c_operators": list(c_operators),
        "unit": str(unit),
        "signal_source": str(signal_source),
        "buffer_samples": int(buffer_samples),
        "fft": bool(fft),
        "bounded_range": bounded_range,
        "accumulator": str(accumulator),
        "scale_dependent": bool(scale_dependent),
        "numerical_guard": str(numerical_guard),
        "c_abs_tolerance": float(c_abs_tolerance),
        "c_rel_tolerance": float(c_rel_tolerance),
        "deployment_cost": float(deployment_cost),
        "risk_flags": list(risk_flags),
        "commercial_8_member": bool(commercial_8_member),
        "commercial_original_name": (
            str(commercial_original_name) if commercial_original_name is not None else None
        ),
    }


FEATURE_CATALOG: "OrderedDict[str, dict]" = OrderedDict()


def _add(name: str, **record) -> None:
    if name in FEATURE_CATALOG:
        raise ValueError(f"duplicate Stage2 feature catalog entry: {name}")
    FEATURE_CATALOG[name] = _record(**record)


_add(
    "mode",
    group="acquisition_context",
    preprocessing="acquisition_context",
    formula="integer acquisition mode selected by detect_green_mode (0, 1, or 2)",
    c_operators=_ops("identity", "integer_compare"),
    unit="category_code",
    signal_source="acquisition_mode",
    buffer_samples=0,
    accumulator="int32",
    bounded_range=[0.0, 2.0],
    deployment_cost=0.1,
    risk_flags=("hardware_shortcut", "cross_mode_generalization"),
)


for _name, _source, _formula in [
    ("GREEN_FLAT_RATIO", "green", "fraction(abs(diff(raw_green)) <= scale_floor)"),
    ("GREEN_SPIKE_RATIO", "green", "fraction(abs(diff(raw_green)) > 6*MAD(abs(diff)))"),
    ("AMB_FLAT_RATIO", "ambient", "fraction(abs(diff(raw_ambient)) <= scale_floor)"),
    ("AMB_SPIKE_RATIO", "ambient", "fraction(abs(diff(raw_ambient)) > 6*MAD(abs(diff)))"),
]:
    _add(
        _name,
        group="signal_quality",
        preprocessing="quality_raw",
        formula=_formula,
        c_operators=_ops("finite_replace", "difference", "median", "count"),
        signal_source=_source,
        bounded_range=[0.0, 1.0],
        accumulator="float32",
    )


def _add_channel_family(prefix: str, source: str, group: str) -> None:
    raw = f"{source}_raw"
    pulse = f"{source}_pulse"
    for suffix, formula, unit, scale_dependent, ops in [
        ("DC_MEDIAN", f"median({raw})", "ADC", True, _ops("median")),
        ("DC_IQR", f"P75({raw}) - P25({raw})", "ADC", True, _ops("percentile")),
        ("AC_RMS", f"sqrt(mean({pulse}^2))", "ADC", True, _ops("sum_squares", "sqrt")),
        ("AC_MAD", f"median(abs({pulse} - median({pulse})))", "ADC", True, _ops("median", "absolute")),
        ("AC_DC_RATIO", f"RMS({pulse}) / guarded_abs(median({raw}))", "ratio", False, _ops("median", "sum_squares", "safe_ratio")),
        ("DERIV_MAD", f"MAD(diff({pulse}))", "ADC/sample", True, _ops("difference", "median", "absolute")),
    ]:
        _add(
            f"{prefix}_{suffix}",
            group=group,
            preprocessing="pulse_detrended" if suffix.startswith(("AC_", "DERIV")) else "contact_raw",
            formula=formula,
            c_operators=ops,
            unit=unit,
            signal_source=source,
            scale_dependent=scale_dependent,
        )


for _name, _source, _group, _formula in [
    ("GREEN_ROBUST_RANGE_RATIO", "green", "green_contact", "(P95(green_raw)-P5(green_raw))/guarded_abs(median(green_raw))"),
    ("GREEN_SEG_ACDC_CV", "green", "green_contact", "CV(segment MAD(diff(raw))/abs(segment median(raw)))"),
    ("AMB_ROBUST_RANGE_RATIO", "ambient", "ambient_contact", "(P95(ambient_raw)-P5(ambient_raw))/guarded_abs(median(ambient_raw))"),
    ("AMB_SEG_ACDC_CV", "ambient", "ambient_contact", "CV(segment MAD(diff(raw))/abs(segment median(raw)))"),
    ("GTOP2_ROBUST_RANGE_RATIO", "green_top2", "green_top2_contact", "(P95(top2_raw)-P5(top2_raw))/guarded_abs(median(top2_raw))"),
    ("GTOP2_SEG_ACDC_CV", "green_top2", "green_top2_contact", "CV(segment MAD(diff(raw))/abs(segment median(raw)))"),
    ("GTOP2_HALF_ACDC_DELTA", "green_top2", "green_top2_contact", "relative difference between half-window AC/DC estimates"),
    ("GTOP2_SEG_ACDC_RANGE", "green_top2", "green_top2_contact", "relative range of three segment AC/DC estimates"),
]:
    _add(
        _name,
        group=_group,
        preprocessing="contact_raw",
        formula=_formula,
        c_operators=_ops("percentile", "median", "difference", "safe_ratio", "segment_loop"),
        unit="ratio",
        signal_source=_source,
    )

_add_channel_family("GREEN", "green", "green_contact")
_add_channel_family("AMBX", "ambient", "ambient_contact")
_add_channel_family("GTOP2", "green_top2", "green_top2_contact")

for _name, _source, _group, _commercial_name in [
    ("COMM_GREEN_AC", "green", "green_contact", "GREEN_AC"),
    ("COMM_AMB_AC", "ambient", "ambient_contact", "AMB_AC"),
]:
    _add(
        _name,
        group=_group,
        preprocessing="pulse_detrended",
        formula=f"0.5*RMS({_source}_pulse) + 0.5*1.4826*MAD({_source}_pulse)",
        c_operators=_ops("sum_squares", "sqrt", "median", "absolute", "weighted_sum"),
        unit="ADC",
        signal_source=_source,
        scale_dependent=True,
        commercial_8_member=True,
        commercial_original_name=_commercial_name,
    )


for _name, _source, _formula, _fft in [
    ("GREEN_BAND_ENERGY_RATIO", "green", "energy(0.7-3Hz)/energy(0.5-5Hz)", True),
    ("GREEN_FFT_PEAK_MEDIAN_RATIO", "green", "max(in_band_spectrum)/median(in_band_spectrum)", True),
    ("GREEN_DOM_FREQ", "green", "frequency of maximum 0.5-5Hz spectral magnitude", True),
    ("GREEN_AUTO_CORR_PEAK", "green", "max normalized autocorrelation over 40-180 bpm lags", False),
    ("GTOP2_BAND_ENERGY_RATIO", "green_top2", "energy(0.7-3Hz)/energy(0.5-5Hz)", True),
    ("GTOP2_FFT_PEAK_MEDIAN_RATIO", "green_top2", "max(in_band_spectrum)/median(in_band_spectrum)", True),
    ("GTOP2_DOM_FREQ", "green_top2", "frequency of maximum 0.5-5Hz spectral magnitude", True),
    ("GTOP2_AUTO_CORR_PEAK", "green_top2", "max normalized autocorrelation over 40-180 bpm lags", False),
    ("AMB_BAND_ENERGY_RATIO", "ambient", "energy(0.7-3Hz)/energy(0.5-5Hz)", True),
    ("AMBX_FFT_PEAK_MEDIAN_RATIO", "ambient", "max(in_band_spectrum)/median(in_band_spectrum)", True),
    ("AMBX_DOM_FREQ", "ambient", "frequency of maximum 0.5-5Hz spectral magnitude", True),
    ("AMBX_AUTO_CORR_PEAK", "ambient", "max normalized autocorrelation over 40-180 bpm lags", False),
]:
    _add(
        _name,
        group="frequency",
        preprocessing="pulse_detrended",
        formula=_formula,
        c_operators=_ops("hamming", "rfft", "sum_squares", "argmax") if _fft else _ops("autocorrelation", "argmax"),
        unit="Hz" if _name.endswith(("DOM_FREQ", "width_Hz")) else "ratio",
        signal_source=_source,
        fft=_fft,
        accumulator="float64",
    )

for _name, _formula, _ops_list in [
    ("GREEN_CORR", "corr(green_pulse, moving_average(green_pulse, 0.15s))", _ops("moving_average", "correlation")),
    ("GTOP2_zero_cross_rate", "fraction of adjacent top2_pulse samples crossing the median", _ops("median", "sign", "count")),
    ("GTOP2_abs_diff_ratio", "mean(abs(diff(top2_pulse)))/mean(abs(top2_pulse-median))", _ops("difference", "absolute", "mean", "safe_ratio")),
]:
    _add(
        _name,
        group="pulse_shape",
        preprocessing="pulse_detrended",
        formula=_formula,
        c_operators=_ops_list,
        signal_source="green" if _name == "GREEN_CORR" else "green_top2",
    )

_add(
    "GTOP2_ROBUST_SKEWNESS",
    group="pulse_shape",
    preprocessing="pulse_detrended",
    formula="(P90(top2_pulse)+P10(top2_pulse)-2*P50(top2_pulse))/guarded(P90-P10)",
    c_operators=_ops("percentile", "safe_ratio"),
    signal_source="green_top2",
    bounded_range=[-1.0, 1.0],
)
_add(
    "GTOP2_SPECTRAL_ENTROPY",
    group="frequency",
    preprocessing="pulse_detrended",
    formula="normalized Shannon entropy of top2 spectral power over 0.5-5Hz",
    c_operators=_ops("hamming", "rfft", "sum_squares", "log", "safe_ratio"),
    signal_source="green_top2",
    bounded_range=[0.0, 1.0],
    fft=True,
    accumulator="float64",
)

for _name, _formula, _bounded in [
    ("G_imbalance_mean", "mean(std(g1,g2,g3)/guarded_abs(mean(g1,g2,g3)))", None),
    ("G_imbalance_p90", "P90(std(g1,g2,g3)/guarded_abs(mean(g1,g2,g3)))", None),
    ("G_imbalance_iqr", "IQR(std(g1,g2,g3)/guarded_abs(mean(g1,g2,g3)))", None),
    ("G_rangeNorm_mean", "mean((max(g)-min(g))/sum(abs(g)))", None),
    ("G_rangeNorm_p90", "P90((max(g)-min(g))/sum(abs(g)))", None),
    ("G_ch_dc_cv", "std(channel_dc)/guarded_abs(mean(channel_dc))", None),
    ("G_ch_dc_max_min_ratio", "max(abs(channel_dc))/guarded_min(abs(channel_dc))", None),
    ("GCH_AC_RANGE_RATIO", "(max(channel_ac)-min(channel_ac))/guarded_mean(channel_ac)", None),
    ("G_bp_corr_mean", "mean(pairwise green pulse correlations)", [-1.0, 1.0]),
    ("G_bp_corr_min", "minimum pairwise green pulse correlation", [-1.0, 1.0]),
    ("G_bp_corr_std", "std(pairwise green pulse correlations)", [0.0, 1.0]),
    ("G_2OF3_AC_SUPPORT", "count(channel_ac >= 0.5*max_ac)/3", [0.0, 1.0]),
    ("G_TOP2_TO_ALL_AC_RATIO", "max_pair_ac_sum/guarded_sum(all_ac), averaging equally optimal pair views on ties", [0.0, 1.0]),
    ("G_TOP2_CORR_MIN", "median pair correlation over all equally maximal two-zone AC pairs", [-1.0, 1.0]),
    ("G_WEAK_CHANNEL_GAP", "(max_pair_ac_mean-min_ac)/guarded_max_pair_ac_mean", [0.0, 1.0]),
    ("G_TOP1_TO_TOP2_AC_RATIO", "max_ac/guarded_max_pair_ac_mean", [1.0, 2.0]),
    ("G_TOP2_RANK_STABILITY", "1-fraction(segments whose maximal-pair set differs from the global maximal-pair set)", [0.0, 1.0]),
]:
    _add(
        _name,
        group="green_spatial",
        preprocessing="contact_raw" if "corr" not in _name.lower() else "pulse_detrended",
        formula=_formula,
        c_operators=_ops("three_channel_loop", "median", "sort3", "safe_ratio", "correlation"),
        signal_source="green_3ch",
        bounded_range=_bounded,
    )

_add(
    "G_2OF3_PERIODICITY",
    group="green_spatial",
    preprocessing="pulse_detrended",
    formula="median of three symmetric-zone autocorrelation peaks over 40-180 bpm lags",
    c_operators=_ops("three_channel_loop", "autocorrelation", "argmax", "median"),
    signal_source="green_3zone",
    bounded_range=[-1.0, 1.0],
    deployment_cost=1.8,
)
_add(
    "G_ZONE_LAG_RMS_SEC",
    group="green_spatial",
    preprocessing="pulse_detrended",
    formula="RMS absolute bounded cross-correlation peak lag across all three zone pairs",
    c_operators=_ops("three_channel_loop", "bounded_lag_loop", "correlation", "sum_squares", "sqrt"),
    unit="s",
    signal_source="green_3zone",
    bounded_range=[0.0, 0.4],
    deployment_cost=1.8,
)

for _name, _group, _source, _formula, _ops_list, _unit, _bounded, _cost, _risks in [
    (
        "GMEDIAN_AC_DC_RATIO", "green_contact", "green_median",
        "RMS(pointwise_median(zone_pulse))/guarded_abs(median(pointwise_median(zone_raw)))",
        _ops("median3", "median", "sum_squares", "sqrt", "safe_ratio"),
        "ratio", None, 1.2, (),
    ),
    (
        "GMEDIAN_CORR", "pulse_shape", "green_median",
        "corr(pointwise_median(zone_pulse), moving_average(pointwise_median(zone_pulse), 0.15s))",
        _ops("median3", "moving_average", "correlation"),
        "correlation", [-1.0, 1.0], 1.2, (),
    ),
    (
        "GMEDIAN_AUTO_CORR_PEAK", "frequency", "green_median",
        "maximum normalized autocorrelation of pointwise median pulse over 40-180 bpm lags",
        _ops("median3", "autocorrelation", "argmax"),
        "correlation", [-1.0, 1.0], 1.5, (),
    ),
    (
        "GMEDIAN_FFT_PEAK_MEDIAN_RATIO", "frequency", "green_median",
        "max(in-band median-pulse spectrum)/median(in-band median-pulse spectrum)",
        _ops("median3", "hamming", "rfft", "median", "safe_ratio"),
        "ratio", None, 2.8, (),
    ),
    (
        "GTOP2_CORR", "pulse_shape", "green_top2",
        "corr(tie-aware maximal-pair composite pulse, moving_average(composite, 0.15s))",
        _ops("moving_average", "correlation"),
        "correlation", [-1.0, 1.0], 1.0, (),
    ),
    (
        "G_TOP2_ALL_CORR", "green_spatial", "green_3zone",
        "corr(tie-aware maximal-pair composite pulse, mean(zone_pulse))",
        _ops("sort3", "mean", "correlation"),
        "correlation", [-1.0, 1.0], 1.2, (),
    ),
    (
        "G_WEAK_TO_TOP2_CORR", "green_spatial", "green_3zone",
        "median corr(lowest-AC tied zones, tie-aware maximal-pair composite pulse)",
        _ops("sort3", "correlation"),
        "correlation", [-1.0, 1.0], 1.2, (),
    ),
    (
        "G_ZONE_DOM_FREQ_MAD_HZ", "green_spatial", "green_3zone",
        "mean absolute deviation of valid periodic zone dominant frequencies from their median over 0.5-5Hz",
        _ops("three_channel_loop", "hamming", "rfft", "argmax", "median", "absolute", "mean"),
        "Hz", [0.0, 4.5], 2.9, ("short_window_frequency", "frequency_validity_gate"),
    ),
    (
        "G_ZONE_HR_SUPPORT_RATIO", "green_spatial", "green_3zone",
        "fraction of all three zones with valid periodic evidence and dominant frequency within 0.20Hz of the valid-zone median",
        _ops("three_channel_loop", "hamming", "rfft", "argmax", "median", "absolute", "count"),
        "ratio", [0.0, 1.0], 2.9, ("short_window_frequency", "frequency_validity_gate"),
    ),
    (
        "G_PAIR_PERIODICITY_MAX", "green_spatial", "green_3zone_pairs",
        "maximum autocorrelation peak across all three pair-mean pulses",
        _ops("three_pair_loop", "autocorrelation", "argmax", "max"),
        "correlation", [-1.0, 1.0], 2.0, (),
    ),
    (
        "G_PAIR_PERIODICITY_MEDIAN", "green_spatial", "green_3zone_pairs",
        "median autocorrelation peak across all three pair-mean pulses",
        _ops("three_pair_loop", "autocorrelation", "argmax", "median"),
        "correlation", [-1.0, 1.0], 2.0, (),
    ),
    (
        "G_PAIR_FREQ_GAP_MIN_HZ", "green_spatial", "green_3zone_pairs",
        "minimum absolute dominant-frequency gap across pairs whose two zones both pass the periodic-evidence gate; zero if no valid pair",
        _ops("three_channel_loop", "hamming", "rfft", "argmax", "three_pair_loop", "difference", "absolute", "min"),
        "Hz", [0.0, 4.5], 2.9, ("short_window_frequency", "frequency_validity_gate"),
    ),
    (
        "G_PAIR_FREQ_GAP_MEDIAN_HZ", "green_spatial", "green_3zone_pairs",
        "median absolute dominant-frequency gap across pairs whose two zones both pass the periodic-evidence gate; zero if no valid pair",
        _ops("three_channel_loop", "hamming", "rfft", "argmax", "three_pair_loop", "difference", "absolute", "median"),
        "Hz", [0.0, 4.5], 2.9, ("short_window_frequency", "frequency_validity_gate"),
    ),
    (
        "G_PAIR_ACDC_MEDIAN", "green_spatial", "green_3zone_pairs",
        "median AC/DC across all three pair-mean signals",
        _ops("three_pair_loop", "median", "sum_squares", "sqrt", "safe_ratio"),
        "ratio", None, 1.4, (),
    ),
    (
        "G_PAIR_AMB_ABS_CORR_MIN", "ambient_cross", "green_3zone_pairs+ambient",
        "minimum abs corr(pair-mean pulse, ambient pulse) across all three pairs",
        _ops("three_pair_loop", "correlation", "absolute", "min"),
        "correlation", [0.0, 1.0], 1.4, (),
    ),
    (
        "G_PAIR_AMB_ABS_CORR_MEDIAN", "ambient_cross", "green_3zone_pairs+ambient",
        "median abs corr(pair-mean pulse, ambient pulse) across all three pairs",
        _ops("three_pair_loop", "correlation", "absolute", "median"),
        "correlation", [0.0, 1.0], 1.4, (),
    ),
    (
        "G_AMB_RESIDUAL_2OF3_PERIODICITY", "ambient_cross", "green_3zone+ambient",
        "median zone autocorrelation peak after guarded ambient linear projection removal",
        _ops("three_channel_loop", "covariance", "variance", "safe_ratio", "autocorrelation", "median"),
        "correlation", [-1.0, 1.0], 2.0, (),
    ),
    (
        "G_AMB_RESIDUAL_PAIR_CORR_MAX", "ambient_cross", "green_3zone+ambient",
        "maximum pairwise zone correlation after guarded ambient linear projection removal",
        _ops("three_pair_loop", "covariance", "variance", "safe_ratio", "correlation", "max"),
        "correlation", [-1.0, 1.0], 1.8, (),
    ),
    (
        "G_ZONE_PHASE_CONCENTRATION", "frequency", "green_3zone",
        "magnitude of mean unit phasor across valid periodic zones at their median dominant frequency; requires at least two valid zones",
        _ops("three_channel_loop", "hamming", "rfft", "complex_phase", "mean", "absolute"),
        "concentration", [0.0, 1.0], 3.2, ("experimental_high_cost", "short_window_phase", "frequency_validity_gate"),
    ),
    (
        "G_PAIR_SPECTRAL_CONSENSUS", "frequency", "green_3zone_pairs",
        "median pairwise cosine similarity of zone power spectra over 0.5-5Hz for pairs whose two zones pass the periodic-evidence gate; zero if no valid pair",
        _ops("three_pair_loop", "hamming", "rfft", "sum_squares", "cosine_similarity", "median"),
        "similarity", [0.0, 1.0], 3.2, ("experimental_high_cost", "short_window_spectrum", "frequency_validity_gate"),
    ),
]:
    _add(
        _name,
        group=_group,
        preprocessing="cross_signal" if "AMB" in _name else "pulse_detrended",
        formula=_formula,
        c_operators=_ops_list,
        unit=_unit,
        signal_source=_source,
        fft="rfft" in _ops_list,
        bounded_range=_bounded,
        accumulator="float64",
        deployment_cost=_cost,
        risk_flags=_risks,
    )


_POSITION_RISK_FLAGS = (
    "fixed_position",
    "device_shortcut",
    "cross_mode_generalization",
)

for _suffix, _formula_template, _unit, _bounded, _operators, _preprocessing in [
    (
        "DC_CONTRAST",
        "(zone{zone}_dc-median(zone_dc))/guarded_sum_abs(zone{zone}_dc,median(zone_dc))",
        "contrast",
        [-1.0, 1.0],
        _ops("median", "absolute", "safe_ratio"),
        "contact_raw",
    ),
    (
        "AC_CONTRAST",
        "(zone{zone}_ac-median(zone_ac))/guarded_sum(zone{zone}_ac,median(zone_ac))",
        "contrast",
        [-1.0, 1.0],
        _ops("sum_squares", "sqrt", "median", "safe_ratio"),
        "pulse_detrended",
    ),
    (
        "AC_DC_RATIO",
        "RMS(zone{zone}_pulse)/guarded_abs(median(zone{zone}_raw))",
        "ratio",
        None,
        _ops("sum_squares", "sqrt", "median", "absolute", "safe_ratio"),
        "pulse_detrended",
    ),
    (
        "PERIODICITY",
        "maximum normalized autocorrelation of zone{zone}_pulse over 40-180 bpm lags",
        "correlation",
        [-1.0, 1.0],
        _ops("autocorrelation", "argmax"),
        "pulse_detrended",
    ),
    (
        "AMB_ABS_CORR",
        "abs(corr(zone{zone}_pulse,ambient_pulse))",
        "correlation",
        [0.0, 1.0],
        _ops("correlation", "absolute"),
        "cross_signal",
    ),
]:
    for _zone in (1, 2, 3):
        _add(
            f"GZONE{_zone}_{_suffix}",
            group="green_position",
            preprocessing=_preprocessing,
            formula=_formula_template.format(zone=_zone),
            c_operators=_operators,
            unit=_unit,
            signal_source=f"green_zone{_zone}",
            bounded_range=_bounded,
            scale_dependent=False,
            deployment_cost=1.5 if _suffix == "PERIODICITY" else 1.0,
            risk_flags=_POSITION_RISK_FLAGS,
        )

for _name, _formula, _preprocessing in [
    ("corr_Ambient_Gmean", "corr(ambient_raw, green_raw)", "contact_raw"),
    ("GREEN_AMB_BP_CORR", "corr(green_pulse, ambient_pulse)", "cross_signal"),
    ("GREEN_AMB_ENV_CORR", "corr(envelope(green_pulse), envelope(ambient_pulse))", "cross_signal"),
    ("AMB_AC_TO_GREEN_AC", "ambient_pulse_rms/guarded_green_pulse_rms", "cross_signal"),
    ("AMB_DC_TO_GREEN_DC", "ambient_dc/guarded_abs(green_dc)", "cross_signal"),
    ("GREEN_AMB_LEAK_STABILITY", "CV of segment ambient_acdc/green_top2_acdc", "cross_signal"),
    ("GREEN_AMB_SEG_CORR_RANGE", "range of three segment ambient/green_top2 correlations", "cross_signal"),
    ("corr_Gmean_G_imbalance", "corr(green_raw, three-channel imbalance series)", "cross_signal"),
]:
    _add(
        _name,
        group="ambient_cross" if not _name.startswith("corr_Gmean") else "spatial_coupling",
        preprocessing=_preprocessing,
        formula=_formula,
        c_operators=_ops("segment_loop", "correlation", "rms", "safe_ratio"),
        signal_source="green+ambient",
    )

for _name, _formula, _unit, _bounded, _scale_dep in [
    ("ACC_MAG_MEAN", "mean(norm(acc_xyz))", "g", None, True),
    ("ACC_MAG_STD", "std(norm(acc_xyz))", "g", None, True),
    ("ACC_MAG_MAD", "MAD(norm(acc_xyz))", "g", None, True),
    ("ACC_DYNAMIC_STD", "sqrt(var(acc_x)+var(acc_y)+var(acc_z))", "g", None, True),
    ("ACC_BP_RMS", "RMS(acc_magnitude - rolling_median(acc_magnitude,0.8s))", "g", None, True),
    ("ACC_DIFF_MAD", "MAD(diff(acc_magnitude))", "g/sample", None, True),
    ("ACC_MAG_P90", "P90(norm(acc_xyz))", "g", None, True),
    ("ACC_GRAVITY_RATIO", "norm(mean(acc_xyz))/guarded_mean(norm(acc_xyz))", "ratio", [0.0, 1.0], False),
    ("ACC_GREEN_BP_CORR", "abs(corr(acc_motion, green_top2_pulse))", "correlation", [0.0, 1.0], False),
    ("ACC_REL_MOTION", "RMS(acc_motion)/guarded_mean(acc_magnitude)", "ratio", None, False),
    ("ACC_GREEN_REL_MOTION_GAP", "abs(log1p(ACC_REL_MOTION)-log1p(GTOP2_AC_DC_RATIO))", "log-ratio", None, False),
    ("ACC_JERK_TAIL_MEAN_REL", "mean(largest 10% abs(diff(acc_magnitude)))/guarded_mean(acc_magnitude)", "ratio/sample", None, False),
    ("ACC_GREEN_MAX_LAG_CORR", "max abs corr(acc_motion, green_top2_pulse) over +/-0.4s", "correlation", [0.0, 1.0], False),
    ("ACC_GREEN_PSD_SIMILARITY", "cosine similarity of acc-motion and green-top2 spectral power over 0.5-5Hz", "similarity", [0.0, 1.0], False),
]:
    _acc_ops = _ops(
        "vector_norm", "rolling_median", "difference", "rms", "safe_ratio", "correlation"
    )
    if _name == "ACC_JERK_TAIL_MEAN_REL":
        _acc_ops = _ops("vector_norm", "difference", "absolute", "partial_sort", "mean", "safe_ratio")
    elif _name == "ACC_GREEN_MAX_LAG_CORR":
        _acc_ops = _ops("vector_norm", "rolling_median", "bounded_lag_loop", "correlation", "absolute")
    elif _name == "ACC_GREEN_PSD_SIMILARITY":
        _acc_ops = _ops("vector_norm", "rolling_median", "hamming", "rfft", "sum_squares", "cosine_similarity")
    _add(
        _name,
        group="acc_motion" if not _name.startswith("ACC_GREEN") else "acc_green_coupling",
        preprocessing="acc_motion" if not _name.startswith("ACC_GREEN") else "cross_signal",
        formula=_formula,
        c_operators=_acc_ops,
        unit=_unit,
        signal_source="acc" if not _name.startswith("ACC_GREEN") else "acc+green_top2",
        bounded_range=_bounded,
        scale_dependent=_scale_dep,
    )


def model_candidate_names() -> list[str]:
    return list(FEATURE_CATALOG.keys())


def feature_record(name: str) -> Mapping[str, object]:
    try:
        return FEATURE_CATALOG[str(name)]
    except KeyError as exc:
        raise KeyError(f"unknown Stage2 model candidate: {name}") from exc


def feature_group(name: str) -> str:
    record = FEATURE_CATALOG.get(str(name))
    return str(record["group"]) if record is not None else "other"


def is_model_candidate(name: str) -> bool:
    return str(name) in FEATURE_CATALOG


def filter_model_candidates(names: Iterable[str]) -> list[str]:
    names_set = {str(name) for name in names}
    return [name for name in FEATURE_CATALOG if name in names_set]


def selected_catalog(selected_features: Iterable[str]) -> OrderedDict:
    selected = [str(name) for name in selected_features]
    unknown = [name for name in selected if name not in FEATURE_CATALOG]
    if unknown:
        raise ValueError("unknown Stage2 model candidates: " + ", ".join(unknown))
    return OrderedDict((name, dict(FEATURE_CATALOG[name])) for name in selected)


SHARED_PREPROCESSING = OrderedDict([
    ("acquisition_context", {
        "steps": ["reuse_detected_mode", "integer_range_check_0_2"],
        "purpose": "expose the acquisition/channel-layout mode as an optional governed feature",
    }),
    ("quality_raw", {
        "steps": ["finite_replace"],
        "purpose": "measure flatness and isolated spikes before repair",
    }),
    ("contact_raw", {
        "steps": ["finite_replace", "isolated_spike_repair"],
        "purpose": "preserve contact level, spatial balance, and real step transitions",
    }),
    ("pulse_detrended", {
        "steps": [
            "finite_replace",
            "isolated_spike_repair",
            "rolling_median_detrend_0.8s",
            "moving_average_0.04s",
        ],
        "purpose": "extract pulsatile, correlation, autocorrelation, and FFT evidence",
    }),
    ("acc_motion", {
        "steps": [
            "finite_replace",
            "vector_magnitude",
            "rolling_median_baseline_0.8s",
            "robust_residual_clip",
        ],
        "purpose": "separate gravity from orientation-invariant motion",
    }),
    ("cross_signal", {
        "steps": ["reuse_preprocessed_inputs", "length_align", "guarded_correlation_or_ratio"],
        "purpose": "combine optical, ambient, spatial, and motion evidence safely",
    }),
])


def build_selected_feature_contract(
    selected_features: Iterable[str],
    *,
    fs: float,
    window_samples: int,
) -> dict:
    selected = selected_catalog(selected_features)
    preprocessing_names = list(OrderedDict(
        (str(record["preprocessing"]), None) for record in selected.values()
    ))
    operators = sorted({
        str(operator)
        for record in selected.values()
        for operator in record["c_operators"]
    })
    return {
        "feature_pool_version": FEATURE_POOL_VERSION,
        "feature_order": list(selected),
        "sample_rate_hz": float(fs),
        "window_samples": int(window_samples),
        "features": selected,
        "shared_preprocessing": OrderedDict(
            (name, dict(SHARED_PREPROCESSING[name]))
            for name in preprocessing_names
        ),
        "operator_inventory": operators,
        "fft_sources": sorted({
            str(record["signal_source"])
            for record in selected.values()
            if bool(record["fft"])
        }),
    }


def validate_candidate_names(names: Iterable[str]) -> None:
    actual = [str(name) for name in names]
    expected = model_candidate_names()
    if actual != expected:
        missing = [name for name in expected if name not in actual]
        extra = [name for name in actual if name not in expected]
        raise ValueError(
            "Stage2 feature pool does not match catalog order; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )


if not 80 <= len(FEATURE_CATALOG) <= 140:
    raise RuntimeError(f"interpretable Stage2 catalog size out of contract: {len(FEATURE_CATALOG)}")

for _commercial_name, _candidate_name in COMMERCIAL_8_FEATURE_MAPPING.items():
    FEATURE_CATALOG[_candidate_name]["commercial_8_member"] = True
    FEATURE_CATALOG[_candidate_name]["commercial_original_name"] = _commercial_name
