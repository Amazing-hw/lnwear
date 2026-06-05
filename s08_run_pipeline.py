# s08_run_pipeline.py
# -*- coding: utf-8 -*-
"""
主控脚本：一键运行完整训练、搜参、评估和部署导出流程。

推荐一条命令（含 XGBoost 模型搜参；不含商用 baseline 对比、NPZ 缓存导出和 s07 后处理搜参）:
    python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts

这条命令会默认跑到 s06_cb：
    s01 数据切分
    s02 Stage1 固定阈值
    s03 3s/1s Stage2 特征窗口
    s04 特征筛选 + s04_search 候选子集搜索
    s05 XGBoost 训练；默认执行复杂度受限搜参
    s06_opt legacy 状态机优化参考
    s06_eval 用当前已固化/默认状态机做 test 端到端评估
    s06_xpt/s06_feat/s06_plot/s06_cb 导出部署产物、特征脚本、错误图和部署配方

NPZ 缓存导出和 s07 后处理搜参很耗时，默认不跑；需要时显式运行：
    python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --export_window_cache --optimize_postprocess

商用 baseline 对比暂不属于默认全流程；需要时显式运行：
    python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --commercial_compare --stop_after s09_cmp

用法:
    # 主流程运行（含 XGBoost 搜参、评估和部署导出；不含 NPZ/s07 搜参/商用对比）
    python new/s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts

    # 需要时再打开 NPZ 缓存导出和 s07 后处理搜参
    python new/s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --export_window_cache --optimize_postprocess

    # 跳过某些步骤
    python new/s08_run_pipeline.py --skip s02,s03

    # 只跑到特征筛选
    python new/s08_run_pipeline.py --stop_after s04

    # 只复用已有 artifacts 做评估和部署导出
    python new/s08_run_pipeline.py --artifact_dir artifacts --skip s01,s02,s03,s04,s04_search,s05

流程:
    s01: 数据扫描 & train/valid/test 切分
    s02: Stage1 IR DC/ACDC 固定阈值配置
    s03: Stage2 特征池提取（预切窗直接使用；连续时序按 3s/1s 滑窗）
    s04: 稳定性特征筛选
    s05: XGBoost 最终模型训练
    s06_opt:  legacy 状态机参数网格搜索 (--optimize)
    s06_cache: 导出 valid 逐窗缓存（默认不跑；需 --export_window_cache）
    s07_post: FP 敏感后处理搜参（默认不跑；需 --optimize_postprocess）
    s06_eval: 端到端评估
    s06_xpt: 导出部署产物 (--export_deploy)
    s09_cmp: 我们方案 vs 商用方案对比（默认不跑；需 --commercial_compare --stop_after s09_cmp）
"""

import argparse
import glob
import os
import json
import subprocess
import joblib


THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def configure_thread_env():
    """Cap inherited BLAS/OpenMP threads unless the caller already set them."""
    for name, value in THREAD_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)
    return {name: os.environ.get(name) for name in THREAD_ENV_DEFAULTS}


def dataset_has_h5_files(dataset_dir):
    patterns = [os.path.join(dataset_dir, "*.h5")]
    if not os.path.isabs(dataset_dir):
        patterns.append(os.path.join("..", dataset_dir, "*.h5"))
    return any(glob.glob(pattern) for pattern in patterns)

EXTRA_FEATURE_FORMULAS = {
    "GREEN_CORR": {
        "formula": "safe_corr(g_mean_bp, moving_average(g_mean_bp, window=round(0.15*fs)))",
        "intermediate_signals": {"g_mean_bp": "(g1_bp + g2_bp + g3_bp) / 3"},
    },
    "GREEN_AC": {
        "formula": "0.5*sqrt(mean(g_mean_bp^2)) + 0.5*1.4826*robust_mad(g_mean_bp)",
        "intermediate_signals": {"g_mean_bp": "(g1_bp + g2_bp + g3_bp) / 3"},
    },
    "AMB_AC": {
        "formula": "0.5*sqrt(mean(amb_bp^2)) + 0.5*1.4826*robust_mad(amb_bp)",
        "intermediate_signals": {"amb_bp": "preprocessed ambient bandpass signal"},
    },
    "ACC_YSUM": {
        "formula": "mean(sqrt(acc_x^2 + acc_y^2 + acc_z^2))",
        "intermediate_signals": {"acc_mag": "sqrt(acc_x^2 + acc_y^2 + acc_z^2)"},
    },
    "GREEN_DC": {
        "formula": "median(g_mean_raw)",
        "intermediate_signals": {"g_mean_raw": "(g1_raw + g2_raw + g3_raw) / 3"},
    },
    "AMB_DC": {
        "formula": "median(raw ambient input)",
        "intermediate_signals": {"ambient": "raw input ambient window"},
    },
    "GREEN_XCORR": {
        "formula": "max(normalized_autocorr(g_mean_bp)[lag_min:lag_max]) for 40-180 bpm",
        "intermediate_signals": {"g_mean_bp": "(g1_bp + g2_bp + g3_bp) / 3"},
    },
    "FFT_PEAK_MEDIAN_RATIO": {
        "formula": "fft_peak_features(g_mean_bp, 25, 0.5, 5.0)[0]",
        "intermediate_signals": {"g_mean_bp": "(g1_bp + g2_bp + g3_bp) / 3"},
    },
    "GREEN_FFT_harmonic_ratio": {
        "formula": "max_power_near(2*GREEN_DOM_FREQ, +/-0.3Hz) / (GREEN_DOM_FREQ_power + 1e-12)",
        "intermediate_signals": {
            "g_mean_bp": "(g1_bp + g2_bp + g3_bp) / 3",
            "band_spec": "FFT spectrum over 0.5-5Hz",
        },
    },
    "GREEN_FFT_harmonic_present": {
        "formula": "1 if second_harmonic_power > 0.1 * fundamental_power else 0",
        "intermediate_signals": {
            "g_mean_bp": "(g1_bp + g2_bp + g3_bp) / 3",
            "band_spec": "FFT spectrum over 0.5-5Hz",
        },
    },
    "ACC_PPG_coherence_mean": {
        "formula": "mean(coherence(acc_mag, g_mean_bp, fs=25) over 0.5-3Hz)",
        "intermediate_signals": {
            "acc_mag": "sqrt(acc_x^2 + acc_y^2 + acc_z^2)",
            "g_mean_bp": "(g1_bp + g2_bp + g3_bp) / 3",
        },
    },
    "ACC_PPG_coherence_max": {
        "formula": "max(coherence(acc_mag, g_mean_bp, fs=25) over 0.5-3Hz)",
        "intermediate_signals": {
            "acc_mag": "sqrt(acc_x^2 + acc_y^2 + acc_z^2)",
            "g_mean_bp": "(g1_bp + g2_bp + g3_bp) / 3",
        },
    },
    "SIG_LEN": {"formula": "len(window)", "intermediate_signals": {}},
    "SIG_SEC": {"formula": "len(window) / fs", "intermediate_signals": {}},
    "mode": {
        "formula": "green-channel hardware mode supplied by caller",
        "intermediate_signals": {},
    },
    "SQI_FLAT_RATIO": {
        "formula": "mean(diff-flat-ratio over IR, Ambient, GreenMean before artifact removal)",
        "intermediate_signals": {
            "ir": "raw input IR window",
            "ambient": "raw input ambient window",
            "g_mean": "(g1 + g2 + g3) / 3 before preprocessing",
        },
    },
    "SQI_SPIKE_RATIO": {
        "formula": "mean(diff > 6*MAD(diff) ratio over IR, Ambient, GreenMean before artifact removal)",
        "intermediate_signals": {
            "ir": "raw input IR window",
            "ambient": "raw input ambient window",
            "g_mean": "(g1 + g2 + g3) / 3 before preprocessing",
        },
    },
    "IR_ROBUST_RANGE_RATIO": {
        "formula": "(p95(ir_raw)-p5(ir_raw)) / |median(ir_raw)|",
        "intermediate_signals": {"ir_raw": "preprocessed IR raw-clean signal"},
    },
    "GREEN_ROBUST_RANGE_RATIO": {
        "formula": "(p95(g_mean_raw)-p5(g_mean_raw)) / |median(g_mean_raw)|",
        "intermediate_signals": {"g_mean_raw": "(g1_raw + g2_raw + g3_raw) / 3"},
    },
    "AMB_ROBUST_RANGE_RATIO": {
        "formula": "(p95(amb_raw)-p5(amb_raw)) / |median(amb_raw)|",
        "intermediate_signals": {"amb_raw": "preprocessed ambient raw-clean signal"},
    },
    "IR_SEG_ACDC_CV": {
        "formula": "std(AC/DC over three 1s IR segments) / mean(AC/DC)",
        "intermediate_signals": {"ir_raw": "preprocessed IR raw-clean signal"},
    },
    "GREEN_SEG_ACDC_CV": {
        "formula": "std(AC/DC over three 1s GreenMean segments) / mean(AC/DC)",
        "intermediate_signals": {"g_mean_raw": "(g1_raw + g2_raw + g3_raw) / 3"},
    },
    "AMB_SEG_ACDC_CV": {
        "formula": "std(AC/DC over three 1s Ambient segments) / mean(AC/DC)",
        "intermediate_signals": {"amb_raw": "preprocessed ambient raw-clean signal"},
    },
    "GREEN_BAND_ENERGY_RATIO": {
        "formula": "sum(FFT(g_mean_bp)^2 over 0.7-3Hz) / sum(FFT(g_mean_bp)^2 over 0.5-5Hz)",
        "intermediate_signals": {"g_mean_bp": "(g1_bp + g2_bp + g3_bp) / 3"},
    },
    "IR_BAND_ENERGY_RATIO": {
        "formula": "sum(FFT(ir_bp)^2 over 0.7-3Hz) / sum(FFT(ir_bp)^2 over 0.5-5Hz)",
        "intermediate_signals": {"ir_bp": "preprocessed IR bandpass signal"},
    },
    "AMB_BAND_ENERGY_RATIO": {
        "formula": "sum(FFT(amb_bp)^2 over 0.7-3Hz) / sum(FFT(amb_bp)^2 over 0.5-5Hz)",
        "intermediate_signals": {"amb_bp": "preprocessed ambient bandpass signal"},
    },
}


def _json_float_map(values, feature_names):
    out = {}
    for name in feature_names:
        value = values.get(name, 0.0)
        out[name] = float(0.0 if value is None else value)
    return out


def build_selected_feature_formulas(selected_features):
    recipe, *_ = _build_full_feature_recipe(selected_features)
    for name in selected_features:
        if str(recipe.get(name, {}).get("formula", "")).startswith("[") and name in EXTRA_FEATURE_FORMULAS:
            recipe[name] = EXTRA_FEATURE_FORMULAS[name]
    missing = [
        name for name, info in recipe.items()
        if str(info.get("formula", "")).startswith("[")
    ]
    if missing:
        raise ValueError(
            "No deploy formula registered for selected features: "
            + ", ".join(missing)
        )
    return recipe


def _render_selected_feature_extractor(selected_features, fill_values, formulas, scripts_dir):
    order_json = json.dumps(selected_features, ensure_ascii=False, indent=2)
    fill_json = json.dumps(fill_values, ensure_ascii=False, indent=2)
    formulas_json = json.dumps(formulas, ensure_ascii=False, indent=2)
    scripts_dir_json = json.dumps(str(scripts_dir), ensure_ascii=False)

    return f'''# -*- coding: utf-8 -*-
"""Auto-generated selected-feature extractor for watch wearing-liveness.

This reference deployment script exports only the selected model features.
It delegates preprocessing and feature math to s03_extract_feature_pool so
training/evaluation/export stay numerically aligned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import coherence


FEATURE_ORDER = {order_json}
FILL_VALUES = {fill_json}
FEATURE_FORMULAS = {formulas_json}


_SCRIPT_DIR = Path(__file__).resolve().parent
_CANDIDATE_CODE_DIRS = [
    _SCRIPT_DIR,
    _SCRIPT_DIR.parent,
    Path({scripts_dir_json}),
]
for _code_dir in _CANDIDATE_CODE_DIRS:
    if (_code_dir / "s03_extract_feature_pool.py").exists():
        sys.path.insert(0, str(_code_dir))
        break

from s03_extract_feature_pool import (  # noqa: E402
    extract_acc_features,
    extract_acc_ppg_cross_features,
    extract_feature_pool_from_window,
)


def _clean_value(name, value):
    if value is None or not np.isfinite(value):
        return float(FILL_VALUES.get(name, 0.0))
    return float(value)


def _add_acc_ppg_coherence(features, acc, g_mean_bp, fs):
    if acc is None or len(acc) < 16:
        features["ACC_PPG_coherence_mean"] = 0.0
        features["ACC_PPG_coherence_max"] = 0.0
        return
    try:
        acc_arr = np.asarray(acc, dtype=float)
        acc_mag = np.sqrt(np.sum(acc_arr ** 2, axis=1) + 1e-12)
        n = min(len(acc_mag), len(g_mean_bp))
        nperseg = min(32, n // 2)
        if nperseg < 8:
            features["ACC_PPG_coherence_mean"] = 0.0
            features["ACC_PPG_coherence_max"] = 0.0
            return
        freq, cxy = coherence(acc_mag[:n], g_mean_bp[:n], fs=fs, nperseg=nperseg)
        mask = (freq >= 0.5) & (freq <= 3.0)
        if np.any(mask):
            features["ACC_PPG_coherence_mean"] = float(np.mean(cxy[mask]))
            features["ACC_PPG_coherence_max"] = float(np.max(cxy[mask]))
        else:
            features["ACC_PPG_coherence_mean"] = 0.0
            features["ACC_PPG_coherence_max"] = 0.0
    except Exception:
        features["ACC_PPG_coherence_mean"] = 0.0
        features["ACC_PPG_coherence_max"] = 0.0


def extract_feature_dict(ir, ambient, g1, g2, g3, acc=None, fs=25, mode=0):
    """Return selected features as a plain dict in model order."""
    features, preprocessed = extract_feature_pool_from_window(
        ir=ir,
        ambient=ambient,
        g1=g1,
        g2=g2,
        g3=g3,
        fs=fs,
        return_preprocessed=True,
    )
    features.update(extract_acc_features(acc, fs=fs, prefix="ACC"))
    features.update(
        extract_acc_ppg_cross_features(
            acc,
            preprocessed["g_mean_bp"],
            preprocessed["ir_bp"],
            fs=fs,
        )
    )
    _add_acc_ppg_coherence(features, acc, preprocessed["g_mean_bp"], fs)
    features["mode"] = float(mode)

    missing = [name for name in FEATURE_ORDER if name not in features]
    if missing:
        raise KeyError("Selected features missing from s03 extractor: " + ", ".join(missing))

    return {{name: _clean_value(name, features[name]) for name in FEATURE_ORDER}}


def extract_features(ir, ambient, g1, g2, g3, acc=None, fs=25, mode=0):
    """Return the selected feature vector in model order."""
    feature_dict = extract_feature_dict(ir, ambient, g1, g2, g3, acc=acc, fs=fs, mode=mode)
    return [feature_dict[name] for name in FEATURE_ORDER]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 125
    t = np.linspace(0, 5, n, endpoint=False)
    ir = 4.0e6 + 1.0e4 * np.sin(2 * np.pi * 1.2 * t)
    ambient = 1.0e5 + 500.0 * np.sin(2 * np.pi * 0.4 * t)
    g1 = 2.0e6 + 8.0e3 * np.sin(2 * np.pi * 1.2 * t + 0.01)
    g2 = 2.1e6 + 7.5e3 * np.sin(2 * np.pi * 1.2 * t + 0.03)
    g3 = 1.9e6 + 8.5e3 * np.sin(2 * np.pi * 1.2 * t - 0.02)
    acc = rng.normal(0, 0.01, (n, 3))
    vec = extract_features(ir, ambient, g1, g2, g3, acc=acc, fs=25)
    print(f"Feature vector: {{len(vec)}} values")
    for i, (name, value) in enumerate(zip(FEATURE_ORDER, vec)):
        print(f"{{i:02d}} {{name}} = {{value:.8g}}")
'''


def export_feature_extractor_script(artifact_dir):
    """Export a compact extractor for the actual selected deployment features."""
    bp = os.path.join(artifact_dir, "model_bundle.pkl")
    if not os.path.exists(bp):
        print("[WARN] model_bundle.pkl not found, skip feature extractor script")
        return None

    bundle = joblib.load(bp)
    selected = list(bundle["feature_names"])
    fill_values = _json_float_map(bundle.get("fill_values", {}), selected)
    formulas = build_selected_feature_formulas(selected)
    script_text = _render_selected_feature_extractor(
        selected,
        fill_values,
        formulas,
        scripts_dir=SCRIPTS_DIR,
    )

    out_path = os.path.join(artifact_dir, "deploy_feature_extractor.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(script_text)

    formula_path = os.path.join(artifact_dir, "deploy_selected_feature_formulas.json")
    with open(formula_path, "w", encoding="utf-8") as f:
        json.dump(formulas, f, indent=2, ensure_ascii=False)

    print(f"[OK] selected deploy feature extractor -> {out_path}")
    print(f"[OK] selected deploy formulas -> {formula_path}")
    return out_path
import sys
import time
from datetime import timedelta

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def _script_path(name):
    return os.path.join(SCRIPTS_DIR, f"{name}.py")


def _run(name, cmd, dry_run=False):
    """执行一个子步骤。返回 True/False。"""
    print(f"\n{'─' * 70}")
    print(f"[RUN] {name}")
    print(f"  {cmd}")
    if dry_run:
        print("  (dry-run, skipped)")
        return True
    t0 = time.time()
    rc = subprocess.call(cmd, shell=True)
    dt = time.time() - t0
    if rc == 0:
        print(f"[OK] {name}  [{timedelta(seconds=int(dt))}]")
        return True
    else:
        print(f"[FAIL] {name}  FAILED (exit={rc})")
        return False


def _load_eval_details(artifact_dir, split="test", method="state_machine"):
    import json as _json
    import os as _os
    eval_path = _os.path.join(artifact_dir, f"end_to_end_eval_{split}_{method}.json")
    if not _os.path.exists(eval_path):
        return None
    with open(eval_path, "r", encoding="utf-8") as f:
        return _json.load(f).get("details", [])


def _build_full_feature_recipe(selected_features):
    """
    为每个入选特征构建完整的自包含计算配方。
    从 25Hz 原始窗口 → 预处理 → 中间信号 → 特征值，全链路。
    """
    # ---- 通道提取 ----
    CHANNEL_EXTRACT = {
        "ir": "window[:, 0]",
        "ambient": "window[:, 1] if N_ch>1 else window[:, 0]",
        "g1": "mode==1 ? window[:,3] : mode==2 ? (window[:,6]+window[:,9]+window[:,12])/3 : window[:,2]",
        "g2": "mode==1 ? window[:,4] : mode==2 ? (window[:,7]+window[:,10]+window[:,13])/3 : window[:,2]",
        "g3": "mode==1 ? window[:,5] : mode==2 ? (window[:,8]+window[:,11]+window[:,14])/3 : window[:,2]",
    }
    MODE_DETECT = (
        "mode=1 if mean(var(ch3,ch4,ch5)) > var(ch0) and mean(var(ch3,ch4,ch5)) > 1e6 else mode=2"
    )

    # ---- 预处理（每通道） ----
    PREPROCESS_STEPS = [
        ("remove_burr",    "if |x[i]-x[i-1]|>6*MAD(diff) and |x[i]-x[i+1]|>6*MAD(diff): x[i]=(x[i-1]+x[i+1])/2"),
        ("remove_step",    "if |x[i]-x[i-1]|>10*MAD(diff): x[i]=x[i-1]"),
        ("medfilt",        "median_filter(x, kernel=max(3,round(0.05*fs)), odd)   // ~50ms"),
        ("moving_avg",     "convolve(x, ones(k)/k, 'same'), k=max(2,round(0.03*fs))  // ~30ms"),
        ("bandpass",       "butterworth_4th_order(x, lowcut=0.4Hz, highcut=6.0Hz, fs=25, zero_phase=filtfilt)"),
    ]
    PREPROCESS_OUTPUT = {
        "raw_clean": "medfilt + moving_avg 之后的信号",
        "bp":       "bandpass(raw_clean)  ← AC 信号",
        "dc":       "median(raw_clean)",
    }

    # ---- 复合信号 ----
    COMPOSITE = {
        "g_mean_raw": "(g1_raw + g2_raw + g3_raw) / 3.0",
        "g_mean_bp":  "(g1_bp + g2_bp + g3_bp) / 3.0",
        "g_mean_dc":  "median(g_mean_raw)",
        "acc_mag":    "sqrt(acc_x^2 + acc_y^2 + acc_z^2)",
        "acc_mag_bp": "bandpass(acc_mag - mean(acc_mag), 0.5Hz, 5.0Hz, 4th_order, fs=25)",
    }

    # ---- 通用函数 ----
    UTILS = {
        "safe_div(a,b)":  "a / (b + 1e-12)",
        "robust_mad(x)":  "median(|x - median(x)|)",
        "robust_iqr(x)":  "percentile(x,75) - percentile(x,25)",
        "safe_corr(x,y)": "x=x-mean(x); y=y-mean(y); sx=std(x); sy=std(y);\n"
                          "  if sx<1e-12 or sy<1e-12: return 0.0\n"
                          "  return mean((x/sx) * (y/sy))",
        "smooth_envelope(x)": "convolve(|bp|, ones(w)/w, 'same'), w=max(3,round(0.25*fs)), odd",
        "max_norm_xcorr(x,y,max_lag)": "correlate(x,y,'full'), keep lags in [-max_lag,max_lag],\n"
                          "  normalize by (N*sx*sy+1e-12), return max(|val|)",
        "fft_peak_features(bp,fs,fmin,fmax)": "x=bp-mean(bp); xw=x*hamming(N); nfft=next_pow2(N,max=256);\n"
                          "  spec=|rfft(xw,nfft)|; freqs=rfftfreq(nfft,1/fs);\n"
                          "  band=(freqs>=fmin & freqs<=fmax); peak_ratio=max(band)/median(band);\n"
                          "  dom_freq=band_freqs[argmax(band)]; return (peak_ratio, dom_freq)",
        "normalized_autocorr(bp)": "x=bp-mean(bp); ac=correlate(x,x,'full'); ac=ac[N-1:];\n"
                          "  return ac/ac[0] if ac[0]>1e-12 else zeros",
        "autocorr_peak_lag(bp,fs,bpm_min,bpm_max)": "ac=normalized_autocorr(bp);\n"
                          "  lag_min=max(1,int(fs*60/bpm_max)); lag_max=min(len(ac)-1,int(fs*60/bpm_min));\n"
                          "  peak=max(ac[lag_min:lag_max+1]); lag_sec=(lag_min+argmax)/fs;\n"
                          "  return (peak, lag_sec)",
    }

    # ---- 逐特征完整配方（包含中间值展开） ----
    # 每个特征的 value 是一个 dict: {"depends": [中间信号], "formula": "完整公式"}
    FEATURE_FULL = {
        # == Core stats ==
        "IR_mean":          {"depends": ["ir_raw"], "formula": "mean(ir_raw)"},
        "IR_std":           {"depends": ["ir_raw"], "formula": "std(ir_raw)"},
        "IR_p95":           {"depends": ["ir_raw"], "formula": "percentile(ir_raw, 95)"},
        "IR_diff_std":      {"depends": ["ir_raw"], "formula": "std(diff(ir_raw))"},
        "IR_acdc":          {"depends": ["ir_bp", "ir_dc"], "formula": "safe_div(sqrt(mean(ir_bp^2)), |ir_dc|)"},

        "G_mean_mean":      {"depends": ["g_mean_raw"], "formula": "mean(g_mean_raw)"},
        "G_mean_std":       {"depends": ["g_mean_raw"], "formula": "std(g_mean_raw)"},
        "G_mean_diff_std":  {"depends": ["g_mean_raw"], "formula": "std(diff(g_mean_raw))"},
        "G_mean_acdc":      {"depends": ["g_mean_bp", "g_mean_dc"], "formula": "safe_div(sqrt(mean(g_mean_bp^2)), |g_mean_dc|)"},

        "log_IR_Gmean_mean":{"depends": ["ir_raw", "g_mean_raw"], "formula": "mean(log|ir_raw| - log|g_mean_raw|)"},
        "IR_over_Gmean_mean":{"depends": ["ir_raw", "g_mean_raw"], "formula": "mean(ir_raw / (g_mean_raw + 1e-12))"},
        "IR_over_Gmean_std":{"depends": ["ir_raw", "g_mean_raw"], "formula": "std(ir_raw / (g_mean_raw + 1e-12))"},
        "corr_IR_Gmean":    {"depends": ["ir_raw", "g_mean_raw"], "formula": "safe_corr(ir_raw, g_mean_raw)"},

        "Ambient_mean":     {"depends": ["amb_raw"], "formula": "mean(amb_raw)"},
        "Ambient_std":      {"depends": ["amb_raw"], "formula": "std(amb_raw)"},
        "Ambient_p95":      {"depends": ["amb_raw"], "formula": "percentile(amb_raw, 95)"},
        "corr_Ambient_IR":  {"depends": ["amb_raw", "ir_raw"], "formula": "safe_corr(amb_raw, ir_raw)"},
        "corr_Ambient_Gmean":{"depends": ["amb_raw", "g_mean_raw"], "formula": "safe_corr(amb_raw, g_mean_raw)"},

        "IR_over_Ambient_mean":{"depends": ["ir_raw", "amb_raw"], "formula": "mean(ir_raw / (amb_raw + 1e-12))"},
        "IR_over_Ambient_std": {"depends": ["ir_raw", "amb_raw"], "formula": "std(ir_raw / (amb_raw + 1e-12))"},

        # == Single-channel: GREEN ==
        "GREEN_DC_MEDIAN":  {"depends": ["g_mean_raw"], "formula": "median(g_mean_raw)"},
        "GREEN_DC_IQR":     {"depends": ["g_mean_raw"], "formula": "robust_iqr(g_mean_raw)"},
        "GREEN_AC_RMS":     {"depends": ["g_mean_bp"], "formula": "sqrt(mean(g_mean_bp^2))"},
        "GREEN_AC_MAD":     {"depends": ["g_mean_bp"], "formula": "robust_mad(g_mean_bp)"},
        "GREEN_AC_DC_RATIO":{"depends": ["g_mean_bp", "g_mean_dc"], "formula": "safe_div(sqrt(mean(g_mean_bp^2)), |g_mean_dc|)"},
        "GREEN_DERIV_MAD":  {"depends": ["g_mean_bp"], "formula": "robust_mad(diff(g_mean_bp))"},

        # == Single-channel: IRX ==
        "IRX_DC_MEDIAN":    {"depends": ["ir_raw"], "formula": "median(ir_raw)"},
        "IRX_DC_IQR":       {"depends": ["ir_raw"], "formula": "robust_iqr(ir_raw)"},
        "IRX_AC_RMS":       {"depends": ["ir_bp"], "formula": "sqrt(mean(ir_bp^2))"},
        "IRX_AC_MAD":       {"depends": ["ir_bp"], "formula": "robust_mad(ir_bp)"},
        "IRX_AC_DC_RATIO":  {"depends": ["ir_bp", "ir_dc"], "formula": "safe_div(sqrt(mean(ir_bp^2)), |ir_dc|)"},
        "IRX_DERIV_MAD":    {"depends": ["ir_bp"], "formula": "robust_mad(diff(ir_bp))"},

        # == Single-channel: AMBX ==
        "AMBX_DC_MEDIAN":   {"depends": ["amb_raw"], "formula": "median(amb_raw)"},
        "AMBX_DC_IQR":      {"depends": ["amb_raw"], "formula": "robust_iqr(amb_raw)"},
        "AMBX_AC_RMS":      {"depends": ["amb_bp"], "formula": "sqrt(mean(amb_bp^2))"},
        "AMBX_AC_MAD":      {"depends": ["amb_bp"], "formula": "robust_mad(amb_bp)"},
        "AMBX_AC_DC_RATIO": {"depends": ["amb_bp", "amb_dc"], "formula": "safe_div(sqrt(mean(amb_bp^2)), |amb_dc|)"},
        "AMBX_DERIV_MAD":   {"depends": ["amb_bp"], "formula": "robust_mad(diff(amb_bp))"},

        # == FFT / Frequency ==
        "GREEN_FFT_PEAK_MEDIAN_RATIO": {"depends": ["g_mean_bp"], "formula": "fft_peak_features(g_mean_bp, 25, 0.5, 5.0)[0]"},
        "GREEN_DOM_FREQ":     {"depends": ["g_mean_bp"], "formula": "fft_peak_features(g_mean_bp, 25, 0.5, 5.0)[1]"},
        "GREEN_AUTO_CORR_PEAK":{"depends": ["g_mean_bp"], "formula": "autocorr_peak_lag(g_mean_bp, 25, 40, 180)[0]"},
        "GREEN_AUTO_CORR_LAG_SEC":{"depends": ["g_mean_bp"], "formula": "autocorr_peak_lag(g_mean_bp, 25, 40, 180)[1]"},

        "IRX_FFT_PEAK_MEDIAN_RATIO": {"depends": ["ir_bp"], "formula": "fft_peak_features(ir_bp, 25, 0.5, 5.0)[0]"},
        "IRX_DOM_FREQ":     {"depends": ["ir_bp"], "formula": "fft_peak_features(ir_bp, 25, 0.5, 5.0)[1]"},
        "IRX_AUTO_CORR_PEAK":{"depends": ["ir_bp"], "formula": "autocorr_peak_lag(ir_bp, 25, 40, 180)[0]"},
        "IRX_AUTO_CORR_LAG_SEC":{"depends": ["ir_bp"], "formula": "autocorr_peak_lag(ir_bp, 25, 40, 180)[1]"},

        "AMBX_FFT_PEAK_MEDIAN_RATIO": {"depends": ["amb_bp"], "formula": "fft_peak_features(amb_bp, 25, 0.5, 5.0)[0]"},
        "AMBX_DOM_FREQ":    {"depends": ["amb_bp"], "formula": "fft_peak_features(amb_bp, 25, 0.5, 5.0)[1]"},
        "AMBX_AUTO_CORR_PEAK":{"depends": ["amb_bp"], "formula": "autocorr_peak_lag(amb_bp, 25, 40, 180)[0]"},
        "AMBX_AUTO_CORR_LAG_SEC":{"depends": ["amb_bp"], "formula": "autocorr_peak_lag(amb_bp, 25, 40, 180)[1]"},

        "GREEN_FFT_peak_width_Hz": {"depends": ["g_mean_bp", "fft_spec", "fft_freqs"],
                                     "formula": "spec,freqs=fft_peak_features内部结果; band=0.5-5Hz; "
                                                "above_half=band_spec>max(band_spec)*0.5; "
                                                "width=freqs[above_half][-1]-freqs[above_half][0]"},
        "GREEN_FFT_SNR":    {"depends": ["g_mean_bp", "fft_spec"],
                             "formula": "in_band=sum(band_spec^2); out_band=sum(spec^2)-in_band; "
                                        "SNR=in_band/(out_band+1e-12)"},

        # == Green spatial ==
        "G_imbalance_mean": {"depends": ["g1_raw", "g2_raw", "g3_raw"],
                             "formula": "g_stack=[g1_raw,g2_raw,g3_raw]; g_std=std(g_stack,axis=0); "
                                        "g_mean=mean(g_stack,axis=0); imb=g_std/|g_mean|; "
                                        "return mean(imb)"},
        "G_imbalance_p90":  {"depends": ["g1_raw", "g2_raw", "g3_raw"], "formula": "percentile(imb_series, 90)"},
        "G_imbalance_iqr":  {"depends": ["g1_raw", "g2_raw", "g3_raw"], "formula": "robust_iqr(imb_series)"},
        "G_rangeNorm_mean": {"depends": ["g1_raw", "g2_raw", "g3_raw"],
                             "formula": "g_range=(max(g_stack,axis=0)-min(g_stack,axis=0)) / "
                                        "(|g1|+|g2|+|g3|+1e-12); return mean(g_range)"},
        "G_rangeNorm_p90":  {"depends": ["g1_raw", "g2_raw", "g3_raw"], "formula": "percentile(g_range, 90)"},
        "G_spatial_vmag_mean":{"depends": ["g1_raw", "g2_raw", "g3_raw"],
                               "formula": "vx=g1-0.5*g2-0.5*g3; vy=sqrt(3)/2*(g2-g3); "
                                          "vmag=|v|/(|g1|+|g2|+|g3|+1e-12); return mean(vmag)"},
        "G_spatial_vmag_p90":{"depends": ["g1_raw", "g2_raw", "g3_raw"], "formula": "percentile(vmag, 90)"},
        "G_spatial_vmag_iqr":{"depends": ["g1_raw", "g2_raw", "g3_raw"], "formula": "robust_iqr(vmag)"},
        "G_spatial_vmag_std":{"depends": ["g1_raw", "g2_raw", "g3_raw"], "formula": "std(vmag)"},
        "G_ch_dc_cv":      {"depends": ["g1_raw", "g2_raw", "g3_raw"],
                            "formula": "dc=[median(g1),median(g2),median(g3)]; "
                                       "return std(dc)/|mean(dc)+1e-12|"},
        "G_ch_dc_max_min_ratio":{"depends": ["g1_raw", "g2_raw", "g3_raw"],
                                 "formula": "dc=[median(g1),median(g2),median(g3)]; "
                                            "return max(|dc|)/min(|dc|+1e-12)"},

        # == Green 3ch consistency ==
        "G_bp_corr_mean":  {"depends": ["g1_bp", "g2_bp", "g3_bp"],
                            "formula": "c12=safe_corr(g1_bp,g2_bp); c23=safe_corr(g2_bp,g3_bp); "
                                       "c31=safe_corr(g3_bp,g1_bp); return mean([c12,c23,c31])"},
        "G_bp_corr_min":   {"depends": ["g1_bp", "g2_bp", "g3_bp"], "formula": "min(c12,c23,c31)"},
        "G_bp_corr_std":   {"depends": ["g1_bp", "g2_bp", "g3_bp"], "formula": "std(c12,c23,c31)"},
        "G_bp_lag_std":    {"depends": ["g1_bp", "g2_bp", "g3_bp"],
                            "formula": "c12=correlate(g1-mean(g1),g2-mean(g2),'same'); "
                                       "lag12=argmax(|c12|)-N/2; "
                                       "c23=correlate(g2-mean(g2),g3-mean(g3),'same'); "
                                       "lag23=argmax(|c23|)-N/2; return std([lag12,lag23])"},

        # == Cross-channel ==
        "GREEN_IR_RAW_CORR":  {"depends": ["g_mean_raw", "ir_raw"], "formula": "safe_corr(g_mean_raw, ir_raw)"},
        "GREEN_IR_BP_CORR":   {"depends": ["g_mean_bp", "ir_bp"], "formula": "safe_corr(g_mean_bp, ir_bp)"},
        "GREEN_IR_ENV_CORR":  {"depends": ["g_mean_bp", "ir_bp"], "formula": "safe_corr(smooth_envelope(g_mean_bp,25), smooth_envelope(ir_bp,25))"},
        "GREEN_IR_MAX_XCORR": {"depends": ["g_mean_bp", "ir_bp"], "formula": "max_norm_xcorr(g_mean_bp, ir_bp, max_lag=int(0.3*25)=7)"},
        "GREEN_IR_DOM_FREQ_DIFF":{"depends": ["g_mean_bp", "ir_bp"], "formula": "|GREEN_DOM_FREQ - IRX_DOM_FREQ|"},
        "GREEN_IR_AC_RATIO":  {"depends": ["g_mean_bp", "ir_bp"], "formula": "safe_div(sqrt(mean(g_mean_bp^2)), sqrt(mean(ir_bp^2)))"},
        "GREEN_IR_DC_RATIO":  {"depends": ["g_mean_dc", "ir_dc"], "formula": "safe_div(|g_mean_dc|, |ir_dc|)"},
        "GREEN_IR_ACDC_RATIO_RATIO":{"depends": ["g_mean_bp", "ir_bp", "g_mean_dc", "ir_dc"],
                                      "formula": "safe_div(GREEN_AC_DC_RATIO, IRX_AC_DC_RATIO)"},

        "GREEN_AMB_BP_CORR":  {"depends": ["g_mean_bp", "amb_bp"], "formula": "safe_corr(g_mean_bp, amb_bp)"},
        "IR_AMB_BP_CORR":     {"depends": ["ir_bp", "amb_bp"], "formula": "safe_corr(ir_bp, amb_bp)"},
        "GREEN_AMB_ENV_CORR": {"depends": ["g_mean_bp", "amb_bp"], "formula": "safe_corr(smooth_envelope(g_mean_bp,25), smooth_envelope(amb_bp,25))"},
        "IR_AMB_ENV_CORR":    {"depends": ["ir_bp", "amb_bp"], "formula": "safe_corr(smooth_envelope(ir_bp,25), smooth_envelope(amb_bp,25))"},
        "GREEN_AMB_LEAK":     {"depends": ["g_mean_bp", "amb_bp"], "formula": "|GREEN_AMB_BP_CORR| * sqrt(mean(amb_bp^2)) / (sqrt(mean(g_mean_bp^2))+1e-12)"},
        "IR_AMB_LEAK":        {"depends": ["ir_bp", "amb_bp"], "formula": "|IR_AMB_BP_CORR| * sqrt(mean(amb_bp^2)) / (sqrt(mean(ir_bp^2))+1e-12)"},

        # == Spatial coupling ==
        "corr_Gmean_G_imbalance": {"depends": ["g_mean_raw", "g1_raw", "g2_raw", "g3_raw"],
                                   "formula": "safe_corr(g_mean_raw, imb_series)"},
        "corr_Gmean_vmag":       {"depends": ["g_mean_raw", "g1_raw", "g2_raw", "g3_raw"],
                                   "formula": "safe_corr(g_mean_raw, vmag_series)"},
        "corr_IR_G_imbalance":   {"depends": ["ir_raw", "g1_raw", "g2_raw", "g3_raw"],
                                   "formula": "safe_corr(ir_raw, imb_series)"},
        "corr_IR_vmag":          {"depends": ["ir_raw", "g1_raw", "g2_raw", "g3_raw"],
                                   "formula": "safe_corr(ir_raw, vmag_series)"},
        "corr_Ambient_vmag":     {"depends": ["amb_raw", "g1_raw", "g2_raw", "g3_raw"],
                                   "formula": "safe_corr(amb_raw, vmag_series)"},

        # == Hjorth ==
        "GREEN_Hjorth_Activity":   {"depends": ["g_mean_bp"], "formula": "var(g_mean_bp)"},
        "GREEN_Hjorth_Mobility":   {"depends": ["g_mean_bp"], "formula": "sqrt(var(diff(g_mean_bp))/var(g_mean_bp))"},
        "GREEN_Hjorth_Complexity": {"depends": ["g_mean_bp"], "formula": "sqrt(var(diff2(g_mean_bp))/var(diff(g_mean_bp)))"},
        "IRX_Hjorth_Activity":     {"depends": ["ir_bp"], "formula": "var(ir_bp)"},
        "IRX_Hjorth_Mobility":     {"depends": ["ir_bp"], "formula": "sqrt(var(diff(ir_bp))/var(ir_bp))"},
        "IRX_Hjorth_Complexity":   {"depends": ["ir_bp"], "formula": "sqrt(var(diff2(ir_bp))/var(diff(ir_bp)))"},

        # == Entropy (GREEN only) ==
        "GREEN_Entropy_Shannon": {"depends": ["g_mean_bp"], "formula": "hist=histogram(bp,10,density=True); -sum(hist[hist>0]*log(hist[hist>0]+1e-12))"},
        "GREEN_Entropy_ApEn":    {"depends": ["g_mean_bp"], "formula": "ApproximateEntropy(bp,m=2,r=0.2*std(bp)): "
                                            "patterns_i=[bp[i:i+m] for i]; D_ij=max|patterns_i-patterns_j|; "
                                            "C_i=sum(D_ij<=r)/(N-m); phi=mean(log(C_i+eps)); "
                                            "ApEn=phi(m)-phi(m+1)"},
        "GREEN_Entropy_SampEn":  {"depends": ["g_mean_bp"], "formula": "SampleEntropy(bp,m=2,r=0.2*std(bp)): "
                                            "same as ApEn but exclude self-matches (D_ii=inf); "
                                            "B=sum(D_ij<=r); SampEn=-log(B(m+1)/B(m)+eps)"},

        # == Derivative ==
        "GREEN_Deriv_d1_mean": {"depends": ["g_mean_bp"], "formula": "mean(diff(g_mean_bp))"},
        "GREEN_Deriv_d1_std":  {"depends": ["g_mean_bp"], "formula": "std(diff(g_mean_bp))"},
        "GREEN_Deriv_d1_max":  {"depends": ["g_mean_bp"], "formula": "max(diff(g_mean_bp))"},
        "GREEN_Deriv_d1_min":  {"depends": ["g_mean_bp"], "formula": "min(diff(g_mean_bp))"},
        "GREEN_Deriv_d1_zcr":  {"depends": ["g_mean_bp"], "formula": "sum(|diff(sign(diff(bp)))|)/(2*len(diff(bp)))"},
        "IRX_Deriv_d1_mean":   {"depends": ["ir_bp"], "formula": "mean(diff(ir_bp))"},
        "IRX_Deriv_d1_std":    {"depends": ["ir_bp"], "formula": "std(diff(ir_bp))"},
        "IRX_Deriv_d1_max":    {"depends": ["ir_bp"], "formula": "max(diff(ir_bp))"},
        "IRX_Deriv_d1_min":    {"depends": ["ir_bp"], "formula": "min(diff(ir_bp))"},
        "IRX_Deriv_d1_zcr":    {"depends": ["ir_bp"], "formula": "sum(|diff(sign(diff(bp)))|)/(2*len(diff(bp)))"},
        "AMBX_Deriv_d1_mean": {"depends": ["amb_bp"], "formula": "mean(diff(amb_bp))"},
        "AMBX_Deriv_d1_std":  {"depends": ["amb_bp"], "formula": "std(diff(amb_bp))"},
        "AMBX_Deriv_d1_max":  {"depends": ["amb_bp"], "formula": "max(diff(amb_bp))"},
        "AMBX_Deriv_d1_min":  {"depends": ["amb_bp"], "formula": "min(diff(amb_bp))"},
        "AMBX_Deriv_d1_zcr":  {"depends": ["amb_bp"], "formula": "sum(|diff(sign(diff(bp)))|)/(2*len(diff(bp)))"},

        # == Hjorth (AMBX) ==
        "AMBX_Hjorth_Activity":   {"depends": ["amb_bp"], "formula": "var(amb_bp)"},
        "AMBX_Hjorth_Mobility":   {"depends": ["amb_bp"], "formula": "sqrt(var(diff(amb_bp))/var(amb_bp))"},
        "AMBX_Hjorth_Complexity": {"depends": ["amb_bp"], "formula": "sqrt(var(diff2(amb_bp))/var(diff(amb_bp)))"},

        # == Entropy (AMBX) ==
        "AMBX_Entropy_Shannon": {"depends": ["amb_bp"], "formula": "Shannon entropy on 10-bin histogram of amb_bp"},

        # == Temporal (AMBX) ==
        "AMBX_Temporal_valley_ratio": {"depends": ["amb_bp"], "formula": "len(valleys)/len(amb_bp)"},

        # == Temporal ==
        "GREEN_Temporal_slope_mean":      {"depends": ["g_mean_bp"], "formula": "linear_regression(bp~arange(N)): slope=cov(t,bp)/var(t)"},
        "GREEN_Temporal_slope_std":       {"depends": ["g_mean_bp"], "formula": "std(residuals after detrend)"},
        "GREEN_Temporal_peak_prominence": {"depends": ["g_mean_bp"], "formula": "find_peaks(bp,prominence>0); return mean(prominences)"},
        "GREEN_Temporal_peak_ratio":      {"depends": ["g_mean_bp"], "formula": "len(peaks)/len(bp)"},
        "GREEN_Temporal_valley_ratio":    {"depends": ["g_mean_bp"], "formula": "len(valleys)/len(bp)"},
        "IRX_Temporal_slope_mean":      {"depends": ["ir_bp"], "formula": "linear_regression(bp~arange(N)): slope=cov(t,bp)/var(t)"},
        "IRX_Temporal_slope_std":       {"depends": ["ir_bp"], "formula": "std(residuals after detrend)"},
        "IRX_Temporal_peak_prominence": {"depends": ["ir_bp"], "formula": "find_peaks(bp,prominence>0); return mean(prominences)"},
        "IRX_Temporal_peak_ratio":      {"depends": ["ir_bp"], "formula": "len(peaks)/len(bp)"},
        "AMBX_Temporal_slope_mean":      {"depends": ["amb_bp"], "formula": "linear_regression(bp~arange(N)): slope=cov(t,bp)/var(t)"},
        "AMBX_Temporal_slope_std":       {"depends": ["amb_bp"], "formula": "std(residuals after detrend)"},
        "AMBX_Temporal_peak_prominence": {"depends": ["amb_bp"], "formula": "find_peaks(bp,prominence>0); return mean(prominences)"},
        "AMBX_Temporal_peak_ratio":      {"depends": ["amb_bp"], "formula": "len(peaks)/len(bp)"},

        # == Waveform shape ==
        "GREEN_bp_skewness": {"depends": ["g_mean_bp"], "formula": "mean((bp-mean(bp))^3) / std(bp)^3"},
        "IRX_bp_skewness":   {"depends": ["ir_bp"], "formula": "mean((bp-mean(bp))^3) / std(bp)^3"},
        "GREEN_bp_kurtosis": {"depends": ["g_mean_bp"], "formula": "mean((bp-mean(bp))^4) / std(bp)^4"},
        "IRX_bp_kurtosis":   {"depends": ["ir_bp"], "formula": "mean((bp-mean(bp))^4) / std(bp)^4"},

        # == ACC ==
        "ACC_MAG_MEAN":     {"depends": ["acc_mag"], "formula": "mean(acc_mag)"},
        "ACC_MAG_STD":      {"depends": ["acc_mag"], "formula": "std(acc_mag)"},
        "ACC_MAG_MAD":      {"depends": ["acc_mag"], "formula": "robust_mad(acc_mag)"},
        "ACC_AXIS_STD_SUM": {"depends": ["acc"], "formula": "sum(std(acc, axis=0))"},
        "ACC_GRAVITY_DOM_RATIO":{"depends": ["acc"], "formula": "max(|mean(acc_x)|,|mean(acc_y)|,|mean(acc_z)|)/(sum|mean|+1e-8)"},
        "ACC_BP_RMS":       {"depends": ["acc_mag_bp"], "formula": "sqrt(mean(acc_mag_bp^2))"},
        "ACC_DIFF_MAD":     {"depends": ["acc_mag"], "formula": "robust_mad(diff(acc_mag))"},
        "ACC_STILL_SCORE":  {"depends": ["acc_mag"], "formula": "1.0/(1.0+50.0*std(acc_mag)/(|mean(acc_mag)|+1e-6))"},
        "ACC_MAG_P50":      {"depends": ["acc_mag"], "formula": "percentile(acc_mag, 50)"},
        "ACC_MAG_P90":      {"depends": ["acc_mag"], "formula": "percentile(acc_mag, 90)"},
        "ACC_GREEN_BP_CORR":{"depends": ["acc_mag_bp", "g_mean_bp"], "formula": "|safe_corr(acc_mag_bp, g_mean_bp)|"},
        "ACC_IR_BP_CORR":   {"depends": ["acc_mag_bp", "ir_bp"], "formula": "|safe_corr(acc_mag_bp, ir_bp)|"},
    }

    # ---- 为每个入选特征组装完整配方 ----
    recipe = {}
    for f in selected_features:
        info = FEATURE_FULL.get(f)
        if info is None:
            recipe[f] = {"formula": "[未匹配]"}
            continue

        # 展开依赖的中间信号
        deps_expanded = {}
        for dep in info["depends"]:
            if dep in PREPROCESS_OUTPUT:
                deps_expanded[dep] = f"preprocess_signal({dep.split('_')[0]}) → {PREPROCESS_OUTPUT.get(dep, dep)}"
            elif dep in COMPOSITE:
                deps_expanded[dep] = COMPOSITE[dep]
            elif dep.endswith("_raw") or dep.endswith("_bp") or dep.endswith("_dc"):
                ch = dep.replace("_raw", "").replace("_bp", "").replace("_dc", "")
                signal_type = "raw" if "_raw" in dep else ("bp" if "_bp" in dep else "dc")
                deps_expanded[dep] = f"preprocess_signal({ch}) → {signal_type}"
            elif dep in CHANNEL_EXTRACT:
                deps_expanded[dep] = CHANNEL_EXTRACT[dep]
            elif dep in ("acc_mag", "acc_mag_bp"):
                deps_expanded[dep] = COMPOSITE.get(dep, dep)
            else:
                deps_expanded[dep] = "(computed on the fly)"

        recipe[f] = {
            "formula": info["formula"],
            "intermediate_signals": deps_expanded,
        }

    for f in selected_features:
        if str(recipe.get(f, {}).get("formula", "")).startswith("[") and f in EXTRA_FEATURE_FORMULAS:
            recipe[f] = EXTRA_FEATURE_FORMULAS[f]

    return recipe, CHANNEL_EXTRACT, MODE_DETECT, PREPROCESS_STEPS, PREPROCESS_OUTPUT, COMPOSITE, UTILS


def export_deploy_cookbook(artifact_dir):
    """
    导出完整部署配方: 每个入选特征从 25Hz 原始窗口 → 预处理 → 中间信号 → 特征值的完整链。
    嵌入式工程师可直接翻译为 C/Rust，无需查任何其他文件。
    """
    import os as _os
    import json as _json
    import joblib as _joblib

    bundle_path = _os.path.join(artifact_dir, "model_bundle.pkl")
    th_path = _os.path.join(artifact_dir, "stage1_threshold.json")

    if not _os.path.exists(bundle_path):
        print("[WARN] model_bundle.pkl not found, skip deploy cookbook")
        return

    bundle = _joblib.load(bundle_path)
    selected = bundle["feature_names"]
    fill_values = bundle["fill_values"]
    threshold = float(bundle["threshold"])
    model = bundle["model"]
    raw = bundle.get("raw_model", model)
    booster = raw.get_booster()
    n_estimators = raw.n_estimators

    # 生成完整特征配方
    recipe, ch_extract, mode_detect, preproc_steps, preproc_out, composite, utils = _build_full_feature_recipe(selected)

    # 组装输出
    cookbook = {
        "_title": "手表佩戴活体检测 — 部署配方 (Deployment Cookbook)",
        "_for": "嵌入式/工程化部署工程师。本文件自包含，无需查任何其他文件。",
        "_input": "25Hz PPG窗口 (125 samples x N_ch) + ACC窗口 (125 samples x 3)",

        # ---- Section A: 公共计算（所有特征共用） ----
        "A_channel_extraction": {
            "_note": "从 PPG 窗口提取 ir/ambient/g1/g2/g3",
            "mode_detection": mode_detect,
            "channels": ch_extract,
        },
        "A_preprocessing": {
            "_note": "对 ir/ambient/g1/g2/g3 各执行以下管线，产出 *_raw_clean, *_bp, *_dc",
            "pipeline": [{"step": i+1, "name": name, "formula": formula} for i, (name, formula) in enumerate(preproc_steps)],
            "outputs": preproc_out,
        },
        "A_composite_signals": {
            "_note": "从各通道预处理结果合成复合信号",
            "signals": composite,
        },
        "A_utility_functions": {
            "_note": "以下所有公式中引用的工具函数",
            "definitions": utils,
        },

        # ---- Section B: 入选特征完整配方 ----
        "B_selected_features": {
            "_note": f"共 {len(selected)} 个特征，按此顺序组成 XGBoost 输入向量 feature_vec[0..{len(selected)-1}]",
            "feature_order": selected,
            "recipes": {f: recipe.get(f, {"formula": "[未匹配]"}) for f in selected},
        },

        # ---- Section C: XGBoost 推理 ----
        "C_xgboost_inference": {
            "_note": "拿到 feature_vec 后的推理步骤",
            "fill_values": fill_values,
            "fill_rule": "feature_vec[i] 为 NaN/inf 时用 fill_values[feature_name] 替换",
            "model_threshold": threshold,
            "n_estimators": n_estimators,
            "model_json": _json.loads(booster.save_config()),
            "inference": [
                "1. feature_vec = [compute_feature(f) for f in feature_order]",
                "2. for i, v in enumerate(feature_vec): if isnan(v) or isinf(v): feature_vec[i] = fill_values[feature_order[i]]",
                "3. proba = xgboost_predict(model, feature_vec)  // → float in [0,1]",
                "4. window_pred = 1 if proba >= threshold else 0",
            ],
        },

        # ---- Section D: Stage1 & Stage3 ----
        "D_stage1_gate": {
            "_note": "XGBoost 之前先跑 IR 粗筛",
            "ir_5hz": "resample(ir_raw_100Hz -> 5Hz)",
            "primitive_window": "1s stride=1s (5 points @5Hz)",
            "decision_window": "3 consecutive primitive decisions",
            "dc_formula": "min(neighbor_mean) where neighbor_mean[i]=(x[i]+x[i+1])/2",
            "ac_formula": "median(|diff(x)|)",
            "rule": "dc > dc_thresh AND ac/|dc| < acdc_thresh",
            "streaming_gate_rule": "open Stage2 after 3 consecutive pass primitives; close after 3 consecutive fail primitives",
            "thresholds": {},
        },
        "D_stage3_postprocess": {
            "_note": "对 XGBoost 输出的逐窗概率做时序平滑",
            "algorithm": "EMA + hysteresis + cooldown",
            "params": {"alpha": 0.4, "T_on": 0.75, "T_off": 0.35, "K_on": 5, "K_off": 3, "cooldown_sec": 2.0},
            "pseudocode": [
                "score[t] = alpha * quality[t] * proba[t] + (1-alpha*quality[t]) * score[t-1]",
                "IF state==0 and count(score>T_on) >= K_on and cooldown_expired: state=1, reset counter",
                "IF state==1 and count(score<T_off) >= K_off and cooldown_expired: state=0, reset counter",
                "quality[t] from Ambient_std / G_mean_mean / IR_mean thresholds (from bundle)",
            ],
            "quality_thresholds": bundle.get("quality_thresholds", {}),
        },
    }

    # Stage1 阈值
    if _os.path.exists(th_path):
        with open(th_path, "r", encoding="utf-8") as f:
            th_data = _json.load(f)
        dth = th_data.get("deploy_stage1_threshold", {})
        cookbook["D_stage1_gate"]["thresholds"] = {
            "dc_threshold": float(dth.get("dc_threshold", 0)),
            "ac_dc_threshold": float(dth.get("ac_dc_threshold", 0)),
        }

    out_path = _os.path.join(artifact_dir, "deploy_cookbook.json")
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(cookbook, f, indent=2, ensure_ascii=False)

    xgb_path = _os.path.join(artifact_dir, "deploy_xgboost.json")
    with open(xgb_path, "w", encoding="utf-8") as f:
        _json.dump({
            "feature_names": selected,
            "feature_order": selected,
            "n_estimators": n_estimators,
            "threshold": threshold,
            "model": _json.loads(booster.save_config()),
        }, f, indent=2, ensure_ascii=False)

    print(f"[OK] deploy_cookbook.json -> {out_path}")
    print(f"[OK] deploy_xgboost.json  -> {xgb_path}")

def generate_eval_csv(artifact_dir, split="test", method="state_machine"):
    """生成逐样本 CSV: info, target, total_windows, correct_windows。"""
    import os as _os
    details = _load_eval_details(artifact_dir, split, method)
    if not details:
        print("[WARN] 评估结果为空，跳过 CSV")
        return

    csv_path = _os.path.join(artifact_dir, "per_sample_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("sample_name,target,total_windows,correct_windows\n")
        for d in details:
            wpreds = d.get("window_preds", [])
            t = d.get("target", 0)
            n_win = len(wpreds)
            n_correct = sum(1 for p in wpreds if p == t) if n_win > 0 else 0
            f.write(f"{d['sample_name']},{t},{n_win},{n_correct}\n")
    print(f"[OK] 逐样本 CSV: {csv_path}")


def plot_error_samples(artifact_dir, split="test", method="state_machine",
                       window_sec=3, stride_sec=1):
    """
    准确率非 100% 的样本画图。4 子图:
      1. 原始 target (0/1 横线)
      2. 窗口级 XGBoost 概率 (probs)
      3. 窗口级状态机 EMA score
      4. 后处理标签值 (states 0/1)
    """
    import os as _os
    import numpy as _np

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
    except ImportError:
        print("[WARN] matplotlib not installed, skip plotting")
        return

    details = _load_eval_details(artifact_dir, split, method)
    if not details:
        print("[WARN] no eval details found")
        return

    errors = [d for d in details if d["pred"] != d["target"]]
    if not errors:
        print("[OK] all samples correct, no plots needed")
        return

    out_dir = _os.path.join(artifact_dir, "error_plots")
    _os.makedirs(out_dir, exist_ok=True)
    print(f"\n  Wrong predictions: {len(errors)}/{len(details)}")

    for d in errors:
        target = d["target"]
        pred = d["pred"]
        probs = d.get("window_probs", [])
        scores = d.get("window_scores", [])
        states = d.get("window_states", [])
        n_win = d.get("n_windows", len(probs))
        t = _np.arange(n_win) * stride_sec + window_sec / 2.0 if n_win > 0 else _np.array([])

        fig, axes = _plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        # 1. Ground truth
        ax = axes[0]
        ax.set_ylabel("Target", fontsize=11)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{d['sample_name']}  target={target}  pred={pred}  "
                     f"s1_pass={d.get('stage1_pass',False)}  fb={d.get('fallback',False)}",
                     fontsize=10)
        if n_win > 0:
            c = "green" if target == 1 else "red"
            ax.axhline(y=target, color=c, linewidth=2, linestyle="--", alpha=0.7)
            ax.fill_between([t[0], t[-1]], target - 0.05, target + 0.05,
                            alpha=0.15, color=c)

        # 2. XGBoost probs (continuous)
        ax = axes[1]
        ax.set_ylabel("Model Probs", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if n_win > 0:
            ax.step(t, probs, where="mid", linewidth=1.5, color="steelblue")
            ax.fill_between(t, 0, _np.array(probs), alpha=0.12, color="steelblue", step="mid")
            ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

        # 3. State machine EMA scores (continuous)
        ax = axes[2]
        ax.set_ylabel("State Machine Score", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if n_win > 0 and len(scores) > 0:
            ax.plot(t, scores, linewidth=2, color="darkorange", marker=".", markersize=3)
            ax.fill_between(t, 0, _np.array(scores), alpha=0.10, color="darkorange")

        # 4. State machine labels (0/1)
        ax = axes[3]
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("State Labels", fontsize=11)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.grid(True, alpha=0.3)
        if n_win > 0 and len(states) > 0:
            ax.step(t, states, where="mid", linewidth=2, color="crimson")
            is_wrong = pred != target
            ax.text(t[len(t) // 2] if len(t) > 0 else 0, 0.5,
                    "WRONG" if is_wrong else "OK",
                    fontsize=28, color="red" if is_wrong else "green",
                    alpha=0.25, weight="bold", ha="center", va="center")

        _plt.tight_layout()
        safe_name = d["sample_name"].replace("/", "_").replace("\\", "_")
        fig.savefig(_os.path.join(out_dir, f"{safe_name}.png"), dpi=120, bbox_inches="tight")
        _plt.close(fig)

    print(f"[OK] {len(errors)} plots -> {out_dir}/")


def main():
    p = argparse.ArgumentParser(
        description="手表佩戴活体检测 — 全流程主控脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --dataset_dir dataset --artifact_dir artifacts
  %(prog)s --stop_after s04
  %(prog)s --skip s02,s03
  %(prog)s --artifact_dir artifacts --export_deploy  (仅评估 + 导出)
""")

    # ── 通用参数 ──
    p.add_argument("--artifact_dir", default="artifacts", help="产物目录")
    p.add_argument("--dataset_dir", default="dataset", help="原始 H5 数据目录")
    p.add_argument("--n_workers", type=int,
                   default=max(1, min(4, (os.cpu_count() or 4) // 2)),
                   help="并行 worker 数")
    p.add_argument("--dry_run", action="store_true", help="只打印命令不执行")

    # ── 步骤控制 ──
    p.add_argument("--skip", default="", help="跳过的步骤，逗号分隔 (如 s03,s04)")
    p.add_argument("--stop_after", default="s06_cb",
                   help="运行到此步骤后停止 (默认跑完全部)")

    # ── s03 参数 ──
    p.add_argument("--window_sec", type=int, default=3, help="特征提取窗口秒数")
    p.add_argument("--stride_sec", type=int, default=1, help="训练/评估特征提取步长秒数")
    p.add_argument("--skip_initial_windows", type=int, default=3,
                   help="drop this many leading Stage2 windows per sample")
    p.add_argument("--use_stage2_ir", action=argparse.BooleanOptionalAction, default=False,
                   help="whether Stage2 feature extraction uses IR channel values")

    # ── s04 参数 ──
    p.add_argument("--max_features", type=int, default=15, help="最终选择的特征数")
    p.add_argument("--min_fold_auc", type=float, default=0.55, help="稳定性选择最低有效 AUC")
    p.add_argument("--skip_vif", action=argparse.BooleanOptionalAction, default=False,
                   help="跳过 s04 VIF 清洗步骤；特征很多或 VIF 卡住时建议开启")
    p.add_argument("--deployment_score_weight", type=float, default=0.25,
                   help="s04 部署导向重排权重。0=保持原始重要性排序")

    # ── s05 参数 ──
    p.add_argument("--fp_cost_weight", type=float, default=0.25,
                   help="s04 sample/state-machine FP cost proxy reranking weight")
    p.add_argument("--fp_proxy_recall_floor", type=float, default=0.95,
                   help="s04 FP proxy positive-window recall floor")
    p.add_argument("--fp_proxy_state_k_on", type=int, default=3,
                   help="s04 FP proxy consecutive windows for state trigger")
    p.add_argument("--threshold_objective", default="fbeta",
                   choices=["f1", "precision", "recall", "fbeta", "precision_constrained"])
    p.add_argument("--threshold_beta", type=float, default=0.5,
                   help="F-beta 参数 (<1 偏precision)")

    # ── s06 参数 ──
    # s05 model-search params
    p.add_argument("--model_search", action=argparse.BooleanOptionalAction, default=True,
                   help="enable s05 XGBoost param search under a node budget; use --no-model_search to disable")
    p.add_argument("--max_model_nodes", type=int, default=400,
                   help="s05 max total XGBoost nodes for --model_search")
    p.add_argument("--model_search_fp_cost", type=float, default=2.0,
                   help="s05 FP penalty in model-search score")
    p.add_argument("--model_search_size_cost", type=float, default=0.1,
                   help="s05 model-size penalty in model-search score")
    p.add_argument("--model_search_valid_fraction", type=float, default=0.5,
                   help="fraction of valid calibration pool used for model search")
    p.add_argument("--model_search_n_estimators", default="20,30,40",
                   help="comma-separated s05 n_estimators candidates")
    p.add_argument("--model_search_max_depth", default="2,3",
                   help="comma-separated s05 max_depth candidates")
    p.add_argument("--model_search_learning_rate", default="0.03,0.05,0.08",
                   help="comma-separated s05 learning_rate candidates")
    p.add_argument("--model_search_min_child_weight", default="20,30,50",
                   help="comma-separated s05 min_child_weight candidates")
    p.add_argument("--model_search_reg_lambda", default="10,20",
                   help="comma-separated s05 reg_lambda candidates")
    p.add_argument("--model_search_reg_alpha", default="1,2",
                   help="comma-separated s05 reg_alpha candidates")
    p.add_argument("--model_search_subsample", default="0.7,0.8,0.9",
                   help="comma-separated s05 subsample candidates")
    p.add_argument("--model_search_colsample_bytree", default="0.7,0.8,0.9",
                   help="comma-separated s05 colsample_bytree candidates")
    # s06 eval / calibration params
    p.add_argument("--calibration_method", default="isotonic", choices=["none", "isotonic"])
    p.add_argument("--threshold_valid_fraction", type=float, default=0.5)
    p.add_argument("--calibration_random_state", type=int, default=42)
    p.add_argument("--split", default="test", choices=["train", "valid", "test"],
                   help="s06 评估用的数据 split")
    p.add_argument("--export_deploy", action=argparse.BooleanOptionalAction, default=True,
                   help="s06 导出部署产物 (--no-export_deploy 跳过)")
    p.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=True,
                   help="s06 运行状态机参数优化 (--no-optimize 跳过)")
    p.add_argument("--plot_errors", action=argparse.BooleanOptionalAction, default=True,
                   help="s06 评估后画出错误样本图 (--no-plot_errors 跳过)")
    p.add_argument("--export_window_cache", action=argparse.BooleanOptionalAction, default=False,
                   help="export window-level NPZ cache for s07 postprocess optimization")
    p.add_argument("--optimize_postprocess", action=argparse.BooleanOptionalAction, default=False,
                   help="run s07 FP-sensitive postprocess optimization on cached windows")
    p.add_argument("--postprocess_split", default="valid", choices=["train", "valid", "test"],
                   help="split used by s07 postprocess optimization")
    p.add_argument("--postprocess_fp_cost", type=float, default=4.0,
                   help="s07 sample false-positive cost")
    p.add_argument("--max_sample_fp_rate", type=float, default=0.02,
                   help="s07 maximum FP / true-negative-sample rate")
    p.add_argument("--max_false_worn_event_rate", type=float, default=0.02,
                   help="s07 maximum negative-sample false-worn event rate")
    p.add_argument("--max_first_worn_output_p95_sec", type=float, default=6.0,
                   help="s07 maximum P95 first worn output latency for positive samples")
    p.add_argument("--commercial_compare", action=argparse.BooleanOptionalAction, default=False,
                   help="run optional s09 commercial-vs-project comparison")
    p.add_argument("--commercial_split", default="test", choices=["train", "valid", "test"],
                   help="split used by s09 commercial comparison")
    p.add_argument("--commercial_fp_cost", type=float, default=4.0,
                   help="s09 commercial comparison FP cost")
    p.add_argument("--keep_window_probs", action="store_true",
                   help="keep per-window probabilities in s09 commercial comparison details")
    p.add_argument("--export_deploy_cookbook", action=argparse.BooleanOptionalAction, default=True,
                   help="导出部署配方给嵌入式同事 (--no-export_deploy_cookbook 跳过)")

    args = p.parse_args()
    thread_env = configure_thread_env()
    print("[parallel] thread caps inherited by child steps: " +
          ", ".join(f"{k}={v}" for k, v in thread_env.items()))

    # 步骤定义: (key, display_name, 是否默认启用)
    all_steps = [
        ("s01",       "数据扫描 & 切分"),
        ("s02",       "Stage1 固定阈值配置"),
        ("s03",       "特征池提取"),
        ("s04",       "稳定性特征筛选"),
        ("s04_search","候选特征子集搜索"),
        ("s05",       "XGBoost 模型训练"),
        ("s06_opt",   "状态机参数优化"),
        ("s06_cache", "导出 valid 逐窗 NPZ 缓存"),
        ("s06_replay_cache", "导出 replay 逐窗 NPZ 缓存"),
        ("s07_post",  "FP 敏感后处理搜参"),
        ("s06_eval",  "端到端评估"),
        ("s06_feat",  "导出特征提取脚本"),
        ("s06_plot",  "画错误样本图"),
        ("s06_cb",    "导出部署配方"),
        ("s09_cmp",   "商用方案对比"),
    ]

    skip_set = {s.strip() for s in args.skip.split(",") if s.strip()}
    stop_after = args.stop_after
    stage2_ir_flag = "--use_stage2_ir" if args.use_stage2_ir else "--no-use_stage2_ir"
    s04_skip_vif_flag = "--skip_vif" if args.skip_vif else ""
    model_search_flag = "--model_search" if args.model_search else "--no-model_search"
    cache_export_flag = " --export_window_cache" if args.export_window_cache else ""
    if not args.dry_run and "s01" not in skip_set and not dataset_has_h5_files(args.dataset_dir):
        print(f"[ERROR] no .h5 files found in dataset_dir={args.dataset_dir!r}")
        print("        Pass --dataset_dir with the real H5 directory, or use --skip s01 when reusing artifacts.")
        sys.exit(2)

    # ── 构建命令 ──
    commands = {}

    # s01
    if "s01" not in skip_set:
        commands["s01"] = (
            f'"{PYTHON}" "{_script_path("s01_data_split")}" '
            f'--dataset_dir "{args.dataset_dir}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--n_workers {args.n_workers}'
        )

    # s02
    if "s02" not in skip_set:
        commands["s02"] = (
            f'"{PYTHON}" "{_script_path("s02_ir_dc_threshold")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--n_workers {args.n_workers}'
        )

    # s03
    if "s03" not in skip_set:
        commands["s03"] = (
            f'"{PYTHON}" "{_script_path("s03_extract_feature_pool")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--window_sec {args.window_sec} '
            f'--stride_sec {args.stride_sec} '
            f'--skip_initial_windows {args.skip_initial_windows} '
            f'{stage2_ir_flag} '
            f'--n_workers {args.n_workers}'
        )

    # s04
    if "s04" not in skip_set:
        commands["s04"] = (
            f'"{PYTHON}" "{_script_path("s04_feature_selection")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--max_features {args.max_features} '
            f'--min_fold_auc {args.min_fold_auc} '
            f'--deployment_score_weight {args.deployment_score_weight} '
            f'--fp_cost_weight {args.fp_cost_weight} '
            f'--fp_proxy_recall_floor {args.fp_proxy_recall_floor} '
            f'--fp_proxy_state_k_on {args.fp_proxy_state_k_on} '
            f'{s04_skip_vif_flag} '
            f'--n_workers {args.n_workers}'
        )

    # s04_search: 候选特征子集搜索（可选步骤，覆写 selected_features.json）
    if "s04_search" not in skip_set:
        commands["s04_search"] = (
            f'"{PYTHON}" "{_script_path("s04_feature_selection")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--max_features {args.max_features} '
            f'--min_fold_auc {args.min_fold_auc} '
            f'--deployment_score_weight {args.deployment_score_weight} '
            f'--fp_cost_weight {args.fp_cost_weight} '
            f'--fp_proxy_recall_floor {args.fp_proxy_recall_floor} '
            f'--fp_proxy_state_k_on {args.fp_proxy_state_k_on} '
            f'{s04_skip_vif_flag} '
            f'--run_subset_search '
            f'--subset_search_max_features {args.max_features} '
            f'--n_workers {args.n_workers}'
        )

    # s05
    if "s05" not in skip_set:
        commands["s05"] = (
            f'"{PYTHON}" "{_script_path("s05_train_final_model")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--threshold_objective {args.threshold_objective} '
            f'--threshold_beta {args.threshold_beta} '
            f'--calibration_method {args.calibration_method} '
            f'--threshold_valid_fraction {args.threshold_valid_fraction} '
            f'--calibration_random_state {args.calibration_random_state} '
            f'--window_sec {args.window_sec} '
            f'--step_sec {args.stride_sec} '
            f'{stage2_ir_flag} '
            f'{model_search_flag} '
            f'--max_model_nodes {args.max_model_nodes} '
            f'--model_search_fp_cost {args.model_search_fp_cost} '
            f'--model_search_size_cost {args.model_search_size_cost} '
            f'--model_search_valid_fraction {args.model_search_valid_fraction} '
            f'--model_search_n_estimators "{args.model_search_n_estimators}" '
            f'--model_search_max_depth "{args.model_search_max_depth}" '
            f'--model_search_learning_rate "{args.model_search_learning_rate}" '
            f'--model_search_min_child_weight "{args.model_search_min_child_weight}" '
            f'--model_search_reg_lambda "{args.model_search_reg_lambda}" '
            f'--model_search_reg_alpha "{args.model_search_reg_alpha}" '
            f'--model_search_subsample "{args.model_search_subsample}" '
            f'--model_search_colsample_bytree "{args.model_search_colsample_bytree}"'
        )

    # s06_opt
    if "s06_opt" not in skip_set and args.optimize:
        commands["s06_opt"] = (
            f'"{PYTHON}" "{_script_path("s06_deploy_eval")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--split valid '
            f'--n_workers {args.n_workers} '
            f'--optimize '
            f'--window_sec {args.window_sec} '
            f'--stride_sec {args.stride_sec} '
            f'--skip_initial_windows {args.skip_initial_windows} '
            f'{stage2_ir_flag}'
        )
    elif "s06_opt" not in skip_set:
        print("(s06_opt: --no-optimize 跳过)")

    # s06_cache: export the postprocess-search split independently from legacy s06 optimization.
    if "s06_cache" not in skip_set and args.export_window_cache:
        commands["s06_cache"] = (
            f'"{PYTHON}" "{_script_path("s06_deploy_eval")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--split {args.postprocess_split} '
            f'--n_workers {args.n_workers} '
            f'--window_sec {args.window_sec} '
            f'--stride_sec {args.stride_sec} '
            f'--skip_initial_windows {args.skip_initial_windows} '
            f'{stage2_ir_flag} '
            f'--window_output_root window_outputs'
            f'{cache_export_flag}'
        )
    elif "s06_cache" not in skip_set:
        print("(s06_cache: --no-export_window_cache skipped)")

    if ("s06_replay_cache" not in skip_set and args.export_window_cache
            and args.optimize_postprocess and args.split != args.postprocess_split):
        commands["s06_replay_cache"] = (
            f'"{PYTHON}" "{_script_path("s06_deploy_eval")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--split {args.split} '
            f'--n_workers {args.n_workers} '
            f'--window_sec {args.window_sec} '
            f'--stride_sec {args.stride_sec} '
            f'--skip_initial_windows {args.skip_initial_windows} '
            f'{stage2_ir_flag} '
            f'--window_output_root window_outputs'
            f'{cache_export_flag}'
        )
    elif "s06_replay_cache" not in skip_set:
        print("(s06_replay_cache: skipped)")

    # s06_eval
    if "s06_eval" not in skip_set:
        commands["s06_eval"] = (
            f'"{PYTHON}" "{_script_path("s06_deploy_eval")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--split {args.split} '
            f'--n_workers {args.n_workers} '
            f'--window_sec {args.window_sec} '
            f'--stride_sec {args.stride_sec} '
            f'--skip_initial_windows {args.skip_initial_windows} '
            f'{stage2_ir_flag} '
            f'--window_output_root window_outputs'
        )

    # s07_post: tune the richer FP-sensitive state machine on cached windows.
    if "s07_post" not in skip_set and args.optimize_postprocess:
        commands["s07_post"] = (
            f'"{PYTHON}" "{_script_path("s07_postprocess_optimize")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--split {args.postprocess_split} '
            f'--cache_root window_outputs '
            f'--fp_cost {args.postprocess_fp_cost} '
            f'--max_sample_fp_rate {args.max_sample_fp_rate} '
            f'--max_false_worn_event_rate {args.max_false_worn_event_rate} '
            f'--max_first_worn_output_p95_sec {args.max_first_worn_output_p95_sec} '
            f'--replay_split {args.split}'
        )
    elif "s07_post" not in skip_set:
        print("(s07_post: --no-optimize_postprocess 跳过)")

    # s06_xpt
    if "s06_xpt" not in skip_set and args.export_deploy:
        commands["s06_xpt"] = (
            f'"{PYTHON}" "{_script_path("s06_deploy_eval")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--split {args.split} '
            f'--n_workers {args.n_workers} '
            f'--window_sec {args.window_sec} '
            f'--stride_sec {args.stride_sec} '
            f'--skip_initial_windows {args.skip_initial_windows} '
            f'{stage2_ir_flag} '
            f'--export_deploy'
        )
    elif "s06_xpt" not in skip_set:
        print("(s06_xpt: --no-export_deploy 跳过)")

    # s09_cmp
    if "s09_cmp" not in skip_set and args.commercial_compare:
        keep_probs = " --keep_window_probs" if args.keep_window_probs else ""
        commands["s09_cmp"] = (
            f'"{PYTHON}" "{_script_path("s09_commercial_compare")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--split {args.commercial_split} '
            f'--fp_cost {args.commercial_fp_cost} '
            f'--method state_machine '
            f'--window_sec {args.window_sec} '
            f'--stride_sec {args.stride_sec} '
            f'--skip_initial_windows {args.skip_initial_windows} '
            f'{stage2_ir_flag}'
            f'{keep_probs}'
        )
    elif "s09_cmp" not in skip_set:
        print("(s09_cmp: --no-commercial_compare skipped)")

    # s06_plot (纯 Python 调用，非子进程)
    if "s06_plot" not in skip_set and args.plot_errors:
        commands["s06_plot"] = "__plot__"  # 特殊标记
    elif "s06_plot" not in skip_set:
        print("(s06_plot: --no-plot_errors 跳过)")

    # s06_feat (纯 Python 调用)
    if "s06_feat" not in skip_set and args.export_deploy_cookbook:
        commands["s06_feat"] = "__extractor__"
    elif "s06_feat" not in skip_set:
        print("(s06_feat: --no-export_deploy_cookbook 跳过)")

    # s06_cb (纯 Python 调用)
    if "s06_cb" not in skip_set and args.export_deploy_cookbook:
        commands["s06_cb"] = "__cookbook__"
    elif "s06_cb" not in skip_set:
        print("(s06_cb: --no-export_deploy_cookbook 跳过)")

    # ── 执行 ──
    print("=" * 70)
    print(" 手表佩戴活体检测 — 全流程")
    print("=" * 70)
    print(f"  产物目录:     {args.artifact_dir}")
    print(f"  数据目录:     {args.dataset_dir}")
    print(f"  并行 worker:  {args.n_workers}")
    print(f"  入选特征数:   {args.max_features}")
    print(f"  评估 split:   {args.split}")
    print(f"  导出部署:     {'是' if args.export_deploy else '否'}")
    print(f"  状态机优化:   {'是' if args.optimize else '否'}")
    print("=" * 70)

    total_start = time.time()
    results = {}
    completed_keys = set()
    stopped_at = None

    for key, display_name in all_steps:
        if key in skip_set:
            print(f"\n  [SKIP] {display_name} (--skip)")
            continue

        cmd = commands.get(key)
        if cmd is None:
            continue  # optional step not requested

        if args.dry_run and cmd in {"__plot__", "__extractor__", "__cookbook__"}:
            print(f"\n[RUN] {display_name}")
            print(f"  {cmd}")
            print("  (dry-run, skipped)")
            completed_keys.add(key)
            if key == stop_after:
                stopped_at = key
                print(f"\n[STOP] 已达到 --stop_after={stop_after}，停止")
                break
            continue

        if cmd == "__plot__":
            t0 = time.time()
            generate_eval_csv(args.artifact_dir, split=args.split)
            plot_error_samples(args.artifact_dir, split=args.split,
                               window_sec=args.window_sec, stride_sec=args.stride_sec)
            dt = time.time() - t0
            completed_keys.add(key)
            print(f"[OK] {display_name}  [{timedelta(seconds=int(dt))}]")
            if key == stop_after:
                stopped_at = key
                print(f"\n[STOP] 已达到 --stop_after={stop_after}，停止")
                break
            continue

        if cmd == "__extractor__":
            t0 = time.time()
            export_feature_extractor_script(args.artifact_dir)
            dt = time.time() - t0
            completed_keys.add(key)
            print(f"[OK] {display_name}  [{timedelta(seconds=int(dt))}]")
            if key == stop_after:
                stopped_at = key
                print(f"\n[STOP] 已达到 --stop_after={stop_after}，停止")
                break
            continue

        if cmd == "__cookbook__":
            t0 = time.time()
            export_deploy_cookbook(args.artifact_dir)
            dt = time.time() - t0
            completed_keys.add(key)
            print(f"[OK] {display_name}  [{timedelta(seconds=int(dt))}]")
            if key == stop_after:
                stopped_at = key
                print(f"\n[STOP] 已达到 --stop_after={stop_after}，停止")
                break
            continue

        ok = _run(display_name, cmd, dry_run=args.dry_run)
        results[key] = ok
        if ok:
            completed_keys.add(key)
        if not ok:
            print(f"\n[FAIL] 流水线中断于: {display_name}")
            sys.exit(1)

        if key == stop_after:
            stopped_at = key
            print(f"\n[STOP] 已达到 --stop_after={stop_after}，停止")
            break

    total_dt = time.time() - total_start
    print(f"\n{'=' * 70}")
    if stopped_at:
        print(f"[OK] 流水线已按 --stop_after={stopped_at} 结束  [{timedelta(seconds=int(total_dt))}]")
    else:
        print(f"[OK] 全流程完成  [{timedelta(seconds=int(total_dt))}]")
    print(f"{'=' * 70}")

    if args.export_deploy and "s06_xpt" in completed_keys and not args.dry_run:
        pkg = os.path.join(args.artifact_dir, "deploy_package")
        print(f"\n部署产物: {pkg}/")
        if os.path.isdir(pkg):
            for f in sorted(os.listdir(pkg)):
                sz = os.path.getsize(os.path.join(pkg, f))
                print(f"  {f}  ({sz:,} bytes)")


if __name__ == "__main__":
    main()
