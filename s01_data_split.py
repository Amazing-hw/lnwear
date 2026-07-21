# s01_data_split.py
# -*- coding: utf-8 -*-

"""
步骤1：数据整理与切分（优化）

主要改进：
1. H5 扫描按文件并行（IO bound）
2. 严格读取 frequency 与 ppg_config；缺失、非法或分组内不一致时跳过并统计原因。
3. 支持 grouped-window H5：一个 record 下多个 *_w20_1 窗口 group，
   按 w 编号排序并保留窗口 label 序列。
5. 切分逻辑、输出 schema 向后兼容

CLI 兼容旧版：仅新增 --n_workers 可选参数。
"""

import os
import glob
import json
import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import h5py
from sklearn.model_selection import train_test_split


WINDOW_NAME_RE = re.compile(r"(?:^|_)w(?P<index>\d+)_(?P<label>[01])$")
VALID_FREQUENCIES = {25, 100}
VALID_PPG_CONFIGS = {0, 1, 2}
FILTER_REASON_KEYS = (
    "missing_frequency",
    "invalid_frequency",
    "missing_ppg_config",
    "invalid_ppg_config",
    "inconsistent_metadata",
)


def _env_flag(name):
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def resolve_n_workers(n_workers=None, n_items=None, cap=4):
    if _env_flag("WL_FORCE_SERIAL"):
        return 1
    if n_workers is None:
        n_workers = max(1, min(cap, (os.cpu_count() or cap) // 2))
    try:
        resolved = max(1, int(n_workers))
    except (TypeError, ValueError):
        resolved = 1
    if n_items is not None and int(n_items) <= 1:
        return 1
    if n_items is not None:
        resolved = min(resolved, max(1, int(n_items)))
    return resolved


def multiprocessing_context_from_env():
    method = os.environ.get("WL_MP_START_METHOD", "").strip()
    if not method:
        return None
    import multiprocessing as mp
    return mp.get_context(method)


def find_h5_files(dataset_dir):
    h5_files = glob.glob(os.path.join(dataset_dir, "*.h5"))
    if len(h5_files) == 0:
        h5_files = glob.glob(os.path.join("..", dataset_dir, "*.h5"))
    return sorted(h5_files)


def parse_window_name(name):
    """Return (window_index, label) parsed from names ending in *_w20_1."""
    match = WINDOW_NAME_RE.search(str(name))
    if not match:
        return None
    return int(match.group("index")), int(match.group("label"))


def _empty_filter_counts():
    return {key: 0 for key in FILTER_REASON_KEYS}


def _read_scalar_int(group, field):
    if field not in group:
        return None
    try:
        return int(group[field][()])
    except (TypeError, ValueError, OverflowError):
        return None


def _record_filtered(filtered, h5_file, sample_name, reason, detail=""):
    filtered[reason] += 1
    suffix = f": {detail}" if detail else ""
    print(f"[SKIP] h5={h5_file}, sample={sample_name}, reason={reason}{suffix}")


def _validate_standard_metadata(group):
    frequency = _read_scalar_int(group, "frequency")
    ppg_config = _read_scalar_int(group, "ppg_config")
    if "frequency" not in group:
        return None, None, "missing_frequency"
    if frequency not in VALID_FREQUENCIES:
        return None, None, "invalid_frequency"
    if "ppg_config" not in group:
        return None, None, "missing_ppg_config"
    if ppg_config not in VALID_PPG_CONFIGS:
        return None, None, "invalid_ppg_config"
    return int(frequency), int(ppg_config), None


def _resolve_grouped_field(parent, children, field, valid_values):
    missing_reason = f"missing_{field}"
    invalid_reason = f"invalid_{field}"
    parent_has_value = field in parent
    parent_value = _read_scalar_int(parent, field) if parent_has_value else None
    if parent_has_value and parent_value not in valid_values:
        return None, invalid_reason

    child_values = []
    for child in children:
        if field not in child:
            if not parent_has_value:
                return None, missing_reason
            continue
        value = _read_scalar_int(child, field)
        if value not in valid_values:
            return None, invalid_reason
        child_values.append(int(value))

    if parent_has_value:
        if any(value != parent_value for value in child_values):
            return None, "inconsistent_metadata"
        return int(parent_value), None
    if not child_values:
        return None, missing_reason
    if len(set(child_values)) != 1:
        return None, "inconsistent_metadata"
    return int(child_values[0]), None


def _sample_target_from_window_labels(labels):
    if not labels:
        return 0
    if len(set(labels)) == 1:
        return int(labels[0])
    return int(np.mean(labels) >= 0.5)


def _scan_grouped_window_sample(h5_file, sample_name, grp, filtered):
    windows = []
    for child_name in grp.keys():
        parsed = parse_window_name(child_name)
        if parsed is None:
            continue
        child = grp[child_name]
        if not isinstance(child, h5py.Group) or "ppg" not in child:
            continue
        shape = child["ppg"].shape
        if not is_supported_ppg_shape(shape):
            filtered["channel_count"] += 1
            continue
        window_index, label = parsed
        windows.append((window_index, label, child_name, shape, child))

    if not windows:
        return None

    windows.sort(key=lambda item: item[0])
    children = [item[4] for item in windows]
    frequency, reason = _resolve_grouped_field(
        grp, children, "frequency", VALID_FREQUENCIES
    )
    if reason is not None:
        _record_filtered(filtered, h5_file, sample_name, reason)
        return None
    ppg_config, reason = _resolve_grouped_field(
        grp, children, "ppg_config", VALID_PPG_CONFIGS
    )
    if reason is not None:
        _record_filtered(filtered, h5_file, sample_name, reason)
        return None
    labels = [int(item[1]) for item in windows]
    shapes = [list(item[3]) for item in windows]
    return {
        "sample_name": sample_name,
        "h5_file": h5_file,
        "target": _sample_target_from_window_labels(labels),
        "ppg_shape": [len(windows)] + shapes[0],
        "frequency": int(frequency),
        "ppg_config": int(ppg_config),
        "window_layout": "grouped_windows",
        "window_names": [str(item[2]) for item in windows],
        "window_indices": [int(item[0]) for item in windows],
        "window_labels": labels,
        "window_label_counts": {
            "target0": int(sum(1 for x in labels if x == 0)),
            "target1": int(sum(1 for x in labels if x == 1)),
        },
    }


def _scan_one_h5(h5_file):
    """单文件扫描。返回 (samples_list, filtered_counts_dict)。"""
    samples = []
    filtered = _empty_filter_counts()
    try:
        with h5py.File(h5_file, "r") as f:
            for sample_name in f.keys():
                if re.search(r"(0003|0103)\d{4}", sample_name):
                    continue

                if re.search(r"(0104|0002)\d{4}", sample_name):
                    continue

                grp = f[sample_name]
                if "ppg" not in grp:
                    grouped = _scan_grouped_window_sample(h5_file, sample_name, grp, filtered)
                    if grouped is not None:
                        samples.append(grouped)
                    continue
                if "target" not in grp:
                    continue
                try:
                    label = int(grp["target"][()])
                except (TypeError, ValueError, KeyError):
                    continue

                shape = grp["ppg"].shape

                if not is_supported_ppg_shape(shape):
                    filtered["channel_count"] += 1
                    continue
                frequency, ppg_config, reason = _validate_standard_metadata(grp)
                if reason is not None:
                    _record_filtered(filtered, h5_file, sample_name, reason)
                    continue

                samples.append({
                    "sample_name": sample_name,
                    "h5_file": h5_file,
                    "target": int(label),
                    "ppg_shape": list(shape),
                    "frequency": int(frequency),
                    "ppg_config": int(ppg_config),
                })
    except OSError as e:
        print(f"读取 {h5_file} 失败: {e}")
    except Exception as e:
        # 兜底但要打印，方便定位
        print(f"读取 {h5_file} 异常: {e}")
    return samples, filtered


def is_supported_ppg_shape(shape):
    """Accept any 2-D (samples, channels) or 3-D (windows, samples, channels) PPG."""
    return len(shape) in (2, 3)


def scan_h5_samples(dataset_dir, n_workers=None):
    """并行扫描 H5（n_workers=None 自动，=1 单进程）。"""
    h5_files = find_h5_files(dataset_dir)
    print(f"找到 {len(h5_files)} 个 H5 文件")
    print(f"[生产] 使用全部H5文件: {h5_files}")

    n_workers = resolve_n_workers(n_workers, n_items=len(h5_files))

    samples = []
    filtered_count = _empty_filter_counts()

    if n_workers == 1 or len(h5_files) <= 1:
        for h5_file in h5_files:
            s, fc = _scan_one_h5(h5_file)
            samples.extend(s)
            for key in FILTER_REASON_KEYS:
                filtered_count[key] += fc[key]
    else:
        pool_kwargs = {"max_workers": n_workers}
        mp_ctx = multiprocessing_context_from_env()
        if mp_ctx is not None:
            pool_kwargs["mp_context"] = mp_ctx
        with ProcessPoolExecutor(**pool_kwargs) as ex:
            futures = {ex.submit(_scan_one_h5, h5_file): h5_file for h5_file in h5_files}
            for done_count, fut in enumerate(as_completed(futures), 1):
                try:
                    s, fc = fut.result()
                except Exception as e:
                    print(f"扫描 {futures[fut]} 异常: {e}")
                    s, fc = [], _empty_filter_counts()
                samples.extend(s)
                for key in FILTER_REASON_KEYS:
                    filtered_count[key] += fc[key]
                if len(futures) >= 10 and (done_count % max(1, len(futures) // 10) == 0 or done_count == len(futures)):
                    print(f"  s01 progress: {done_count}/{len(futures)} files", flush=True)

    # 保证并行下样本顺序的确定性（按 h5_file + sample_name 排序）
    for reason in FILTER_REASON_KEYS:
        if filtered_count[reason] > 0:
            print(f"filtered samples: {reason}={filtered_count[reason]}")

    samples.sort(key=lambda s: (s["h5_file"], s["sample_name"]))
    return samples


def split_samples(samples, valid_size=0.15, test_size=0.15, random_state=42):
    """sample 级分层切分。"""
    y = np.array([s["target"] for s in samples])
    indices = np.arange(len(samples))

    train_valid_idx, test_idx = train_test_split(
        indices, test_size=test_size,
        random_state=random_state, stratify=y
    )
    y_train_valid = y[train_valid_idx]
    valid_ratio_in_train_valid = valid_size / (1.0 - test_size)
    train_idx, valid_idx = train_test_split(
        train_valid_idx, test_size=valid_ratio_in_train_valid,
        random_state=random_state, stratify=y_train_valid
    )
    return {
        "train": [samples[i] for i in train_idx],
        "valid": [samples[i] for i in valid_idx],
        "test": [samples[i] for i in test_idx],
    }


def summarize_split(split):
    for part in ["train", "valid", "test"]:
        arr = split[part]
        n0 = sum(1 for s in arr if s["target"] == 0)
        n1 = sum(1 for s in arr if s["target"] == 1)
        print(f"{part}: total={len(arr)}, target0={n0}, target1={n1}")


def export_split_analysis_plot(split, artifact_dir):
    """Export a compact PNG summary of split size and class balance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scientific_figures import save_scientific_figure

    parts = ["train", "valid", "test"]
    rows = []
    for part in parts:
        samples = list(split.get(part, []))
        for target in (0, 1):
            count = sum(int(sample.get("target", 0)) == target for sample in samples)
            rows.append({
                "split": part,
                "target": target,
                "sample_count": int(count),
                "split_total": int(len(samples)),
                "class_ratio": float(count / max(len(samples), 1)),
            })

    count0 = [next(row["sample_count"] for row in rows if row["split"] == part and row["target"] == 0) for part in parts]
    count1 = [next(row["sample_count"] for row in rows if row["split"] == part and row["target"] == 1) for part in parts]
    ratios = [next(row["class_ratio"] for row in rows if row["split"] == part and row["target"] == 1) for part in parts]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor="white")
    x = np.arange(len(parts))
    axes[0].bar(x, count0, color="#4C78A8", label="not worn")
    axes[0].bar(x, count1, bottom=count0, color="#E07B53", label="worn")
    axes[0].set_xticks(x, parts)
    axes[0].set_ylabel("samples")
    axes[0].set_title("Split size and class counts", loc="left", weight="bold")
    axes[0].legend(frameon=False)
    axes[1].bar(x, ratios, color="#2A9D8F")
    axes[1].axhline(0.5, color="#7A7A7A", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, parts)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("worn ratio")
    axes[1].set_title("Class balance", loc="left", weight="bold")
    fig.suptitle("Dataset split audit", fontsize=13, weight="bold", x=0.04, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    artifact_dir = os.fspath(artifact_dir)
    out_path = os.path.join(artifact_dir, "report_plots", "s01_split_analysis.png")
    split_path = os.path.join(artifact_dir, "splits.json")
    outputs = save_scientific_figure(
        fig, out_path, source_data=rows,
        core_conclusion="Train, validation, and test partitions preserve visible class balance and sample counts.",
        panel_map={"a": "Sample counts by split and target.", "b": "Positive-class ratio by split."},
        inputs=[split_path] if os.path.isfile(split_path) else (),
        split="train_valid_test",
        n_definition="one source row per split and binary target",
        statistics={"center": "count and proportion", "interval": "none"},
        reviewer_risks=["Class balance does not by itself prove subject or session independence."],
    )
    plt.close(fig)
    return outputs


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--valid_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_workers", type=int,
                        default=max(1, min(4, (os.cpu_count() or 4) // 2)),
                        help="并行 worker 数")

    if args is None:
        args = parser.parse_args()

    os.makedirs(args.artifact_dir, exist_ok=True)
    samples = scan_h5_samples(args.dataset_dir, n_workers=args.n_workers)

    print(f"总样本数: {len(samples)}")
    print(f"target=0: {sum(1 for s in samples if s['target'] == 0)}")
    print(f"target=1: {sum(1 for s in samples if s['target'] == 1)}")

    split = split_samples(
        samples,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.random_state
    )
    summarize_split(split)

    out_path = os.path.join(args.artifact_dir, "splits.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    print(f"切分结果已保存: {out_path}")
    outputs = export_split_analysis_plot(split, args.artifact_dir)
    print(f"切分分析 PNG 已保存: {outputs['png']}")


if __name__ == "__main__":
    main()
