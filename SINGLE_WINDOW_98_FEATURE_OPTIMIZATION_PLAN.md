# Stage2 单窗 98% 特征筛选优化计划

## Summary

目标是在不增加模型复杂度的前提下，将 **Stage2 raw window accuracy** 提升到 98% 以上。主指标只统计真正进入 XGBoost 的 Stage2 raw 窗口，不把 Stage1 关闭窗口计入模型能力评估；后处理 `s07` 继续独立评估，不参与单窗 98% 判定。

核心约束：

- 不通过加树、加深度来提升准确率。
- XGBoost 复杂度不高于当前配置，必要时还要尝试缩小模型。
- 默认特征预算为 12-15 个。
- Stage2 窗口配置保持 `3s / 1s / skip_initial_windows=3`。

## Success Criteria

主验收目标：

- test split 的 Stage2 raw window accuracy >= 98.0%。
- 同时报告 precision、recall、F1、AUC 和 confusion matrix，避免 accuracy 被类别不平衡掩盖。
- valid/test 不允许数据泄漏；所有阈值、特征选择、填充值、校准只由 train/valid 决定。

候选模型优先级：

1. Stage2 raw window accuracy 达标。
2. FP 更低。
3. 特征数量更少。
4. 部署成本更低。
5. train/valid 稳定性更好。

## Current Baseline To Freeze

先固定当前默认流水线作为 baseline：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts
```

记录以下指标：

- `window_model_summary.accuracy`
- `window_model_summary.precision`
- `window_model_summary.recall`
- `window_model_summary.f1`
- `window_model_summary.confusion_matrix`
- `window_model_summary.stage1_pass_samples`

同时保留当前默认配置：

```text
window_sec = 3
stride_sec = 1
skip_initial_windows = 3
max_features = 15
```

## Feature Diagnostics

优化重点不是增加模型容量，而是把特征筛选做成可解释、可复现、可闭环的过程。

需要对所有候选特征生成诊断表：

- 缺失率。
- 方差和低方差过滤结果。
- train/valid 分布漂移。
- 单特征 AUC。
- 单特征 FP proxy。
- 所属 feature group。
- deployment cost。
- SHAP train/valid 一致性。
- 是否 scale-dependent。

重点标记三类风险特征：

- train 排名高但 valid 排名低的疑似过拟合特征。
- FP proxy 高、容易造成非佩戴误触的特征。
- 部署成本高但贡献低的复杂特征，例如部分 FFT、Entropy、复杂 waveform 特征。

## Feature Subset Search

默认特征预算为 12-15 个。不要只输出一个 `selected_features.json`，而是生成多套候选组合并统一评估。

候选组合建议：

| 组合 | 目标 |
|---|---|
| `accuracy_first` | 优先 raw window accuracy |
| `fp_safe` | 优先低 FP proxy |
| `deployment_light` | 优先低部署成本 |
| `balanced` | accuracy、FP proxy、部署成本加权 |
| `commercial_8_baseline` | 商业 8 特征 baseline，用作参照 |

每组候选组合必须使用相同模型复杂度评估，不允许通过模型容量变化解释结果差异。

建议输出：

```text
artifacts/feature_subset_search/
  subset_candidates.csv
  subset_eval_valid.csv
  subset_eval_summary.json
  best_subset_features.json
```

每条候选记录至少包含：

```text
subset_name
feature_names
n_features
group_count
deployment_cost_mean
scale_dependent_count
valid_raw_window_accuracy
valid_precision
valid_recall
valid_f1
valid_auc
valid_fp
valid_fn
score
```

## Feature Group Strategy

保留当前 `s04_feature_selection.py` 的分组体系，但需要把“为什么入选/为什么淘汰”输出清楚。

优先方向：

- IR、Green、Ambient 的低成本统计和 ratio 类特征优先。
- Green spatial 与 IR-Green cross 特征需要通过 valid 证明增益。
- ACC 特征至少进入候选池，但是否入选最终 12-15 个特征由组合评估决定。
- FFT、Entropy、复杂 waveform 特征设置更高进入门槛，除非 valid/test raw window 指标证明它们有稳定增益。
- 商业 8 特征必须作为 baseline 组合单独评估，判断当前新增特征是否真的有效。

不推荐方向：

- 不用 test split 参与筛选。
- 不靠提高 `n_estimators` 或 `max_depth` 达标。
- 不因为单个特征 train importance 高就直接入选。
- 不把后处理端到端效果反向混入单窗模型目标。

## Raw Window Error Analysis

需要导出 Stage2 raw window 级错误明细，用于反推特征问题。

建议输出：

```text
artifacts/window_error_analysis/
  raw_window_errors_valid.csv
  raw_window_errors_test.csv
  raw_window_error_summary.json
```

明细字段：

```text
sample_name
h5_file
target
pred
prob
window_start_sec
window_end_sec
stage1_enabled
mode
quality
ood_rate
is_fp
is_fn
selected_features_version
```

分层统计：

- 按 label 分层。
- 按 sample 分层，找高错误率样本。
- 按 H5 文件分层，找数据源偏差。
- 按 `mode` 分层，判断硬件通道模式是否影响特征。
- 按窗口起点分层，确认跳过前三窗后是否仍有早期窗口异常。
- 按 OOD/quality 分层，判断错误是否来自低质量窗口。

## Implementation Steps

### Step 1: 固化 baseline 指标

- 运行当前默认流水线。
- 保存 `end_to_end_eval_*`、`selected_features.json`、`model_bundle.pkl`。
- 从 `s06` 结果中提取 Stage2 raw window 指标作为 baseline。

### Step 2: 增加 raw window 错误明细导出

建议改动 `s06_deploy_eval.py`：

- 保留 `window_model_summary` 作为主指标。
- 增加 raw window 级 CSV 导出。
- CSV 只统计 XGBoost 实际推理窗口，不统计 Stage1 disabled 窗口。

验收：

- CSV 中 FP/FN 数量能和 `window_model_summary.confusion_matrix` 对齐。
- `stage1_enabled=0` 的窗口不进入主指标。

### Step 3: 增强 s04 特征诊断输出

建议改动 `s04_feature_selection.py`：

- 输出所有候选特征的诊断表。
- 增加 train/valid drift 指标。
- 将 FP proxy、deployment profile、SHAP consistency 汇总到一个 CSV。

验收：

- 每个候选特征都有 group、missing、variance、AUC、FP proxy、deployment cost。
- 被移除的特征能说明原因。

### Step 4: 增加候选特征子集搜索

建议在 `s04_feature_selection.py` 中新增候选组合生成与评估逻辑：

- 输入为清洗后的 train/valid 特征。
- 输出 12-15 个特征的多个候选组合。
- 每组候选使用固定 XGBoost 配置训练和 valid 评估。

固定模型复杂度：

```text
n_estimators <= current
max_depth <= current
learning_rate = current
min_child_weight = current or higher
reg_lambda = current or higher
```

验收：

- 每组组合都有 valid raw window 指标。
- 结果按 score 排序。
- 最优组合写入 `best_subset_features.json`。

### Step 5: 用最佳候选重训最终模型

- 将最佳 12-15 特征写入 `selected_features.json`。
- 运行 `s05_train_final_model.py`。
- 运行 `s06_deploy_eval.py`，评估 valid/test raw window 指标。

验收：

- valid raw window accuracy 达到或接近 98% 后，才看 test。
- test raw window accuracy >= 98.0% 才标记为达标候选。

### Step 6: 达标后冻结 Stage2，再搜后处理

当 Stage2 raw window 指标达标后：

```bash
python s06_deploy_eval.py \
  --artifact_dir artifacts \
  --split valid \
  --window_sec 3 \
  --stride_sec 1 \
  --skip_initial_windows 3 \
  --window_output_root window_outputs \
  --export_window_cache

python s07_postprocess_optimize.py \
  --artifact_dir artifacts \
  --split valid \
  --cache_root window_outputs \
  --max_sample_fp_rate 0.02 \
  --max_false_worn_event_rate 0.02 \
  --max_first_worn_output_p95_sec 6.0 \
  --fp_cost 4.0
```

后处理优化只读 `window_outputs`，不再影响 Stage2 特征筛选目标。

## Experiment Matrix

第一轮只改特征筛选，不改模型复杂度：

| 实验 | max_features | 策略 |
|---|---:|---|
| baseline | 15 | 当前 s04 默认策略 |
| commercial_8 | 8 | 商业 baseline 特征 |
| accuracy_first_15 | 15 | raw window accuracy 优先 |
| fp_safe_15 | 15 | FP proxy 优先 |
| deployment_light_15 | 15 | 低部署成本优先 |
| balanced_15 | 15 | accuracy + FP + cost 平衡 |
| balanced_12 | 12 | 压缩特征版本 |

第二轮根据第一轮结果细化：

- 若 FP 高：提高 FP proxy 权重。
- 若 FN 高：检查正样本召回低的特征组，优先增强 IR/Green/ACC 互补特征。
- 若 train 高 valid 低：降低 SHAP train-only 特征权重，提高 train/valid consistency 权重。
- 若某个 mode 集中出错：增加 mode 分层评估，考虑 mode-aware 特征或剔除 mode 不稳定特征。

## Test Plan

文档与命令检查：

```bash
python -m pytest D:\wearing_liveness\new\tests
```

建议新增测试：

- `s06` raw window 错误明细不统计 Stage1 disabled 窗口。
- raw window 错误明细 FP/FN 数量与 summary confusion matrix 一致。
- `s04` 候选组合搜索不读取 test split。
- `s04` 候选组合输出的特征数在 12-15 范围内。
- `s04` 输出每个候选特征的 group、FP proxy、deployment cost。

## Failure Handling

如果 12-15 个特征无法达到 98%：

1. 不立即增加模型复杂度。
2. 先查看错误分层报告，确认主要错误来源。
3. 尝试 15 个特征的不同组合策略。
4. 若仍无法达标，输出“当前特征预算下不可达”的证据：
   - 最优 valid/test 指标。
   - 主要 FP/FN 来源。
   - 是否集中于特定样本、H5、mode 或低质量窗口。
5. 再由人工决定是否放宽特征预算或重新设计特征。

## Assumptions

- 单窗 98% 主目标指 Stage2 raw window accuracy。
- 特征数量目标为 12-15 个。
- 不增加模型复杂度，必要时可进一步缩小模型。
- 当前无法直接查看真实 H5 数据，因此所有判断必须通过可复现实验、错误明细和分层报告完成。
