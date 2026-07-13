# 人工固化特征选择设计

日期：2026-07-10
状态：已确认，待实施

## 1. 背景与目标

当前流水线会从特征池自动完成清洗、排序、特征数量搜索和局部替换，然后直接训练最终模型。该方式便于追求验证集指标，但最终特征集合可能随数据和搜索过程变化，出现误判时难以从传感机理、信号质量和特征含义解释模型行为。

本设计将特征工程拆成两个明确阶段：

1. 数据驱动阶段负责对全部可部署候选特征生成排序和诊断证据。
2. 人工决策阶段由使用者固化最终特征集合，训练阶段只允许使用该集合。

目标是保证最终模型的特征来源可审计、顺序稳定、选择理由可记录，同时保留 XGBoost 参数搜索、概率校准和阈值选择能力。

## 2. 非目标

- 不改变 Stage1 IR 固定门控逻辑。
- 不修改 Stage2 特征公式或信号预处理算法。
- 不自动推荐新的生理或物理特征。
- 不在人工模式中自动补齐、替换或删除人工选择的特征。
- 不移除现有自动模式，自动模式仅作为显式兼容路径保留。

## 3. 方案比较

### 方案 A：独立人工特征文件（采用）

`s04` 输出排序证据，人工选择写入独立的 `manual_selected_features.json`，`s05` 严格读取该文件。

优点：人工决策与自动产物边界清楚，不会被后续 `s04` 覆盖；便于版本控制、审计和复现实验。
缺点：训练变成两阶段操作，需要人工维护一个文件。

### 方案 B：直接编辑 `selected_features.json`

优点：改动较少。
缺点：该文件属于 `s04` 自动产物，重新排序时会被覆盖；无法清楚区分机器选择和人工选择。

### 方案 C：通过 CLI 传入特征列表

优点：无需新增文件格式。
缺点：长列表容易输错，选择理由和来源无法稳定归档，命令行也不适合作为模型审计记录。

## 4. 工作流

### 4.1 排序阶段

运行到 `s04`：

```powershell
python s08_run_pipeline.py `
  --dataset_dir dataset `
  --artifact_dir artifacts `
  --feature_selection_mode manual `
  --stop_after s04
```

排序阶段继续输出：

- `artifacts/feature_ranking_full.json`
- `artifacts/feature_ranking_full.csv`
- `artifacts/ranked_features.json`
- `artifacts/feature_diagnostics.csv`
- `artifacts/selected_features.json`，仅作为现有自动选择报告，不作为人工模式训练输入
- `artifacts/manual_selected_features.template.json`，人工文件模板

其中 `feature_ranking_full.*` 是人工决策的唯一排序来源；现有
`ranked_features.json` 仅供自动模式兼容使用。

### 4.2 完整排序定义

完整排序覆盖特征池中每个满足以下基础条件的特征，且每个特征只出现一次：

- 数值类型
- 非元数据列
- 非 IR 派生 Stage2 特征
- 满足当前端侧可部署策略

自动清洗发现的高缺失、低方差、高相关或高 VIF 特征仍保留在完整排序中，
并通过 `risk_flags` 和 `removed_reason` 标注。人工模式不会因为这些自动规则
剥夺使用者的选择权；无法计算的排名分量记为 0 并排在末尾。

`eligible_for_manual_selection` 仅在特征无法进入训练时为 false，例如 train
或 valid 缺列、两侧没有可用有限值、违反无 IR 策略或违反部署策略。高相关、
高 VIF、低方差和分布漂移本身只产生风险标记，不会把特征设为不可选。

为避免继续引入难以解释的模型内重要性，完整排序采用透明的 train-only
单特征证据：

```text
fold_separation = 2 * abs(group_fold_auc - 0.5)
ranking_score =
    0.45 * mean(fold_separation)
  + 0.25 * min(fold_separation)
  + 0.20 * fp_proxy_fit
  + 0.10 * deployment_fit
```

其中 group fold 按 `sample_name` 划分；`fp_proxy_fit` 沿用当前连续窗口误报代理，
`deployment_fit` 沿用当前部署成本和尺度依赖评分。valid 上的 PSI、KS、均值漂移
和单特征 AUC 只作为旁证列展示，不进入 `ranking_score`，避免用 valid 反复优化排名。

完整排序的每条记录至少包含：

- `rank`、`feature`、`ranking_score`
- train group-fold AUC/separation 的均值、最小值和标准差
- train FP proxy 的 sample/state FP rate 与 fit
- 部署成本、部署 fit、尺度依赖标记
- missing、variance、correlation、VIF 风险标记
- valid 漂移诊断和 valid 单特征 AUC，仅供人工参考
- `eligible_for_manual_selection` 及不可选原因

### 4.3 人工决策阶段

人工结合排序、稳定性、漂移、误报代理、部署成本和信号机理，创建：

`artifacts/manual_selected_features.json`

建议格式：

```json
{
  "schema_version": 1,
  "ranking_source": "feature_ranking_full.json",
  "selected_features": [
    "GREEN_CORR",
    "GTOP2_BAND_ENERGY_RATIO",
    "ACC_STILL_SCORE"
  ],
  "selection_notes": {
    "GREEN_CORR": "绿光周期波形稳定性",
    "GTOP2_BAND_ENERGY_RATIO": "降低单路绿光失效影响",
    "ACC_STILL_SCORE": "辅助识别静止非佩戴场景"
  }
}
```

`selection_notes` 可省略，不影响训练。

### 4.4 固定特征训练阶段

```powershell
python s08_run_pipeline.py `
  --artifact_dir artifacts `
  --feature_selection_mode manual `
  --manual_feature_file artifacts/manual_selected_features.json `
  --skip s01,s02,s03,s04,s04_search `
  --stop_after s06_cb
```

人工模式下，`s05` 仅训练一次固定特征集合。允许继续执行：

- XGBoost 超参数搜索
- train group-CV
- train 学习 fill/clip 参数
- valid 子集上的 isotonic 概率校准
- valid 独立子集上的窗口阈值选择
- s06 端到端评估与部署产物导出

禁止执行：

- Top-K 特征截断
- `8,10,12,15,18` 特征数量搜索
- local-swap 特征替换
- `s04_search` 自动候选子集覆写

## 5. 命令行与默认行为

### `s08_run_pipeline.py`

新增：

```text
--feature_selection_mode {manual,auto}
--manual_feature_file PATH
```

主流水线默认采用 `manual`。如需复现旧的一键自动流程，必须显式传入：

```powershell
python s08_run_pipeline.py --feature_selection_mode auto
```

当运行范围包含 `s05` 且处于人工模式时，人工文件必须存在并通过校验，否则流水线报错停止，不回退到自动模式。

### `s05_train_final_model.py`

新增相同的模式和文件参数。人工模式直接从人工文件读取有序特征列表。自动模式保持现有 `ranked_features.json`、Top-K、特征数量搜索和 local-swap 行为。

## 6. 严格校验契约

人工模式进入训练前必须同时满足：

1. 文件存在且为合法 JSON 对象。
2. `schema_version` 为支持的版本。
3. `selected_features` 是非空字符串数组。
4. 列表不存在重复项，顺序原样保留。
5. 每个特征都存在于 `feature_ranking_full.json`，且
   `eligible_for_manual_selection=true`。
6. 每个特征都存在于 train 和 valid 特征池。
7. 不包含任何 IR 派生 Stage2 特征。
8. 每个特征都满足当前端侧可部署策略。
9. `selection_notes` 如存在，必须是对象且只能引用已选择特征。

任一校验失败都应输出具体特征名和失败原因，并以非零状态退出。代码不得静默删除、重排、填补或回退。

## 7. 数据流与产物溯源

人工模式的数据流为：

```text
feature_pool_train/valid.csv
  -> s04 完整排序与诊断
  -> feature_ranking_full.json / feature_ranking_full.csv
  -> 人工审查
  -> manual_selected_features.json
  -> s05 固定特征训练和模型参数搜索
  -> model_bundle.pkl / final_model_config.json
  -> s06 评估与部署导出
```

以下溯源信息写入 `model_bundle.pkl` 和 `final_model_config.json`：

- `feature_selection_mode = manual`
- 人工文件路径
- 人工文件 SHA-256
- 排序来源文件路径及 SHA-256
- 最终有序特征列表
- 可选人工选择理由

部署导出的 `FEATURE_ORDER`、fill、clip、XGBoost 输入顺序必须与人工文件完全一致。

## 8. 错误处理

- 完整排序文件缺失：提示先运行到 `s04`。
- 人工文件缺失：提示人工模式不允许自动回退。
- 特征未知或不具备人工选择资格：列出特征及完整排序记录中的不可选原因。
- IR 或不可部署特征：列出违反的策略，不自动过滤。
- 人工模式仍传入多个 `model_search_feature_counts` 或启用 local-swap：直接报参数冲突，避免用户误以为特征被冻结但实际仍被搜索修改。
- 自动模式行为和旧产物格式保持兼容。

## 9. 测试设计

新增或调整测试覆盖：

1. 合法人工文件按原顺序加载。
2. 完整排序恰好覆盖所有非 IR、可部署数值候选，且无重复和遗漏。
3. 排名分数只依赖 train，valid 指标只作为诊断旁证。
4. 文件缺失、空列表、重复项和未知特征均失败。
5. IR 特征和不可部署特征均失败。
6. train/valid 任一特征池缺列时失败。
7. 人工模式不进入特征数量搜索和 local-swap。
8. 人工模式仍可执行固定特征集上的 XGBoost 参数搜索。
9. `model_bundle.feature_names`、配置文件、部署脚本 `FEATURE_ORDER` 与人工文件逐项相等。
10. `s08 --dry_run` 在人工模式只生成一次 `s05` 命令，不产生多 K quick/full search。
11. 显式 `--feature_selection_mode auto` 保持现有自动流程。
12. 端到端合成 H5 测试覆盖“排序 -> 人工文件 -> 训练 -> 部署导出”。

## 10. 验收标准

- 人工模式下，最终模型特征集合只能来自人工文件。
- 完整排序覆盖所有可供人工选择的候选，自动清洗仅标记风险而不缩小人工候选范围。
- 人工名单及顺序在训练、评估和部署产物之间完全一致。
- 不存在隐式 Top-K、特征数量搜索、local-swap 或自动回退。
- 超参数搜索、校准、阈值选择和端到端评估仍可正常运行。
- 自动模式可通过显式参数继续使用。
- 所有相关单元测试和端到端守卫测试通过。
