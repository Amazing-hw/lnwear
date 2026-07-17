# 手表佩戴活体检测流水线

本项目使用 PPG、三固定位置绿光区和 ACC 构建佩戴活体检测模型，训练与最终部署判决统一采用 XGBoost。活动流程中不包含 IR DC/AC/DC 阈值门控，也不存在阈值结果与模型结果的融合；最终混淆矩阵、错误样本和部署输出均只来自 XGBoost。

默认流程先生成完整特征排序和人工选择 CSV，等待人工确认后再训练：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts
```

## 1. 当前流程

```text
H5 数据
  → s01 数据扫描、元数据校验、train/valid/test 分组切分
  → s03 全量合法窗口特征提取
  → s04 完整特征排序与 manual_feature_selection.csv
  → 人工只修改 selected 列
  → s05 固定所选特征，充分搜索 XGBoost 超参数并进行 hard-negative 候选训练
  → s06 纯 XGBoost 评估、错误分析与部署产物导出
  → s07 可选：独立的 EMA/状态机后处理搜索
```

默认不运行后处理搜参。主评估使用模型包内冻结的窗口阈值，样本级默认采用 `prob_mean`，即窗口 XGBoost 概率均值与模型阈值比较。`mean_vote` 和 `state_machine` 只在显式指定时作为独立实验口径，不改变默认主结果。

## 2. H5 数据约定

每条记录必须提供：

- `frequency`：只能是 `25` 或 `100`。25 Hz 数据直接提取特征；100 Hz 数据先用多相降采样到 25 Hz。
- `ppg_config`：只能是 `0`、`1` 或 `2`，用于生成顺序稳定的三个固定物理光区。
- PPG 与可选 ACC 数据。

绿光区映射使用零基通道编号：

| `ppg_config` | 配置 | 三固定光区 |
|---:|---|---|
| 0 | 3 通道 | `g1=ch3`，`g2=ch4`，`g3=ch5` |
| 1 | 6 通道 | `g1=(ch3+ch9)/2`，`g2=(ch4+ch10)/2`，`g3=(ch5+ch11)/2` |
| 2 | 9 通道 | `g1=(ch6+ch9+ch12)/3`，`g2=(ch7+ch10+ch13)/3`，`g3=(ch8+ch11+ch14)/3` |

读取并按窗口编号排序后，每条数据固定删除前三个和后三个窗口。窗口不足七个时，该记录不会产生可用窗口。元数据缺失、取值非法或通道数不足的记录会被跳过，并输出统计与原因。

## 3. 三光区绿光处理

三个区域具有固定物理位置和稳定顺序。特征池同时保留以下互补信息：

- 整体稳健表示：三光区均值、中位波形、两区组合和 top2 复合波形。
- 固定位置特征：每个光区的相对 DC、相对 AC、AC/DC、周期性和环境光相关性。
- 空间一致性：两两相关、主频差、相位集中、频谱共识和三光区支持度。
- 局部异常：弱区差距、top2 与全区能量比、固定区失配和单边翘起相关特征。
- 运动耦合：ACC 强度、jerk、ACC–PPG 延迟相关和 PSD 相似性。

top2 的选择依据是去趋势脉动分量的 AC-RMS 能量；RMS 只用于衡量脉动能量，不直接等同于信号质量。选择后对两个区的原始信号逐点等权平均，再统一预处理和提取特征。相关性、周期性和频谱支持共同约束 top2 的解释，避免只依赖幅值把运动伪影误认为高质量信号。

商用八特征仍保留在受治理特征池中。原有绿光平均类公式按三个固定光区的统一表示计算，另有三光区专用的 top2、中位、pair 和固定位置候选，最终是否使用由完整排序和人工 CSV 决定。

完整 126 项候选的公式、生理/物理意义、预期方向、鲁棒性、泛化风险和工程成本见 [FEATURE_INTERPRETABILITY_GUIDE.md](FEATURE_INTERPRETABILITY_GUIDE.md)。

## 4. 人工选择特征

第一阶段运行到 `s04`：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --stop_after s04
```

主要输出：

- `artifacts/feature_ranking_full.csv`：完整排序。
- `artifacts/manual_feature_selection.csv`：唯一允许人工编辑的选择文件。
- `artifacts/feature_pool_completeness.json`：特征池完整性检查。

人工选择规则：

1. 只修改 `manual_feature_selection.csv` 的 `selected` 列。
2. `selected=1` 表示选中，`selected=0` 表示不选；只能填写整数 0 或 1。
3. 不修改列名、列顺序、行顺序、特征名或其他统计字段。
4. 只选择 `eligible=1` 的特征，且至少选择一个。
5. 人工模式不搜索特征数量，也不会用 Top-K、分组上限或 FFT 上限覆盖人工结果。

恢复训练：

```bash
python s08_run_pipeline.py \
  --artifact_dir artifacts \
  --feature_selection_mode manual \
  --manual_feature_file artifacts/manual_feature_selection.csv \
  --skip s01,s03,s04 \
  --stop_after s06_cb
```

加载时会校验 CSV schema、完整排序 SHA256、特征池版本和所有不可变字段。训练、模型 JSON、独立部署特征提取脚本、公式文件和 C 契约严格使用 CSV 中选中特征的名称、顺序和数量。

## 5. 模型训练与搜参

`s05_train_final_model.py` 固定人工特征后执行 staged group-CV 搜参：

- Stage A 快速筛选大量候选；Stage B 对前若干候选执行完整 grouped CV。
- 最大树深可搜索到 5。
- 树数量候选硬限制为不超过 50。
- 特征数量在人工模式下不参与搜参。
- hard-negative 候选只使用 train OOF 误报挖掘；只有 valid 指标满足接受条件才替换参考模型。
- test 只用于冻结配置后的只读最终评估，不参与特征、阈值或超参数选择。

如需无人值守基线：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts_auto \
  --feature_selection_mode auto
```

## 6. 纯 XGBoost 评估

直接运行：

```bash
python s06_deploy_eval.py \
  --artifact_dir artifacts \
  --split test \
  --method prob_mean \
  --n_workers 8
```

评估语义为 `xgboost_only_v1`：

- 窗口预测：`probability >= model_bundle.pkl` 中冻结的 `threshold`。
- 样本主预测：默认 `mean(window_probability) >= threshold`。
- 单个窗口特征提取失败时会剔除该窗口并保留失败计数/示例；不会用 `0.0` 概率伪装模型输出。
- 全部候选窗口失败或裁边后没有可用窗口时，样本明确标记为 fallback、按 0 输出并记录原因。
- 最终 TP、FP、TN、FN 不受任何 IR 阈值或外部门控影响。

主要输出包括：

- `end_to_end_eval_<split>_<method>.json`
- `per_sample_xgboost_windows.csv`
- `per_sample_final_prediction.csv`
- `hard_negatives_<split>_<method>.json`
- `window_error_analysis_<split>_<method>.csv/json`
- `error_stratification_<split>_<method>.json`
- `report_plots/s06_deploy_report.png`

## 7. 可选后处理

后处理默认关闭。需要时先导出 XGBoost 窗口缓存，再显式运行 `s07`：

```bash
python s06_deploy_eval.py \
  --artifact_dir artifacts \
  --split valid \
  --method prob_mean \
  --export_window_cache

python s07_postprocess_optimize.py \
  --artifact_dir artifacts \
  --split valid
```

缓存 schema 为 `xgboost_window_outputs_v1`，只包含 XGBoost 概率、预测、质量/OOD、窗口时间和模型契约，不包含门控或融合字段。后处理只能在 valid 上选择参数，再对 test 做冻结回放。

## 8. 并行策略

`--n_workers` 会由 `s08` 传递到支持并行的各步骤：

- `s01`：按 H5 文件扫描。
- `s03`：按样本提取特征。
- `s04`：按 seed × group-fold 计算稳定性和排序诊断。
- `s05`：模型候选并行，候选内部 XGBoost 线程限制为 1，避免嵌套过度并行。
- `s06`：按样本推理。
- `s07`：按后处理网格点搜索。

显式传入的 worker 数不设项目级上限，但会按实际任务数量收缩。机器内存、进程启动开销和底层 BLAS/OpenMP 线程仍可能限制有效加速。排错时可设置 `WL_FORCE_SERIAL=1` 强制串行。

## 9. 部署产物

完整流程最终导出：

- `model_bundle.pkl`：训练侧统一模型包。
- `final_model.json` / `deploy_xgboost.json`：XGBoost 模型。
- `deploy_feature_extractor.py`：独立、无项目运行时依赖的特征提取脚本。
- `deploy_selected_feature_formulas.json`：所选特征公式。
- `stage2_feature_catalog.json` / `stage2_c_contract.json`：特征顺序和 C 实现契约。
- `golden_vectors.json`：端侧特征向量和概率一致性测试。
- `deploy_package/model_params.json`：阈值、fill、clip、特征顺序和模型元数据。
- `deploy_cookbook.json`：通道映射、预处理、特征和 XGBoost 推理配方。

部署最小组合为独立的 `deploy_feature_extractor.py`、XGBoost 模型 JSON，以及包含特征顺序、fill/clip 和阈值的模型元数据。部署判决不需要任何 IR 阈值配置文件。

## 10. 验证

语法检查：

```bash
python -m py_compile s01_data_split.py s03_extract_feature_pool.py s04_feature_selection.py s05_train_final_model.py s06_deploy_eval.py s07_postprocess_optimize.py s08_run_pipeline.py
```

全量测试：

```bash
python -m pytest -q --basetemp .pytest_tmp_full
```

快速检查将执行哪些步骤：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --dry_run
```
