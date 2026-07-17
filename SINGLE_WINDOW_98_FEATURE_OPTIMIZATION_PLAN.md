# Stage2 单窗准确率优化方案（现行）

更新时间：2026-07-16
适用版本：`stage2_interpretable_v8`，126 项受治理候选

## 目标与约束

目标是在不依赖后处理掩盖单窗错误的前提下，提高 Stage2 单窗 accuracy、precision、recall、
F1 和跨场景稳定性。不得用 test split 反复选择特征、阈值或模型参数。

现行约束：

- 默认窗口为 5 秒，步长为 1 秒；3 秒窗只作为显式独立实验。
- 每条数据读取并完成窗口排序后直接使用 `[3:-3]`，删除前三包与后三包。
- XGBoost 始终使用全量合法窗口，不经过 IR DC/ACDC 阈值过滤或结果融合。
- 模型特征池不使用 IR 派生特征；最终判决和全部统计均来自 XGBoost。
- 特征池为 v8/126，包含 `mode`、商用八特征映射、三固定位置绿光区及鲁棒空间候选。
- 最终特征名称、顺序和数量由 `manual_feature_selection.csv` 的人工选择决定。
- 人工模式只搜索固定特征集合上的模型参数，不搜索特征数量或执行 local-swap。
- XGBoost `max_depth` 搜索范围为 2、3、4、5，最终模型仍受节点预算约束。
- 商用八特征保留为候选特征，不存在独立商业模型对比流水线。
- 所有绘图只输出 600 DPI PNG，不使用浏览器或 HTML。

## 推荐流程

### 1. 生成完整特征排序

```powershell
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts
```

默认人工模式运行到 s04，生成：

```text
artifacts/feature_ranking_full.csv
artifacts/feature_ranking_full.json
artifacts/manual_feature_selection.csv
```

完整排序覆盖全部具备训练资格的候选。缺失率、低方差、高相关、VIF、漂移、FP proxy、
部署成本和风险标记用于人工判断，不会擅自替换人工选择。

### 2. 人工选择特征

只修改 `manual_feature_selection.csv` 的 `selected` 列：`1` 表示入选，`0` 表示不入选。
不能修改特征名、排序、公式、版本、风险字段或 ranking SHA。保存后重新运行同一命令，s05
会冻结人工选择的名称、顺序、数量和文件哈希。

人工筛选时优先检查：

- train repeated group-CV 与 valid 单窗指标是否一致；
- FP/FN 是否集中在特定 mode、设备、H5、时间位置或三光区失衡场景；
- 固定位置候选是否真正补充空间信息，而不是重复整体均值；
- FFT、熵和复杂相关特征的增益是否覆盖端侧成本；
- 商用八特征是否仍提供稳定基线信息，但不单独训练商业对比模型。

### 3. 固定特征集合搜索模型参数

模型搜索使用 staged group-CV，并行评估候选但限制每个 XGBoost 模型内部线程，避免嵌套
并行过载。树数候选为 20、25、30、35、40、45、50，硬上限为 50；同时搜索深度 2-5、学习率、子采样、列采样、正则化、最小叶节点权重、
概率校准和 valid 阈值。

模型选择优先级：

1. valid 单窗 accuracy；
2. precision 与 FP；
3. recall 与 FN；
4. group-CV 稳定性；
5. 节点数、特征成本和跨 mode 鲁棒性。

### 4. 错误样本和 hard-negative 分析

使用 s05/s06 导出的窗口错误明细，按 sample、H5、mode、时间位置、质量、OOD、三光区支持度
和环境光分层。Hard negative 只从 train 发现和回灌；valid 用于模型/阈值选择，test 只做最终
独立验收。

优先处理：

- 非佩戴但周期性或绿光相关性较高的 FP；
- 单边翘起、一个光区受外光影响但仍佩戴的 FN；
- 三光区只有一处强信号的伪佩戴；
- 强运动、桌面遮挡、弱绿光和强环境光；
- 特定 mode、设备或采集批次集中错误。

### 5. 后处理独立优化

在单窗模型固化后再评估 EMA、中值/投票、迟滞阈值、K-on/K-off、冷却时间和状态机。
后处理只使用 valid 选参，再在 test replay；基础报告始终保留未经后处理的 XGBoost 指标，
显式启用后处理时另行报告后处理结果，二者均不与任何阈值门控融合。

## 部署产物

Python 最小部署文件：

```text
artifacts/deploy_feature_extractor.py
artifacts/final_model.json
```

特征脚本自包含人工选择后的顺序、fill/clip、阈值和特征引擎，不依赖项目源码。NumPy、SciPy
和 XGBoost 属于 Python 运行环境依赖。其他 catalog、cookbook 和 golden-vector 文件用于审计、
固件移植及数值对齐，不是 Python 单窗推理的必需项目文件。

## 验证命令

```powershell
python -m py_compile s03_extract_feature_pool.py s05_train_final_model.py s06_deploy_eval.py s07_postprocess_optimize.py s08_run_pipeline.py
python -m pytest test_deploy_feature_extractor.py test_prewindowed_h5.py test_end_to_end_pipeline_guard.py -q
python -m pytest -q
```

真实数据验收必须额外运行完整训练、valid 模型选择、test 独立评估和两文件隔离部署推理。
