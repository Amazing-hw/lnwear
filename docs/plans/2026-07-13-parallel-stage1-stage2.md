# Stage1 与 Stage2 并行执行实施计划

**Goal:** 让 Stage2 在训练、评估和部署中始终处理全量合法窗口，Stage1 仅屏蔽最终对外输出。

**Architecture:** 保留 Stage1 的独立流式门控状态，同时让 Stage2 特征、概率和后处理状态在
所有窗口持续更新。评估明确拆分 Stage1、独立 Stage2、最终融合三种指标口径，并保留旧缓存
字段的读取兼容。

**Tech Stack:** Python、NumPy、pandas、scikit-learn/XGBoost、pytest、matplotlib PNG。

---

### Task 1: 全量 Stage2 特征池

**Files:** `s03_extract_feature_pool.py`、`test_fixed_stage1_threshold.py`、
`test_prewindowed_h5.py`

- [x] 添加失败测试：Stage1 阈值设置为不可能通过时，连续和预切窗输入仍输出全部合法 Stage2 窗口。
- [x] 运行聚焦测试，确认旧实现因返回空行或少行而失败。
- [x] 删除 `_extract_rows_for_sample` 中两个 Stage1 提前返回/continue，不改变窗口合法性检查。
- [x] 更新函数注释，明确 Stage1 阈值参数仅为兼容调用，不参与 Stage2 数据过滤。
- [x] 运行聚焦测试，确认特征池窗口数与 Stage1 阈值无关。

### Task 2: 持续 Stage2 推理

**Files:** `s06_deploy_eval.py`、`test_end_to_end_pipeline_guard.py`、
`test_prewindowed_h5.py`

- [x] 添加失败测试：逐窗 Stage1 全关闭时，`window_probs`、`window_preds`、质量信息和窗口时间轴仍完整。
- [x] 添加失败测试：连续流中 Stage1 关闭窗口不再向 Stage2 写入 `None` 或零概率占位。
- [x] 修改 `_infer_prewindowed_sample` 和 `_infer_one_sample`，始终提取 Stage2 特征并预测；Stage1 只记录 gate flag。
- [x] 同时输出 `stage1_gate_flags`，并将 `stage2_enabled_flags` 保留为兼容别名。
- [x] 运行推理与缓存聚焦测试。

### Task 3: 独立后处理与融合输出

**Files:** `s06_deploy_eval.py`、`s07_postprocess_optimize.py`、
`test_s07_postprocess_optimize.py`、`test_window_error_reports.py`

- [x] 添加失败测试：Stage1 关闭期间 EMA/状态机仍持续演进，打开门控的第一个窗口可立即输出已有 Stage2 状态。
- [x] 将后处理应用于完整概率流，保存独立 `stage2_states`。
- [x] 单独生成 `output_states = stage2_states AND stage1_gate_flags`，禁止用门控重置 Stage2 状态。
- [x] 让 `s07` 在全量概率流上搜索并额外报告融合输出指标。
- [x] 运行后处理、时延与 false-worn 聚焦测试。

### Task 4: 三套指标口径

**Files:** `s06_deploy_eval.py`、`s08_run_pipeline.py`、`pipeline_acceptance.py`、
`test_pipeline_acceptance.py`、`test_window_error_reports.py`

- [x] 添加失败测试：独立 Stage2 window metrics 包含 Stage1 关闭窗口。
- [x] 添加失败测试：融合输出在 Stage1 关闭窗口恒为 0，但独立 Stage2 预测保持原值。
- [x] 重构指标汇总，分别写出 `stage1_only`、`stage2_independent`、`fused_output`。
- [x] 更新 CSV、错误分层和 PNG 报告标签，避免把 gate flag 描述成 Stage2 未执行。
- [x] 更新 acceptance，校验三套指标和并行语义版本。

### Task 5: 商业八特征基线并行化

**Files:** `s09_commercial_compare.py`、`test_s09_grouped_label_guard.py`、
`test_model_search_config.py`

- [x] 添加失败测试：商业基线训练收集器包含 Stage1 关闭的合法窗口。
- [x] 修改商业特征提取和训练收集器，始终计算八特征；Stage1 flag 只参与融合输出。
- [x] 让商业基线后处理持续更新，并分开报告独立与融合指标。
- [x] 运行商业比较聚焦测试。

### Task 6: 缓存、部署说明和文档

**Files:** `s06_deploy_eval.py`、`s08_run_pipeline.py`、`README.md`、
`test_window_error_reports.py`、`test_pipeline_acceptance.py`

- [x] 添加失败测试：新 NPZ 同时含 `stage1_gate` 和兼容字段，二者完全一致。
- [x] 更新缓存 schema/manifest 和部署 cookbook，声明 Stage2 持续执行、Stage1 只屏蔽输出。
- [x] 更新 README 流程、指标、产物和运行说明；所有图表格式保持 PNG-only。
- [x] 扫描活动代码/文档，删除“Stage1 通过后才计算 Stage2”或“Stage2 enabled”旧语义。

### Task 7: 完整验收

**Files:** 全部生产 Python、测试与活动文档

- [x] 运行 Stage2 全量数据、持续推理、后处理、缓存、商业比较聚焦测试。
- [x] 运行完整 `pytest`。
- [x] 运行生产文件 `py_compile` 和 Pylint errors-only。
- [x] 运行 manual/auto/auto-E2E dry-run，核对命令与 PNG-only 输出。
- [x] 运行 `git diff --check`，扫描旧门控语义和旧字段误用。
- [x] 不提交或推送 Git；保留当前工作区修改供用户检查。
