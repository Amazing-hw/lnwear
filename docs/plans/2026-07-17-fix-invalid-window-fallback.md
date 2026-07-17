# 修复无效窗口伪装为 XGBoost 负预测的实施计划

**目标：** 特征提取失败的窗口不得以 `0.0` 概率参与模型聚合；全部窗口失败时必须明确 fallback 并记录原因。

**设计：** 连续流与预切窗推理统一收集逐窗异常。部分失败时只输出成功窗口及其对齐的时间、标签和 OOD 信息；全部失败时输出空概率序列，设置 `fallback=true`，并记录失败数量和首个异常。`prob_mean` 保持纯 XGBoost 语义，显式 `state_machine` 使用后处理语义标签。

## 任务 1：回归测试

- [x] 在 `test_window_error_reports.py` 增加预切窗全部失败、连续流全部失败和部分失败对齐测试。
- [x] 增加 `state_machine` 语义标签测试。
- [x] 运行聚焦测试，确认旧实现按预期失败。

## 任务 2：最小修复

- [x] 在 `s06_deploy_eval.py` 提取公共的有效窗口结果整理函数。
- [x] 两条推理路径记录异常并仅保留成功窗口。
- [x] 全部窗口失败时写入 `fallback_reason=all_window_feature_extraction_failed`，附带计数和首个错误。
- [x] 按评估方法设置 `xgboost_only_v1` 或 `xgboost_postprocess_only_v1`。

## 任务 3：注释和文档

- [x] 更新 `s06_deploy_eval.py` 顶部说明及 `s08_run_pipeline.py` 步骤元组注释。
- [x] 更新 README 的 fallback 说明，明确部分失败和全部失败行为。

## 任务 4：验证

- [x] 运行聚焦测试、主要脚本语法检查和 dry-run。
- [x] 运行 `python -m pytest -q` 全量回归。
- [x] 运行活动代码残留扫描与 `git diff --check`。
