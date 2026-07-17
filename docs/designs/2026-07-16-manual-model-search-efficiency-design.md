# 人工特征冻结后模型搜参提效设计

## 目标与不变量

人工 CSV 冻结的特征名称、顺序和数量保持不变。正式训练仍使用 staged group-CV、默认 3 folds × 2 repeats、最大 50 棵树、节点预算、hard-negative valid 无退化验收、校准和阈值选择；优化只减少重复计算并提高可并行任务粒度。

## 方案

1. manual + balanced 默认把 Stage A 从 180 降到 120 个候选、Stage B 从 24 降到 12 个候选；fast 使用 80/12，thorough 保持 360/48。显式 CLI 参数始终优先。
2. Stage B 从“每个候选内部串行跑全部 folds”改为“候选 × fold”线程任务，完成后按候选序号、fold 序号确定性聚合；全量节点统计仍按候选并行。
3. hard-negative OOF folds 使用相同外层 worker 池并行，结果按 fold 序号重组。
4. `evaluate_accuracy_first_threshold` 使用排序、累计正负数和 `searchsorted` 一次计算 181 个阈值，保持原阈值集合和排序准则完全相同。
5. 在 `artifacts/model_search_cache/` 增加版本化 JSON 缓存。缓存键覆盖 X/y/groups 完整内容、CV 索引、XGBoost 版本、候选参数及缓存 schema；只缓存预测指标和节点数，最终模型始终重新训练。损坏或不匹配缓存自动忽略。
6. 搜参 summary 增加 Stage A、Stage B CV、Stage B full-fit、best refit、hard-negative OOF 的耗时和缓存命中统计。
7. 删除只解析/转发但从未参与搜索的 `model_search_stage1_top_k` 参数。

## 安全与失败处理

- `WL_FORCE_SERIAL=1` 继续强制单 worker。
- 每个 XGBoost fit 继续默认 `n_jobs=1`，防止 98 个外层 worker 与内部线程相乘。
- 缓存写入使用原子替换；读取失败只造成重新训练，不影响结果。
- 缓存默认启用，并提供 `--no-model_search_cache` 关闭开关。
- Stage B 任一任务异常仍使搜索失败，不静默删除候选。

## 验收

- 阈值向量化结果与逐阈值参考实现逐字段一致。
- Stage B 观察到候选 × fold 任务数，并保持候选输入顺序和 fold 数。
- 第二次相同搜索命中缓存且不调用训练；数据、CV或参数变化不复用缓存。
- hard-negative folds 可并行且 OOF 概率与串行路径一致。
- manual balanced dry-run 输出 120/12，显式参数与 thorough 档不被覆盖。
- 删除无效 CLI 后 `--help`、README与测试无残留。
- 全项目测试通过。
