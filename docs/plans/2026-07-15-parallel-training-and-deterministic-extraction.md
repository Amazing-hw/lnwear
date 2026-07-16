# 并行训练与确定性特征提取实施计划

**目标：** 并行执行 s05 独立模型候选，并让 s03 并行输出顺序确定，同时保持人工特征和模型选择语义不变。

**架构：** s05 使用共享只读矩阵的有序线程映射，候选内 XGBoost 固定单线程；s03 使用按样本索引收集再展开。s08 统一传递 worker 数，README 说明实际并行边界和串行回退条件。

**技术栈：** Python、`concurrent.futures.ThreadPoolExecutor`、`ProcessPoolExecutor`、XGBoost、pytest。

---

### 任务 1：锁定并发契约

**文件：**
- 修改：`test_model_search_config.py`
- 修改：`test_prewindowed_h5.py`

- [ ] 增加 s05 有序并行映射测试，模拟任务逆序完成并断言输出仍按输入索引排列。
- [ ] 增加 `WL_FORCE_SERIAL=1` 的 worker 解析测试。
- [ ] 增加 s03 逆序完成测试，断言输出 `sample_name` 顺序等于输入顺序。
- [ ] 运行三个新测试并确认分别因缺少新接口或旧完成顺序而失败。

### 任务 2：实现 s05 外层候选并行

**文件：**
- 修改：`s05_train_final_model.py`
- 修改：`s08_run_pipeline.py`

- [ ] 增加 `resolve_model_search_workers` 和 `ordered_thread_map`，串行与并行共享相同返回顺序和异常语义。
- [ ] 将 single-split Stage A 候选循环改为有序候选任务，选定后在主线程重训最佳模型。
- [ ] 将 staged group-CV Stage A 和 Stage B 候选循环改为有序候选任务，Stage B 每个候选内部顺序完成 folds。
- [ ] 增加 `--model_search_n_workers`，s08 默认把全局 `--n_workers` 传给 s05。
- [ ] 运行 s05 定向测试并确认通过。

### 任务 3：实现 s03 确定性重组

**文件：**
- 修改：`s03_extract_feature_pool.py`

- [ ] 预分配 `rows_by_sample`，worker 完成后按原始索引存储。
- [ ] 全部 worker 完成后按照输入样本顺序展开行列表。
- [ ] 保留现有单样本异常降级、进度输出和串行路径。
- [ ] 运行 s03 定向测试并确认通过。

### 任务 4：同步文档与注释

**文件：**
- 修改：`README.md`
- 修改：`s05_train_final_model.py`
- 修改：`s03_extract_feature_pool.py`

- [ ] 说明业务语义并行、批处理多进程和模型候选线程并行的区别。
- [ ] 记录默认 worker、串行回退、环境变量和嵌套线程限制。
- [ ] 明确 manual 特征数量固定但模型参数仍并行搜索。
- [ ] 检查新增注释不声称 Stage1/Stage2 在单样本内使用两个 CPU 线程。

### 任务 5：验收

**文件：**
- 验证：全部 Python、测试和 README 变更

- [ ] 运行 `python -m py_compile` 覆盖全部主脚本。
- [ ] 运行并行、模型搜索、预窗口、人工 CSV 和端到端定向测试。
- [ ] 运行完整 pytest；若环境时间限制中断，记录已完成数量并继续分组运行直到覆盖全部测试。
- [ ] 运行 `git diff --check`、占位符扫描和 README/CLI 一致性扫描。
- [ ] 自审候选排序、异常传播、内存占用和旧调用兼容性。

