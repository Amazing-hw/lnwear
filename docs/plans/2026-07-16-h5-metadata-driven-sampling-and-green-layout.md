# H5 元数据驱动采样率与三光区映射实施计划

**目标：** 让 `frequency` 和 `ppg_config` 成为采样率及三光区映射的唯一来源，并对无效数据输出原因统计。

**架构：** `s01` 负责从 H5 读取、校验并写入样本元数据；`s02`、`s03`、`s06` 只消费该元数据。删除名称、长度和方差推断，保留特征名 `mode` 但令其值等于 `ppg_config`。

**技术栈：** Python、h5py、NumPy、SciPy、pytest。

## 任务 1：建立失败测试

**文件：** `test_prewindowed_h5.py`、`test_interpretable_feature_pool.py`、`test_deploy_feature_extractor.py`

- 增加标准 H5 与 grouped-window H5 的 `frequency`/`ppg_config` 扫描测试。
- 增加缺失、非法、分组内不一致元数据的跳过与统计测试。
- 增加 `frequency=25/100` 的降采样行为测试。
- 增加 `ppg_config=0/1/2` 的精确三光区映射测试。
- 先运行定向测试，确认因现有自动判断逻辑而失败。

## 任务 2：实现严格元数据扫描

**文件：** `s01_data_split.py`

- 读取标量字段 `frequency`、`ppg_config`。
- 仅接受 `frequency in {25,100}` 和 `ppg_config in {0,1,2}`。
- grouped-window 数据允许从父 group 读取，或从所有窗口子 group 读取一致值。
- 将合法值写入每个 sample；非法数据按五类原因计数并打印样本级原因。

## 任务 3：统一 Stage1/Stage2 采样率

**文件：** `s02_ir_dc_threshold.py`、`s03_extract_feature_pool.py`、`s06_deploy_eval.py`

- 删除样本名和窗口长度采样率推断。
- Stage1 根据 `frequency` 降到 5 Hz。
- Stage2 在 `frequency=100` 时降到 25 Hz，在 `frequency=25` 时直接使用。
- 保持读入排序后的 `[3:-3]` 同步裁剪规则。

## 任务 4：统一三光区映射并删除自动 mode 判断

**文件：** `s03_extract_feature_pool.py`、`s06_deploy_eval.py`、`s08_run_pipeline.py`、`stage2_feature_catalog.py`

- 按已确认的三套公式由 `ppg_config` 生成固定位置 `g1/g2/g3`。
- 删除所有生产路径中的 `detect_green_mode()` 调用。
- 特征 `mode` 直接记录 `ppg_config`。
- 独立部署脚本中的配置参数语义同步为 `ppg_config`。

## 任务 5：文档与验证

**文件：** `README.md` 及相关当前设计文档

- 记录字段位置、合法值、映射公式、跳过原因和降采样规则。
- 运行定向测试、全项目测试、编译检查、错误级静态检查和 `git diff --check`。
- 扫描确认生产代码不再通过名称、长度或方差推断采样率/绿光配置。

