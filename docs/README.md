# 文档说明

项目当前运行契约以根目录 `README.md`、`FEATURE_INTERPRETABILITY_GUIDE.md` 和
`SINGLE_WINDOW_98_FEATURE_OPTIMIZATION_PLAN.md` 为准。

`designs/` 与 `plans/` 中按日期保存的文件是设计演进记录，不是可直接执行的操作手册。
其中早期记录可能描述已经删除的阈值门控或旧步骤编号；这些内容只用于追溯历史决策，
不得覆盖当前“全量窗口、纯 XGBoost 训练与评估、可选独立后处理”的运行契约。

当前活动流水线为：

```text
s01 数据划分 → s03 全量特征池 → s04 排序/人工 CSV →
s05 XGBoost 搜参训练 → s06 纯模型评估与部署导出
```

`s07` 后处理优化默认关闭，仅在显式启用时独立运行；它不会与任何 IR DC/ACDC
阈值门控融合。
