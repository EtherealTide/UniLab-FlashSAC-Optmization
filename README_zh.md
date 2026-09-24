# FlashSAC 优化实验（中文入口）

本项目针对 `unilab_rl` 的 FlashSAC 在 RTX 4090 上完成了问题分析、实现 patch、数值/显存
验证和性能复现。默认推荐开启完整目标编译：

```yaml
use_compile: true
compile_full_objectives: true
use_cuda_graph_critic: false
use_cuda_graph_actor: false
```

三次运行、150 个稳态 round 样本的结果为 **4.873 ms mean、4.916 ms median、5.368 ms
p90、5.764 ms p95**。deferred eager 为 13.221/13.201/13.724/13.863 ms；因此平均
耗时下降 63.1%。固定 batch 的峰值 hybrid 档为 4.455/4.289/4.957/5.000 ms（mean/
median/p90/p95）。

建议按下面顺序阅读：

1. [中文详细优化报告](FLASH_SAC_OPTIMIZATION.md)：原因、方案、逐次运行表、三次池化统计和限制。
2. [REPORT.md](REPORT.md)：FastSAC 风格的短版报告。
3. [详细复现手册](REPRODUCE.md)：环境、测试、三次采样、统计、patch 和排错命令。
4. [RTX 4090 数据明细](reports/gpu_benchmark_2026-09-24.md)：原始统计口径和结果表。

性能脚本会保存每个 round 样本，并输出 mean、median、p90、p95；三次汇总不是只对三个
均值再取平均，而是对 150 个稳态样本池化，因此尾延迟统计更有代表性。结果是 learner
microbenchmark，不包含环境、collector、IPC、replay 采样和 H2D。
