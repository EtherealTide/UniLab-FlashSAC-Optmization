# FlashSAC RTX 4090 Optimization v1.0.0

本版本发布 FlashSAC 的 full-objective compile 和可选 hybrid CUDA Graph 优化。

## 推荐配置

默认使用 `use_compile=true`、`compile_full_objectives=true`，并保持 actor/critic 手工
Graph 关闭。固定 batch、追求峰值吞吐时可开启 hybrid 的两个 Graph 选项。

## 性能（RTX 4090）

三次运行、每次 50 个稳态 round、共 150 个样本：

| 模式 | mean | median | p90 | p95 |
| --- | ---: | ---: | ---: | ---: |
| deferred eager | 13.221 | 13.201 | 13.724 | 13.863 |
| full-objective compile | 4.873 | 4.916 | 5.368 | 5.764 |
| full compile + manual Graph | 4.455 | 4.289 | 4.957 | 5.000 |

单位为 ms/learner round；这是 learner microbenchmark，不是端到端机器人 FPS。

## 其他内容

- categorical TD projection 已移除 capture 期间的 Python scalar/D2H；
- metrics D2H 延迟、GPU finite gate、actor critic-gradient 冻结已实现；
- 首次 Graph capture replay、固定 metric buffer 和 target update 生命周期已修复；
- 包含 10 步数值验证、1000 replay 显存稳定性结果、patch 和完整复现手册。

详细说明见 `REPORT.md`、`FLASH_SAC_OPTIMIZATION.md` 和 `REPRODUCE.md`。
