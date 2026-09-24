# UniLab FlashSAC 单卡性能优化报告

## 结论

在 RTX 4090、PyTorch 2.8.0+cu128、BF16、batch 2048 的 learner 稳态微基准中，三次
运行各取 50 个 round 样本并池化：

| 路径 | mean | median | p90 | p95 |
| --- | ---: | ---: | ---: | ---: |
| eager deferred metrics | 13.221 | 13.201 | 13.724 | 13.863 |
| manual CUDA Graph | 10.661 | 10.840 | 11.119 | 11.160 |
| full-objective compile | **4.873** | **4.916** | **5.368** | **5.764** |
| full compile + manual Graph | **4.455** | **4.289** | **4.957** | **5.000** |

单位为 ms/round。相对 deferred eager，full compile 平均降低 63.1%，hybrid 平均降低
66.3%。默认推荐 full-objective compile；hybrid 作为固定 batch 的峰值性能档。

## 三次 run 的均值

| 路径 | run 1 | run 2 | run 3 | run mean average ± stdev |
| --- | ---: | ---: | ---: | ---: |
| eager deferred | 13.143 | 13.056 | 13.465 | 13.221 ± 0.216 |
| manual CUDA Graph | 10.595 | 10.661 | 10.728 | 10.661 ± 0.066 |
| full-objective compile | 5.083 | 4.756 | 4.779 | 4.873 ± 0.182 |
| full compile + manual Graph | 4.438 | 4.423 | 4.503 | 4.455 ± 0.042 |

完整逐次 mean/median/p90/p95 表、优化原因、限制和正确性证据见
[FLASH_SAC_OPTIMIZATION.md](FLASH_SAC_OPTIMIZATION.md)；可复现实验命令见
[REPRODUCE.md](REPRODUCE.md)。

## 关键修复

- metrics D2H 延迟到 cycle 末尾；
- finite guard 留在 GPU；
- actor 更新冻结 critic 参数但保留 `dQ/da`；
- categorical TD projection 改为纯 tensor，解决 capture 失败；
- 首次手工 Graph capture 后立即 replay；
- full compile + 外层 Graph 使用固定 metric buffer，避免 overwrite；
- 将 compile 边界扩展到完整 actor/critic objective。

## 验证

聚焦测试 `68 passed`，全量测试 `413 passed, 8 skipped, 3 deselected`；Ruff、mypy、
pyright 通过；10 步 BF16 数值对照和 1000 replay 显存压力测试通过。该报告是 learner
microbenchmark，不声称端到端机器人 reward/FPS 已完成验收。
