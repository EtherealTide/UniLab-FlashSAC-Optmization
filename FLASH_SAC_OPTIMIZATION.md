# UniLab FlashSAC 优化说明

## 优化点

1. **扩大 compile 边界**：从只编译 loss 改为编译完整 actor/critic objective，让 Inductor 跨网络前向、categorical target 和 loss 做融合。
2. **减少无效梯度**：actor 更新时冻结 critic 参数，但保留对 action 的梯度路径。
3. **减少同步**：finite guard 在 GPU 上完成，metrics 的 D2H 拷贝延迟到一轮末尾。
4. **保护 CUDA Graph 输出**：使用固定 metrics buffer，避免 replay 时临时存储复用导致 overwrite。

## 性能

RTX 4090 learner 微基准（150 个稳态 round 池化）：

| 路径 | mean | median | p90 | p95 |
|---|---:|---:|---:|---:|
| deferred eager | 13.221 | 13.201 | 13.724 | 13.863 |
| full-objective compile | 4.873 | 4.916 | 5.368 | 5.764 |
| full compile + manual Graph | 4.455 | 4.289 | 4.957 | 5.000 |

这组数据只衡量 learner，不包含真实环境、collector、IPC、replay 和 H2D。真实训练应使用新的 physics benchmark，观察完整 `iter_ms`。

## 真实训练计时工具

工具位置：`experiments/benchmark_flashsac_training.py`。

它通过子进程调用 UniLab 的 `src/unilab/scripts/train_flashsac.py`，因此 collector 会真正执行物理引擎。子进程通过 PTY 启动，Rich 能保持与直接运行训练命令相同的实时仪表盘。训练结束后解析 TensorBoard timing：

- `timing/learner_train_ms`：learner update；
- `perf/collector_cycle_ms`：collector cycle；
- `perf/iter_ms`：完整 iteration wall time；
- `reward/mean`：当前平均 reward。

输出 JSON 同时包含逐轮数据和 mean/median/p90/p95。`--skip-first N` 只影响统计样本，不删除原始 `rows_all`。

## 解释

compile 首轮可能很慢，这是 Inductor 编译和 CUDA Graph capture 的冷启动，不应与稳态混在一起。collector 和 learner 在 double-buffer 中并行/重叠，所以总轮时间通常小于两者相加。正式比较至少应固定 backend、task、num_envs、batch、updates_per_step，并重复多次。
