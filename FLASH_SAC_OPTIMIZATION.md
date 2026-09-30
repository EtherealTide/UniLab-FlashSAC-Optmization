# UniLab FlashSAC 默认配置优化报告

## 1. 结论

本轮在 RTX 4090 上重新按 owner config 默认参数测量：

```text
2048 env / batch 8192 / updates_per_step 8 / policy_frequency 4
```

一个 iteration 包含 8 次 critic、2 次 actor、2 次 temperature 和 8 次 target soft update。正式 1000 iter 训练的最后 500 iter 中，`learner_train_ms` 为：

```text
mean 55.886 ms, median 55.898 ms, p90 56.488 ms, p95 56.653 ms
```

因此 `<20 ms` 目标在不改变 batch、update 数、网络和优化器语义的前提下**未达成**。本报告不使用其他配置的 24 ms 日志代替默认参数结果。

## 2. 为什么历史 24 ms 不是这个基准

历史日志的实际配置是：

```text
4096 env / batch 2048 / updates 8 / policy_frequency 2
```

它每 iter 是 8 critic + 4 actor，4999 个稳态样本的 learner median 为 24.430 ms。本轮虽然 actor 少两次，但 critic batch 从 2048 增加到 8192，每次 critic 处理 4 倍数据。24 ms 不能作为当前默认配置基线。

## 3. 当前生产路径

最新 `unilab_rl` 在 NVIDIA CUDA 上的 FlashSAC 路径是：

```text
full-objective torch.compile
  -> Inductor/Triton 算子融合（禁用内层 cudagraph）
  -> 外层 raw CUDA Graph 捕获整个 update cycle
  -> 每 iter graph replay
```

整轮 graph 已覆盖 actor / critic / temperature optimizer、device finite guard、graph-safe LR tensor、参数归一化、target soft update 和延迟 metrics。`--no-compile` 在当前 NVIDIA 实现中也不是纯 eager，它仍然使用 whole-cycle 路径，不应当做 eager 对照。

## 4. 本轮优化

### 4.1 仅保留整轮最终 metrics

旧 whole-cycle 代码在 8 次 critic 中都构造 metric stack，但前 7 次会被覆盖；2 次 actor 也只会读取最后一次。修改后只在最后一次 critic/actor 写 buffer，但仍执行全部 optimizer、scheduler、RNG 和 target update，轮末仍只有一次 D2H。这是低风险小优化，不是 56 ms 到 20 ms 的大项。

### 4.2 基准脚本修正

脚本的四个核心参数默认值改为 `None`，只在显式传入时才生成 Hydra override。训练后从 `run_config.json` 回读实际 compose 值，再计算 update 次数。它还兼容新旧 runner timing tag，将新 runner 的秒统一换成毫秒。

### 4.3 已否决候选

尝试了“encoder 仍用 2B 保持 BatchNorm 统计，predictor 只算有用的 B 半区”：

| 候选 | learner mean | median | p90 | p95 | 结论 |
|---|---:|---:|---:|---:|---|
| critic predictor 半区 | 56.240 | 56.168 | 56.702 | 56.799 | 无收益，撤销 |
| actor + critic predictor 半区 | 57.034 | 57.122 | 57.460 | 57.570 | 约慢 1 ms，撤销 |

两组均为 100 iter 试验、统计后 50 iter，只用于快速否决，不与正式结果混合。

## 5. 正式实验数据

- GPU：NVIDIA GeForce RTX 4090（compute capability 8.9）；PyTorch 2.8.0+cu128；MuJoCo / G1 Walk Flat；seed 1。
- 1000 iter，统计最后 500 iter；正式命令没有传四个核心参数。

| 阶段 | n | mean / ms | median / ms | p90 / ms | p95 / ms |
|---|---:|---:|---:|---:|---:|
| learner | 500 | 55.886 | 55.898 | 56.488 | 56.653 |
| collector cycle | 500 | 57.804 | 57.797 | 58.602 | 58.927 |
| iteration wall | 500 | 57.550 | 57.546 | 58.192 | 58.311 |

`learner` 是一轮全部 update 的总时间，不是单次 critic update。collector 与 learner 在 double buffer 中重叠，所以不能直接相加。

## 6. 为什么 update 乘 4 但不是时间乘 4

时间可写成 `T = F_fixed + N * C_update`。metrics D2H、最终 synchronize、replay `after_tick`和 Python bookkeeping 每 iter 只付出一次，且 double buffer 使 collector 与 learner 重叠。因此 update 数增加时，固定成本不会重复。这不意味着少执行了 update；每个 iteration 仍是 8 critic + 2 actor + 2 temperature。

从以前同系统的 2/8 比例可见：`4.323 -> 13.304 ms` 是 3.08倍，而非 4 倍；但那组是 batch 256 的旧口径，不是本轮默认基准，不用于本轮验收。

## 7. 为什么当前难以再降到 20 ms

从 55.886 ms 到 20 ms 需要 64.2% 的总加速，即2.79倍。现有路径已是一次 whole-cycle CUDA Graph replay：Python dispatch、逐 update D2H、host finite check 和小 graph launch 都已移出。剩余主体是 8 次 batch-8192 critic 的矩阵、BatchNorm、categorical TD、backward 和 Adam，以及每 iter 约 107.5 MiB 的 static input copy。只优化小 kernel 不可能省下约 36 ms。

若 `<20 ms` 是硬性条件，需要允许结构性变更：减小 batch/update、减小 network/atom、合并 optimizer update（改变算法语义）、专用 categorical TD Triton kernel，或 FP8 等更激进低精度路径。

## 8. 验证和限制

Focused tests：`31 passed, 8 skipped`。正式训练完成 1000/1000 iter，指标 finite，Rich 面板和 TensorBoard 均来自生产 runner。本报告明确保留“目标未达成”，没有用微基准或不同配置替换默认训练结果。
