# UniLab FlashSAC 优化报告

## 1. 范围与结论

本次优化针对 FlashSAC learner 和 off-policy runner 的训练热路径，不改变网络结构、loss 定义、optimizer 数量、update 次数或采样语义。核心方案是扩大 `torch.compile` 边界，并消除 update 循环中的无效梯度、同步和逐次 metrics D2H。

生产默认推荐：

```yaml
use_compile: true
compile_full_objectives: true
use_cuda_graph_critic: false
use_cuda_graph_actor: false
```

真实 MuJoCo 训练中：

- 1 actor / 2 critic：learner mean 从 10.118 ms 降至 4.323 ms，降低 57.3%；完整 iter mean 从 11.009 ms 降至 5.341 ms。
- 4 actor / 8 critic：learner mean 从 36.885 ms 降至 13.304 ms，降低 63.9%；完整 iter mean 从 37.760 ms 降至 14.235 ms。

## 2. update 和计时口径

`updates_per_step` 表示一个 runner iteration 中的 critic update 次数。actor 在 `update_idx % policy_frequency == 0` 时更新；temperature optimizer 跟随 actor 一起更新。本报告固定 `policy_frequency=2`：

| 名称 | critic | actor | temperature | target soft update |
|---|---:|---:|---:|---:|
| 1 actor / 2 critic | 2 | 1 | 1 | 2 |
| 4 actor / 8 critic | 8 | 4 | 4 | 8 |

`learner_train_ms` 是一个 iter 内上述全部 learner update、末尾 metrics 读取和最终 device synchronize 的总时间，不是单次 update。`collector_cycle_ms` 是 collector 一个周期的时间，含等待 learner inference 的影响。`iter_ms` 是完整 runner iteration 墙钟时间。

double-buffer runner 会让 collector 和 learner 重叠，所以：

```text
iter_ms != learner_train_ms + collector_cycle_ms
```

## 3. 具体优化

### 3.1 编译完整 objective

旧路径只编译较小的 loss 函数，optimizer 前后的网络调用、categorical target、tensor 变换和 loss 之间仍有大量独立 kernel。新路径编译 `_critic_objective_tensors` 和 `_actor_objective_tensors`，允许 Inductor 跨网络前向、分布式 critic target 和 loss 生成做更大范围融合。

它主要减少：

- Python dispatch；
- 小 kernel launch；
- 中间 tensor materialization；
- loss 边界两侧无法融合的算子。

### 3.2 actor 更新时冻结 critic 参数

actor loss 需要 critic 对 action 的梯度，但不需要 critic 参数梯度。旧路径会为 critic 参数构造和累积无用梯度。新路径在 actor objective 中临时将 critic 参数设为不求梯度，同时保留：

```text
actor -> action -> critic(action) -> actor loss
```

因此 actor 梯度保持正确，critic 参数的无效 autograd 和显存写入被移除。

### 3.3 finite guard 留在 GPU

逐 update 执行 `isfinite(...).item()` 或 Python truth check 会迫使 CPU 等待 GPU。新路径使用 device-side `found_inf`/grad-scale tensor，让 optimizer gating 留在 CUDA 上，消除热路径中的 host synchronization。

### 3.4 metrics 延迟 D2H

loss、entropy、temperature 等指标不参与下一个 update 的控制流，无需每次都复制到 CPU。新路径把指标写入固定 GPU buffer，只在一轮最后读取：

- critic metrics：只读取最后一次 critic update；
- actor metrics：一轮结束统一读取；
- D2H 和同步次数从“随 update 数增长”变为“每 iter 一次”。

这也是 8-update 没有严格变成 2-update 四倍耗时的重要原因。

### 3.5 CUDA Graph 输出安全

CUDA Graph replay 会复用静态存储。如果直接返回 capture 内临时 tensor，下一次 replay 可能覆盖上一轮还在使用的 metrics。新路径为 critic 和 actor 分配固定 metric buffers，并在 capture 内 `copy_`，读取端只在约定时间消费，从而避免显存 overwrite 和指标串轮。

### 3.6 Inductor Graph 与手工 Graph 的边界

纯 full-objective compile 路径允许 Inductor CUDA Graph Trees，以减少重复 host launch。外层手工 CUDA Graph 已经负责 capture/replay 时，则关闭 Inductor 内部 Graph，避免嵌套 capture；Inductor 在这种路径中只负责算子融合。

手工 Graph 还要求固定输入 shape、固定 staging buffer 和稳定 optimizer 状态，生命周期更复杂。因此它保留为可选峰值路径，不作为默认生产配置。

## 4. 各路径含义

| 路径 | deferred metrics | compile | full objective | 手工 CUDA Graph | 含义 |
|---|---:|---:|---:|---:|---|
| eager sync | 否 | 否 | 否 | 否 | 每次 update 立即读取指标，作为最原始基线 |
| deferred eager | 是 | 否 | 否 | 否 | 不融合算子，只延迟 metrics D2H |
| loss-only compile | 是 | 是 | 否 | 否 | 只编译旧的局部 loss 边界 |
| full-objective compile | 是 | 是 | 是 | 否 | 编译完整 actor/critic objective，推荐默认路径 |
| manual Graph | 是 | 否 | 否 | 是 | 手工 capture 完整 update，但没有 Inductor objective 融合 |
| loss compile + manual Graph | 是 | 是 | 否 | 是 | 局部 loss 融合加外层 Graph |
| full compile + manual Graph | 是 | 是 | 是 | 是 | 完整 objective 融合加外层 Graph，固定 shape 下的峰值路径 |

“compile + manual Graph”不是两个 Graph 嵌套：外层手工 Graph 负责 capture，Inductor 内部 Graph 被关闭。

## 5. 实验环境与方法

### 5.1 Learner 微基准

- GPU：NVIDIA GeForce RTX 4090；
- PyTorch：2.8.0+cu128；
- dtype：BF16 AMP；
- matmul precision：`highest`；
- batch：2048；
- actor：hidden 128，2 blocks；
- critic：hidden 256，2 blocks，101 atoms；
- obs / critic obs / action：98 / 101 / 29；
- 每条路径 warmup 10 round，计时 50 round；
- 每种 update 口径独立运行 3 次；
- 表中统计池化三个 run 的 150 个稳态样本。

每个 microbenchmark round 对应完整 learner 更新组合，并在末尾执行 CUDA synchronize。

### 5.2 真实物理训练

- GPU：NVIDIA GeForce RTX 4090；
- task/backend：G1 Walk Flat / MuJoCo；
- num envs：256；
- batch：256；
- replay buffer N：32；
- learning starts：8；
- 总训练：1000 iter；
- 统计窗口：最后 500 iter；
- seed：1；
- compile 路径：full-objective compile，未启用手工 Graph。

2-update 的 deferred eager 和 compile 各有 500 个统计样本。8-update deferred eager 有 500 个样本；8-update compile 使用两次独立 1000-iter 训练的最后 500 iter，合计池化 1000 个样本。真实训练的 `--no-compile` 对照仍保留 deferred metrics、critic 冻结等非 compile 优化，用于隔离 full-objective compile 的增益；它不是未打任何补丁的历史版本。

## 6. Learner 微基准详细数据

### 6.1 1 actor / 2 critic

单位为 ms/round，`n=150`：

| 路径 | mean | median | p90 | p95 |
|---|---:|---:|---:|---:|
| eager sync | 13.821 | 13.833 | 14.342 | 14.532 |
| deferred eager | 13.221 | 13.201 | 13.724 | 13.863 |
| loss-only compile | 12.008 | 12.009 | 12.357 | 12.574 |
| full-objective compile | **4.873** | **4.916** | **5.368** | **5.764** |
| manual Graph | 10.661 | 10.840 | 11.119 | 11.160 |
| loss compile + manual Graph | 10.635 | 10.669 | 11.105 | 11.173 |
| full compile + manual Graph | **4.455** | **4.289** | **4.957** | **5.000** |

full-objective compile 相对 deferred eager：平均耗时降低 63.1%，加速 2.71 倍。叠加手工 Graph 后降低 66.3%，加速 2.97 倍。

### 6.2 4 actor / 8 critic

单位为 ms/round，`n=150`：

| 路径 | mean | median | p90 | p95 |
|---|---:|---:|---:|---:|
| eager sync | 52.818 | 52.968 | 53.743 | 53.892 |
| deferred eager | 49.960 | 50.350 | 51.003 | 51.133 |
| loss-only compile | 43.850 | 43.919 | 44.119 | 44.380 |
| full-objective compile | **17.472** | **17.475** | **17.513** | **17.519** |
| manual Graph | 40.168 | 40.145 | 40.255 | 40.310 |
| loss compile + manual Graph | 40.077 | 40.038 | 40.157 | 40.309 |
| full compile + manual Graph | **16.706** | **16.676** | **16.711** | **16.792** |

full-objective compile 相对 deferred eager：平均耗时降低 65.0%，加速 2.86 倍。叠加手工 Graph 后降低 66.6%，加速 2.99 倍。

## 7. 真实 MuJoCo 训练详细数据

### 7.1 Learner 时间

单位为 ms/iter：

| update 口径 | 路径 | n | mean | median | p90 | p95 |
|---|---|---:|---:|---:|---:|---:|
| 1 actor / 2 critic | deferred eager | 500 | 10.118 | 9.965 | 10.760 | 11.092 |
| 1 actor / 2 critic | full-objective compile | 500 | **4.323** | **4.397** | **4.874** | **5.088** |
| 4 actor / 8 critic | deferred eager | 500 | 36.885 | 36.183 | 39.010 | 40.171 |
| 4 actor / 8 critic | full-objective compile | 1000 | **13.304** | **13.145** | **14.344** | **14.888** |

### 7.2 Collector cycle

单位为 ms/cycle：

| update 口径 | 路径 | n | mean | median | p90 | p95 |
|---|---|---:|---:|---:|---:|---:|
| 1 actor / 2 critic | deferred eager | 500 | 12.144 | 11.990 | 12.827 | 13.347 |
| 1 actor / 2 critic | full-objective compile | 500 | **6.839** | **6.776** | **7.598** | **7.922** |
| 4 actor / 8 critic | deferred eager | 500 | 38.632 | 38.097 | 40.875 | 42.275 |
| 4 actor / 8 critic | full-objective compile | 1000 | **15.323** | **15.077** | **16.544** | **17.207** |

collector 本身没有执行 optimizer update，但它会等待 learner inference；learner 更快后，collector cycle 的等待时间也会降低。

### 7.3 完整 iteration 墙钟时间

单位为 ms/iter：

| update 口径 | 路径 | n | mean | median | p90 | p95 |
|---|---|---:|---:|---:|---:|---:|
| 1 actor / 2 critic | deferred eager | 500 | 11.009 | 10.850 | 11.681 | 12.032 |
| 1 actor / 2 critic | full-objective compile | 500 | **5.341** | **5.391** | **5.908** | **6.213** |
| 4 actor / 8 critic | deferred eager | 500 | 37.760 | 37.060 | 39.935 | 41.041 |
| 4 actor / 8 critic | full-objective compile | 1000 | **14.235** | **14.064** | **15.278** | **15.886** |

端到端 iter mean 分别降低 51.5% 和 62.3%。

## 8. 为什么 update 数乘 4，时间只有约 3 倍

从 1 actor / 2 critic 变成 4 actor / 8 critic 后，critic、actor、temperature 和 target update 的数量都严格增加到 4 倍；算法没有少做 update。但一次 iteration 的时间可以近似写成：

```text
T(iter) = F_fixed + N × C_update
```

`F_fixed` 是每 iter 只执行一次的固定成本，`N` 才随 update 数增长。真实训练优化后：

```text
T2 = 4.323 ms
T8 = 13.304 ms
T8 / T2 = 3.08
```

用 `T2 = F + 2C`、`T8 = F + 8C` 估算：

```text
F ≈ (4 × T2 - T8) / 3 ≈ 1.329 ms/iter
```

也就是说，2-update 情况下约 1.33 ms 是不会随 update 数乘 4 的固定成本。其来源主要是：

1. **metrics 只读一次**：8 次 critic 和 4 次 actor 不会做 12 次 D2H；固定 buffer 在末尾统一读取。
2. **只做一次最终同步**：update 内核先异步提交，整轮结束才执行一次 CUDA stream synchronize。
3. **runner 固定工作只做一次**：deferred metrics 汇总、replay pipeline `after_tick`、scheduler/版本发布和 Python 侧 bookkeeping 不按 update 数翻倍。
4. **GPU 利用率改善**：更多连续 update 能摊薄 CPU launch 和调度空隙，使 GPU 保持更连续的工作队列。
5. **double buffer 重叠**：端到端 iter 中 collector、replay 和 learner 会重叠，墙钟时间不会按各阶段简单相加。

对照数据也支持这一解释：

| 口径 | 2-update mean | 8-update mean | 倍率 |
|---|---:|---:|---:|
| 真实训练 deferred eager learner | 10.118 | 36.885 | 3.65× |
| 真实训练 full-compile learner | 4.323 | 13.304 | **3.08×** |
| 微基准 full-objective compile | 4.873 | 17.472 | 3.59× |
| 真实训练 full-compile iter wall | 5.341 | 14.235 | 2.67× |

优化后的单次 update 计算更短，固定成本在 2-update 总时间中占比更高，因此从 2 增至 8 时总时间倍率反而更接近 3，而不是 4。这是固定开销被摊薄的结果，不代表 8-update 少执行了训练步骤。

## 9. 选择建议与限制

- 默认选择 full-objective compile：速度接近最优，shape 和 graph 生命周期约束较少。
- 固定 batch/shape 且追求最后 4%–9% learner 性能时，可评估 full compile + manual Graph。
- 不建议只开手工 Graph：没有完整 objective 融合时，8-update mean 仍为 40.168 ms，明显慢于 full-objective compile 的 17.472 ms。
- compile 首轮包含 Inductor 编译和 graph capture 冷启动，必须与稳态窗口分开。
- 微基准只隔离 learner；生产判断应优先看真实物理训练的 `iter_ms`。
- 本报告不包含另行研究的 Triton categorical-target kernel。

## 10. 数据文件

- 2-update 微基准：`results/production_bf16_samples_run{1,2,3}.json`；
- 2-update 聚合：`results/production_bf16_samples_summary.json`；
- 8-update 微基准：`results/production_bf16_u8_samples_run{1,2,3}.json`；
- 8-update 聚合：`results/production_bf16_u8_samples_summary.json`；
- 真实训练对比摘要：`results/physical_training_comparison.json`。
