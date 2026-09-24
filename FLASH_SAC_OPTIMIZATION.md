# FlashSAC RTX 4090 优化报告

## 结论先说

在 RTX 4090、BF16、batch 2048 的 FlashSAC learner 微基准中，最值得合入生产的方案是：

```yaml
use_compile: true
compile_full_objectives: true
use_cuda_graph_critic: false
use_cuda_graph_actor: false
```

三次运行、每次 50 个稳态 round（共 150 个样本）池化后，纯 full-objective compile 为
**4.873 ms/round（median 4.916，p90 5.368，p95 5.764）**；deferred eager 为
**13.221 ms/round**，即平均耗时降低 63.1%，吞吐约 2.71 倍。

如果 batch 形状固定、愿意承担 CUDA Graph 生命周期维护成本，可以使用峰值档：

```yaml
use_compile: true
compile_full_objectives: true
use_cuda_graph_critic: true
use_cuda_graph_actor: true
use_cuda_graph_critic_packed_staging: true
use_cuda_graph_actor_packed_staging: true
```

full compile 外再套手工 Graph 的池化均值为 **4.455 ms/round**（median 4.289，p90
4.957，p95 5.000），相对纯 full compile 的均值再快约 8.6%。这是可选的峰值性能档，
默认仍建议纯 full compile，因为它更简单、更容易随 PyTorch/CUDA 升级维护。

## 1. 为什么 FlashSAC 也有同类问题

代码审查和 capture 测试确认了以下问题：

1. metrics 在每次 update 中 `.item()/float()/cpu()`，使 CPU 等待 GPU；把 metrics 延迟到
   cycle 末尾一次读取，可以减少同步。
2. `bool(torch.isfinite(loss))` 是 host-visible 分支；finite gate 改为 GPU 上的
   found-inf/grad-scale gate 后，capture 和 compile 都不会被同步点打断。
3. actor 只需要 critic 对 action 的梯度，不需要 critic 参数梯度。actor 更新期间临时
   冻结 critic 参数，仍保留 `dQ/da`，减少无用计算和显存写入。
4. categorical TD projection 原先把 CUDA tensor 转成 Python scalar（`float(support.min())`、
   `float(support.max())`），capture 时会触发 stream-capturing 错误。现在全部使用 tensor
   的 `amin/amax/clamp`，不再访问 host。
5. 手工 Graph 首次 capture 只记录了图，却没有 replay，第一次 optimizer update 会被静默
   丢弃。修复后 capture 完立即 replay 一次。
6. full compile + 外层 Graph 的 metrics 需要在 capture 前预分配固定 buffer，并在图内
   `copy_`；否则后续 backward/optimizer 可能复用临时地址，表现为 metrics 为零或被覆盖。

## 2. 为什么之前“Inductor 不如手工 Graph”

旧路径只 compile loss 尾部，网络前向、ensemble critic、categorical projection 和张量
拼接仍是 eager 小 kernel。手工 Graph 虽然没有算子融合，但能把整段 update 的 launch 固定
下来，因此当时更快。

本次把编译边界扩大到完整 objective：target actor、target/online critic、categorical
projection、critic loss，以及 actor forward、critic forward、actor loss。这样 Inductor
有足够大的区域做融合、内存规划和自身 cudagraph replay，结果从“手工 Graph 胜出”变为
“full compile 明显胜出”。

## 3. 实验设置

| 项目 | 设置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090，49140 MiB |
| Driver / PyTorch | 595.84 / 2.8.0+cu128 |
| AMP | BF16（`use_amp=true`, `amp_dtype=auto`） |
| batch | 2048 |
| G1-like 维度 | obs/critic obs/action = 98/101/29 |
| 网络 | actor 128×2，critic 256×2，101 atoms |
| 每 round | 2 critic updates + 1 actor update |
| warmup / 测量 | 10 / 50 rounds |
| 重复 | 3 次；每次 50 样本，池化共 150 样本 |

warmup 排除了首次 Inductor 编译和 CUDA Graph capture cold start。该基准只测 learner
update，batch 已在 GPU，不包含环境、collector、IPC、replay 采样和 H2D。

## 4. 每次运行的完整统计

下表是每个 run 的 50 个稳态 round 的统计；p90/p95 使用线性插值百分位数。

| 模式 | run | mean | median | p90 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager immediate | 1 | 13.496 | 13.529 | 13.928 | 14.222 |
| eager immediate | 2 | 14.053 | 14.020 | 14.488 | 14.653 |
| eager immediate | 3 | 13.913 | 13.914 | 14.354 | 14.525 |
| eager deferred | 1 | 13.143 | 13.142 | 13.653 | 13.784 |
| eager deferred | 2 | 13.056 | 13.022 | 13.507 | 13.650 |
| eager deferred | 3 | 13.465 | 13.384 | 13.888 | 14.280 |
| loss-only compile | 1 | 12.072 | 12.034 | 12.446 | 12.885 |
| loss-only compile | 2 | 12.011 | 12.028 | 12.471 | 12.574 |
| loss-only compile | 3 | 11.942 | 11.995 | 12.255 | 12.339 |
| manual CUDA Graph | 1 | 10.595 | 10.881 | 10.984 | 11.006 |
| manual CUDA Graph | 2 | 10.661 | 10.902 | 11.068 | 11.129 |
| manual CUDA Graph | 3 | 10.728 | 10.709 | 11.170 | 11.222 |
| loss compile + Graph | 1 | 10.610 | 10.842 | 11.036 | 11.076 |
| loss compile + Graph | 2 | 10.581 | 10.834 | 11.016 | 11.068 |
| loss compile + Graph | 3 | 10.715 | 10.610 | 11.187 | 11.249 |
| full-objective compile | 1 | 5.083 | 4.925 | 5.787 | 5.940 |
| full-objective compile | 2 | 4.756 | 4.529 | 5.356 | 5.367 |
| full-objective compile | 3 | 4.779 | 4.888 | 5.054 | 5.085 |
| full compile + Graph | 1 | 4.438 | 4.209 | 4.996 | 5.042 |
| full compile + Graph | 2 | 4.423 | 4.212 | 4.847 | 4.979 |
| full compile + Graph | 3 | 4.503 | 4.513 | 4.741 | 4.800 |

## 5. 三次运行池化汇总

统计对象是每个模式的 150 个稳态 round 样本。`run mean average ± stdev` 用于显示
跨进程/跨运行波动；它与池化 mean 的数值应接近，但含义不同。

| 模式 | mean | median | p90 | p95 | run mean average ± stdev |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager immediate | 13.821 | 13.833 | 14.342 | 14.532 | 13.821 ± 0.290 |
| eager deferred | 13.221 | 13.201 | 13.724 | 13.863 | 13.221 ± 0.216 |
| loss-only compile | 12.008 | 12.009 | 12.357 | 12.574 | 12.008 ± 0.065 |
| manual CUDA Graph | 10.661 | 10.840 | 11.119 | 11.160 | 10.661 ± 0.066 |
| loss compile + Graph | 10.635 | 10.669 | 11.105 | 11.173 | 10.635 ± 0.071 |
| full-objective compile | **4.873** | **4.916** | **5.368** | **5.764** | 4.873 ± 0.182 |
| full compile + Graph | **4.455** | **4.289** | **4.957** | **5.000** | 4.455 ± 0.042 |

以 eager deferred 的池化均值 13.221 ms 为基线：full compile 平均降低 63.1%，hybrid
平均降低 66.3%；相应吞吐约为 2.71× 和 2.97×。hybrid 相对纯 full compile 平均再快
8.6%，但 p95 只从 5.764 降到 5.000，不能把这项 microbenchmark 直接等同于端到端 FPS。

注意：本次 benchmark 开启了 packed-staging 选项，但没有向 learner 传入
`sac_graph_packed_source`，因此结果验证的是 Graph 生命周期和 full compile 组合，不是
packed staging 单独收益。

## 6. 正确性与稳定性

- 10 步 BF16 对照：所有 metrics finite；参数、buffer、optimizer state 差异在随机 actor
  采样导致的正常范围内；optimizer step 和 scheduler epoch 均正确。
- 1000 replay 压力测试：critic/actor/temperature step 为 `1000/500/500`；capture 后与
  1000 次 replay 后 allocated/reserved 显存均不增长（`103,634,432 B` / `371,195,904 B`）。
- 自动化回归：聚焦测试 `68 passed`；全量测试 `413 passed, 8 skipped, 3 deselected`；
  Ruff、mypy、pyright 均通过。

## 7. 修改位置和落地建议

实现位于 `src/uni_rl/algos/flash_sac/learner.py`、categorical TD projection、
`src/uni_rl/offpolicy/double_buffer_runner.py` 和对应 builder；UniLab owner 配置补丁见
`patches/unilab_flash_sac_config.patch`。默认生产档只需开启
`compile_full_objectives: true`；升级 PyTorch/CUDA 后若启用 hybrid，应重新运行数值、
replay 和 capture 测试。

完整命令见 [REPRODUCE.md](REPRODUCE.md)，逐次原始 JSON 位于
`results/production_bf16_samples_run{1,2,3}.json`，汇总位于
`results/production_bf16_samples_summary.json`。
