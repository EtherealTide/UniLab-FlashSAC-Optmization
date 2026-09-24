# RTX 4090 FlashSAC GPU 实验报告（2026-09-24）

这是 `FLASH_SAC_OPTIMIZATION.md` 的数据明细版。正式统计来自
`production_bf16_samples_run{1,2,3}.json`，每次 50 个稳态 round，三次共 150 个样本。

## 环境与口径

```text
GPU: NVIDIA GeForce RTX 4090, 49140 MiB
Driver: 595.84
PyTorch: 2.8.0+cu128
AMP: BF16 (auto)
batch / updates / policy_frequency: 2048 / 2 / 2
obs / critic obs / action: 98 / 101 / 29
warmup / measured / repeats: 10 / 50 / 3
```

每个 round 的计时包含本轮 learner update，warmup 不计入。batch 已经在 GPU，因而这不是
端到端训练 FPS。`mean/median/p90/p95` 都以毫秒为单位；p90/p95 对 round 样本使用线性
插值，三次汇总时先池化 150 个样本再计算。

## 每次运行

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

## 三次池化汇总

| 模式 | mean | median | p90 | p95 | run mean average ± stdev |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager immediate | 13.821 | 13.833 | 14.342 | 14.532 | 13.821 ± 0.290 |
| eager deferred | 13.221 | 13.201 | 13.724 | 13.863 | 13.221 ± 0.216 |
| loss-only compile | 12.008 | 12.009 | 12.357 | 12.574 | 12.008 ± 0.065 |
| manual CUDA Graph | 10.661 | 10.840 | 11.119 | 11.160 | 10.661 ± 0.066 |
| loss compile + Graph | 10.635 | 10.669 | 11.105 | 11.173 | 10.635 ± 0.071 |
| full-objective compile | **4.873** | **4.916** | **5.368** | **5.764** | 4.873 ± 0.182 |
| full compile + Graph | **4.455** | **4.289** | **4.957** | **5.000** | 4.455 ± 0.042 |

以 deferred eager 的池化 mean 为基线：full compile 降低 63.1%，hybrid 降低 66.3%；
hybrid 相对纯 full compile 再快约 8.6%。

## 复现命令

```bash
for i in 1 2 3; do
  uv run experiments/benchmark_flash_sac.py \
    --device cuda --rounds 50 --warmup 10 \
    --output "results/production_bf16_samples_run${i}.json"
done

uv run experiments/summarize_results.py \
  results/production_bf16_samples_run1.json \
  results/production_bf16_samples_run2.json \
  results/production_bf16_samples_run3.json \
  --output results/production_bf16_samples_summary.json
```

数值和显存稳定性实验命令见 [REPRODUCE.md](../REPRODUCE.md)。
