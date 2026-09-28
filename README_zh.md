# UniLab FlashSAC 优化实验

本仓库保存 FlashSAC 的 full-objective compile 优化、回归测试，以及真实物理后端训练计时工具。当前优化不是 Triton categorical-target 实验。

## 结论

RTX 4090 learner 微基准（BF16、batch 2048）中：

| 路径 | mean | median | p90 | p95 |
|---|---:|---:|---:|---:|
| deferred eager | 13.221 | 13.201 | 13.724 | 13.863 |
| full-objective compile | 4.873 | 4.916 | 5.368 | 5.764 |
| full compile + manual Graph | 4.455 | 4.289 | 4.957 | 5.000 |

默认推荐 `use_compile=true` + `compile_full_objectives=true`。手工 Graph 只作为固定 batch 的可选峰值档。

## 真实训练计时

计时工具会启动 UniLab 的生产 `train_flashsac.py`，真正创建 MuJoCo/Motrix 环境，并保留 Rich 的实时训练面板。默认训练 **1000 iter**，只对最后 **500 iter** 统计 mean、median、p90、p95；全部逐轮数据仍保存在 JSON 的 `rows_all` 中。

```bash
cd /path/to/UniLab
uv run /path/to/UniLab-FlashSAC-Optmization/experiments/benchmark_flashsac_training.py \
  --unilab-root /path/to/UniLab \
  --backend mujoco --num-envs 256 \
  --output /path/to/UniLab-FlashSAC-Optmization/results/physical_training/summary.json
```

默认 `updates_per_step=2`、`policy_frequency=2`，所以一个训练 iter 会执行 2 次 critic update，以及 1 次 actor update 和随 actor 一起进行的 1 次 temperature update。报告中的 4.873 ms 是上述完整 learner round 的时间，不是单次 update；同口径优化前是 13.221 ms。Rich 面板的 `Learner` 也是一个 iter 内全部 learner update 的合计，`Iter Wall` 才是包含 runner 协调在内的完整迭代墙钟时间。

最后 500 条默认已排除 compile 冷启动；需要改变窗口时使用 `--summary-last N`，需要额外排除开头样本时使用 `--skip-first N`。训练日志保存在 `results/physical_training/<timestamp>/`，可用 TensorBoard 查看。

完整命令和排错说明见 [REPRODUCE.md](REPRODUCE.md)，结果解释见 [REPORT.md](REPORT.md)。
