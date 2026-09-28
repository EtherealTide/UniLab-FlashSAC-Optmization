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

计时工具会启动 UniLab 的生产 `train_flashsac.py`，真正创建 MuJoCo/Motrix 环境，并保留 Rich 的实时训练面板。它同时把每轮 learner、collector、总 iteration、reward 写入 JSON，并统计 mean、median、p90、p95。

```bash
cd /path/to/UniLab
uv run /path/to/UniLab-FlashSAC-Optmization/experiments/benchmark_flashsac_training.py \
  --unilab-root /path/to/UniLab \
  --backend mujoco --iterations 20 --num-envs 256 \
  --output /path/to/UniLab-FlashSAC-Optmization/results/physical_training/summary.json
```

compile 首轮会包含 Inductor 编译开销；观察稳态时可以加 `--skip-first 4`。训练日志保存在 `results/physical_training/<timestamp>/`，可用 TensorBoard 查看。

完整命令和排错说明见 [REPRODUCE.md](REPRODUCE.md)，结果解释见 [REPORT.md](REPORT.md)。
