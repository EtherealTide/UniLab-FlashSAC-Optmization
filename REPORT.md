# UniLab FlashSAC 优化报告

## 结论

这次优化针对的是 FlashSAC 的 learner，不是 Triton。主要动作是把 actor/critic 的完整 objective 放进 `torch.compile`，减少小 kernel launch 和中间 tensor；同时保留 graph-safe 的固定 metrics buffer、GPU finite guard、延迟 metrics D2H，以及 actor 更新时冻结 critic 参数。

RTX 4090、PyTorch 2.8.0+cu128、BF16、batch 2048 的 learner 稳态微基准如下，单位为 ms/round：

| 路径 | mean | median | p90 | p95 |
|---|---:|---:|---:|---:|
| deferred eager | 13.221 | 13.201 | 13.724 | 13.863 |
| full-objective compile | **4.873** | **4.916** | **5.368** | **5.764** |
| full compile + manual Graph | **4.455** | **4.289** | **4.957** | **5.000** |

full-objective compile 相比 deferred eager 平均减少 63.1%。hybrid 再快一些，但依赖固定 batch 和更复杂的 graph 生命周期，默认不打开。

## 真实物理训练

新增 `experiments/benchmark_flashsac_training.py`。它调用 UniLab 的生产 `train_flashsac.py`，实际启动 MuJoCo 或 Motrix collector，并把 learner/collector 的 Rich CLI 面板实时转发到终端。每轮 timing 同时写入 JSON，统计 mean、median、p90、p95。

示例命令：

```bash
cd /path/to/UniLab
uv run /path/to/UniLab-FlashSAC-Optmization/experiments/benchmark_flashsac_training.py \
  --unilab-root /path/to/UniLab \
  --backend mujoco --iterations 20 --num-envs 256 \
  --output /path/to/UniLab-FlashSAC-Optmization/results/physical_training/summary.json
```

`--skip-first 4` 可排除 compile 冷启动，但 JSON 仍保留 `rows_all`。总 iteration 不等于 learner 加 collector，因为 double-buffer runner 会让两者重叠；性能判断应看完整 `iter_ms`。

本机短跑已确认真实 MuJoCo 链路能够输出实时训练面板和 timing。短跑样本只用于验证工具，不替代正式多次重复实验。

## 注意事项

- 需要在 UniLab 环境中运行，并安装选定物理 backend 的 extra。
- 当前本地执行时必须使用包含优化代码的 `unilab_rl` checkout，脚本默认使用本仓库 `src/uni_rl`。
- `training.log_dir` 已在脚本内部加 Hydra 引号；直接传未加引号的绝对路径会触发 `LexerNoViableAltException`。
- 这份报告覆盖原 learner 优化结果；Triton 实验不属于本报告。
