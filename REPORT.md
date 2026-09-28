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

这里的 `ms/round` 口径必须特别说明：微基准默认 `updates=2`、`policy_frequency=2`，所以一个 round 包含 2 次 critic update、1 次 actor update、1 次 temperature update，以及对应的 target soft update。**4.873 ms 是整个 learner round，不是单次 update**；可直接对比的优化前耗时是 **13.221 ms/round**。4.455 ms 则是再叠加手工 CUDA Graph 的完整 round。

## 真实物理训练

新增 `experiments/benchmark_flashsac_training.py`。它调用 UniLab 的生产 `train_flashsac.py`，实际启动 MuJoCo 或 Motrix collector，并把 learner/collector 的 Rich CLI 面板实时转发到终端。默认训练 1000 iter，只用最后 500 条 timing 计算 mean、median、p90、p95，全部原始记录仍写入 JSON。

示例命令：

```bash
cd /path/to/UniLab
uv run /path/to/UniLab-FlashSAC-Optmization/experiments/benchmark_flashsac_training.py \
  --unilab-root /path/to/UniLab \
  --backend mujoco --num-envs 256 \
  --output /path/to/UniLab-FlashSAC-Optmization/results/physical_training/summary.json
```

脚本默认同样是 `updates_per_step=2`、`policy_frequency=2`。Rich 面板和 JSON 中的 `learner_train_ms` 是每个 iter 内 2 次 critic、1 次 actor 和 1 次 temperature update 的合计，不是单次 update；`iter_ms` 才是完整 iteration 墙钟时间。总 iteration 不等于 learner 加 collector，因为 double-buffer runner 会让两者重叠。

基准脚本会显式覆盖 update 数；当前 `g1_walk_flat/mujoco` 任务 YAML 自身配置的是 `updates_per_step=8`，直接使用该配置时每 iter 是 8 次 critic、4 次 actor/temperature，耗时不能与 2-update 结果直接比较。

`--summary-last 500` 控制末尾统计窗口，`--skip-first N` 可在选取末尾窗口前额外排除开头样本。正式的真实物理性能结论应同时跑优化前、优化后各 1000 iter；13.221 → 4.873 ms 是隔离 learner 的微基准结果，不能冒充端到端物理训练结果。

本机短跑已确认真实 MuJoCo 链路能够输出实时训练面板和 timing。短跑样本只用于验证工具，不替代正式多次重复实验。

## 注意事项

- 需要在 UniLab 环境中运行，并安装选定物理 backend 的 extra。
- 当前本地执行时必须使用包含优化代码的 `unilab_rl` checkout，脚本默认使用本仓库 `src/uni_rl`。
- `training.log_dir` 已在脚本内部加 Hydra 引号；直接传未加引号的绝对路径会触发 `LexerNoViableAltException`。
- 这份报告覆盖原 learner 优化结果；Triton 实验不属于本报告。
