# UniLab FlashSAC 默认配置优化

本仓库记录 FlashSAC 在 UniLab 真实 MuJoCo 训练中的性能分析、代码修改和可复现实验。本轮唯一验收口径是 Hydra owner config 的默认参数，测试脚本不覆盖它们：

```text
num_envs=2048, batch_size=8192
updates_per_step=8, policy_frequency=4
```

每个 iteration 执行 **8 critic + 2 actor + 2 temperature + 8 target soft update**。

## 结论

- 最新 `unilab_rl` 的 NVIDIA 路径已使用 full-objective Inductor 编译，再用外层 CUDA Graph 捕获完整 update cycle。
- 本轮去掉前 7 次 critic 和第 1 次 actor 最终会被覆盖的 metrics stack，只保留整轮最后一组指标。
- 基准脚本不再隐式改成 256 env / batch 256，而是从 `run_config.json` 回读实际配置，并支持新旧 runner timing tag。
- 尝试过“仅计算 actor/critic 实际使用的 predictor 半区”；真实训练反而慢约 1 ms，已撤销。

RTX 4090、MuJoCo、1000 iter，统计最后 500 iter：

| 指标 | mean | median | p90 | p95 |
|---|---:|---:|---:|---:|
| learner / ms | 55.886 | 55.898 | 56.488 | 56.653 |
| collector cycle / ms | 57.804 | 57.797 | 58.602 | 58.927 |
| iteration wall / ms | 57.550 | 57.546 | 58.192 | 58.311 |

**20 ms 目标未达成。** 历史上“约 24 ms”的日志是 `4096 env / batch 2048 / 8 critic / 4 actor`，不是本轮 `batch 8192 / 8 critic / 2 actor` 口径。batch 增大 4 倍后，critic 的大型矩阵乘、BatchNorm、backward 和 Adam 成为主要开销，单纯减少 launch/D2H 不可能再带来 2.8 倍加速。详见 [优化报告](FLASH_SAC_OPTIMIZATION.md)。

## 复现

假设目录为：

```text
/path/to/UniLab
/path/to/UniLab-FlashSAC-Optmization
```

1. 确认 UniLab 环境能访问 GPU、MuJoCo 和 `unisim`：

```bash
cd /path/to/UniLab
uv run --project /path/to/UniLab --no-sync -- python -c \
  'import torch, unisim; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(unisim.__file__)'
```

2. 确认 Hydra 最终合成值：

```bash
uv run --project /path/to/UniLab --no-sync -- \
  python src/unilab/scripts/train_flashsac.py \
  --cfg job --resolve task=g1_walk_flat/mujoco
```

输出中必须看到 `num_envs: 2048`、`batch_size: 8192`、`updates_per_step: 8`、`policy_frequency: 4`。

3. 运行真实物理训练。不传上述四个参数，它们由 owner config 提供：

```bash
uv run --project /path/to/UniLab --no-sync -- python \
  /path/to/UniLab-FlashSAC-Optmization/experiments/benchmark_flashsac_training.py \
  --unilab-root /path/to/UniLab \
  --backend mujoco \
  --iterations 1000 --summary-last 500 \
  --output /path/to/UniLab-FlashSAC-Optmization/results/physical_training_comparison.json
```

脚本保留生产 Rich 实时面板，并在 JSON 中写入实际 compose 参数、update 数、全部原始 timing 行以及 mean / median / p90 / p95。

> 必须显式使用 `--project /path/to/UniLab`。否则 `uv` 可能选中优化仓库的环境，导致 `ModuleNotFoundError: unisim`。

## 验证

```bash
cd /path/to/UniLab-FlashSAC-Optmization
PYTHONPATH=src uv run --project /path/to/UniLab --no-sync -- \
  pytest -q tests/algos/test_flash_sac_learner.py \
  tests/algos/test_double_buffer_builders.py
```

本轮结果：`31 passed, 8 skipped`。
