# UniLab FlashSAC RTX 4090 优化实验

这是一个与原始 `unilab_rl` 工作树隔离的 FlashSAC 优化、实验和发布项目。项目包含实现
patch、回归测试、性能脚本、三次原始采样数据、中文报告以及完整复现手册。

## 最终建议

默认生产档：

```yaml
algo:
  algo_params:
    use_compile: true
    compile_full_objectives: true
    use_cuda_graph_critic: false
    use_cuda_graph_actor: false
```

RTX 4090 的三次池化结果（150 个稳态 round）：4.873 ms/round mean，4.916 median，
5.368 p90，5.764 p95。对比 deferred eager 的 13.221 ms/round，平均降低 63.1%。

固定 batch 且追求峰值吞吐时，可开启 hybrid：

```yaml
    use_cuda_graph_critic: true
    use_cuda_graph_actor: true
    use_cuda_graph_critic_packed_staging: true
    use_cuda_graph_actor_packed_staging: true
```

hybrid 为 4.455 mean、4.289 median、4.957 p90、5.000 p95，平均比纯 full compile
再快约 8.6%，但生命周期和升级维护更复杂。

## 三次实验汇总（ms/round）

| 模式 | mean | median | p90 | p95 | run mean stdev |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager immediate | 13.821 | 13.833 | 14.342 | 14.532 | 0.290 |
| eager deferred | 13.221 | 13.201 | 13.724 | 13.863 | 0.216 |
| loss-only compile | 12.008 | 12.009 | 12.357 | 12.574 | 0.065 |
| manual CUDA Graph | 10.661 | 10.840 | 11.119 | 11.160 | 0.066 |
| loss compile + Graph | 10.635 | 10.669 | 11.105 | 11.173 | 0.071 |
| full-objective compile | **4.873** | **4.916** | **5.368** | **5.764** | 0.182 |
| full compile + Graph | **4.455** | **4.289** | **4.957** | **5.000** | 0.042 |

这是 learner microbenchmark，不是包含环境、collector、IPC、replay 采样和 H2D 的端到端
训练 FPS。warmup 10 轮已排除首次 Inductor 编译和 Graph capture cold start。

## 从零复现

### 1. 环境

```bash
cd /home/pc823/桌面/unilabsim/flash_sac_optimization
export UV_CACHE_DIR=/tmp/flash_sac_uv_cache
uv sync --frozen
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

本次环境：RTX 4090、driver 595.84、PyTorch 2.8.0+cu128、Python 3.13。`torch.cuda.is_available()`
必须为 true 才能复现 CUDA Graph/Inductor 结果。

### 2. 回归和静态检查

```bash
uv run pytest -q tests/algos/test_flash_sac_learner.py \
  tests/algos/test_double_buffer_builders.py \
  tests/algos/test_offpolicy_runner_unit.py
uv run pytest -q
uv run ruff check src tests experiments
uv run ruff format --check src tests experiments
uv run mypy src/uni_rl
uv run pyright
```

本次结果：聚焦 `68 passed`；全量 `413 passed, 8 skipped, 3 deselected`；静态检查通过。

### 3. 数值和显存实验

```bash
uv run experiments/validate_numerics.py \
  --matmul-precision highest --steps 10 --batch-size 2048 \
  --obs-dim 98 --critic-obs-dim 101 --action-dim 29 \
  --use-amp --amp-dtype auto \
  --output results/numerics_production_bf16_10steps.json

uv run experiments/validate_replay_stability.py \
  --steps 1000 --batch-size 2048 --policy-frequency 2 \
  --output results/hybrid_replay_stability_1000.json
```

期望数值验证 `passed: true`；replay 压力测试的 optimizer step 为 critic/actor/temperature
`1000/500/500`，capture 后与 1000 replay 后 allocated/reserved 显存不增长。

### 4. 三次性能采样和统计

关闭其他 GPU 负载后执行：

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

脚本保存每个 round 的样本，并计算 mean、median、p90、p95；汇总脚本对三次共 150 个
样本池化，同时保留每次 run mean 和标准差。`ms_per_round` 为兼容旧格式仍表示 mean。

### 5. 应用 patch

对目标 `unilab_rl` checkout 先 dry-run，再应用：

```bash
cd /path/to/unilab_rl
patch --dry-run -p1 < /path/to/this-repo/patches/flash_sac_optimization.patch
patch -p1 < /path/to/this-repo/patches/flash_sac_optimization.patch

cd /path/to/UniLab
patch --dry-run -p1 < /path/to/this-repo/patches/unilab_flash_sac_config.patch
patch -p1 < /path/to/this-repo/patches/unilab_flash_sac_config.patch
```

UniLab owner patch 只开启 `compile_full_objectives: true`，默认关闭手工 Graph。遇到
PyTorch/CUDA 升级时，先重新跑数值、capture 和 replay 测试，再决定是否启用 hybrid。

## 文件索引

- [REPORT.md](REPORT.md)：仿照 FastSAC 项目格式的短版结论和汇总。
- [FLASH_SAC_OPTIMIZATION.md](FLASH_SAC_OPTIMIZATION.md)：中文详细分析、逐次统计和限制。
- [REPRODUCE.md](REPRODUCE.md)：逐步复现、排错和 patch 使用手册。
- [reports/gpu_benchmark_2026-09-24.md](reports/gpu_benchmark_2026-09-24.md)：RTX 4090 数据明细。
- `experiments/`：性能、数值、显存稳定性和汇总脚本。
- `results/production_bf16_samples_*.json`：新版逐 round 原始数据；旧 `production_bf16_run*.json`
  仅保留为历史口径，不用于正式结论。
