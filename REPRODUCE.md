# FlashSAC 实验详细复现手册

## 1. 实验目录与基线

```text
实验项目: 发布仓库 clone 后的 `flash_sac_optimization` 目录
原始仓库: /home/pc823/桌面/unilabsim/unilab_rl
基线 commit: c543650
```

实验项目是独立副本，原始仓库未被修改。下面的 Python 命令全部通过 `uv run` 执行。

## 2. 软硬件要求

本次已验证环境：

- Linux
- NVIDIA GeForce RTX 4090，49140 MiB
- driver 595.84
- PyTorch 2.8.0+cu128
- Python 3.13

进入目录并安装锁定依赖：

```bash
cd /home/pc823/桌面/unilabsim/flash_sac_optimization
export UV_CACHE_DIR=/tmp/flash_sac_uv_cache
uv sync --frozen
```

检查 GPU 和 PyTorch：

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

uv run python -c '
import torch
assert torch.cuda.is_available()
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
'
```

预期关键输出：

```text
torch: 2.8.0+cu128
torch CUDA: 12.8
GPU: NVIDIA GeForce RTX 4090
```

PyTorch wheel 的 CUDA 12.8 和 `nvidia-smi` 显示的 driver 最高 CUDA 能力不需完全相同，只要 `torch.cuda.is_available()` 为 true 即可。

## 3. 源码和静态检查

```bash
uv run ruff check src tests experiments
uv run ruff format --check src tests experiments
uv run mypy src/uni_rl
uv run pyright
```

本次结果：

```text
Ruff lint: passed
Ruff format: 117 files already formatted
mypy: Success, 75 source files
pyright: 0 errors, 0 warnings
```

## 4. 回归测试

### 4.1 FlashSAC 聚焦测试

```bash
uv run pytest -q \
  tests/algos/test_flash_sac_learner.py \
  tests/algos/test_double_buffer_builders.py \
  tests/algos/test_offpolicy_runner_unit.py
```

该命令会在 CUDA 可用时真实执行：

- categorical TD projection capture；
- 手工 CUDA Graph 首次 capture + replay；
- full-objective Inductor + 外层 Graph；
- reward std/temperature 持久 metric buffer；
- optimizer step 计数和 metrics 更新。

本次结果：`68 passed`。

### 4.2 全量测试

```bash
uv run pytest -q
```

本次结果：

```text
413 passed, 8 skipped, 3 deselected
```

## 5. 数值正确性实验

```bash
uv run experiments/validate_numerics.py \
  --matmul-precision highest \
  --steps 10 \
  --batch-count 4 \
  --batch-size 2048 \
  --obs-dim 98 \
  --critic-obs-dim 101 \
  --action-dim 29 \
  --use-amp \
  --amp-dtype auto \
  --output results/numerics_production_bf16_10steps.json
```

脚本会从完全相同的 learner state 出发，对比：

1. eager；
2. loss-only compile；
3. full-objective compile；
4. manual CUDA Graph；
5. full compile + outer manual Graph。

它分开报告 parameters、running buffers 和 optimizer state，检查 metrics finite/非 overwrite、optimizer step 和 scheduler epoch，并为 hybrid 相对 full compile 设置宽松的随机算法数值门槛。成功时最后显示：

```json
"validation": {
  "passed": true
}
```

若只想观察数据而不让门槛决定退出码，加 `--no-check`。

## 6. 1000 步 replay 和显存稳定性

```bash
uv run experiments/validate_replay_stability.py \
  --steps 1000 \
  --batch-size 2048 \
  --policy-frequency 2 \
  --output results/hybrid_replay_stability_1000.json
```

该脚本只压测峰值 hybrid 路径，并且检查：

- 1000 次 critic 和 500 次 actor/temperature update 均未丢失；
- metrics finite 且固定 buffer 未被覆盖；
- capture 后和 1000 replay 后 allocated/reserved 显存完全一致。

成功时输出 `"passed": true`；本次原始结果位于 `results/hybrid_replay_stability_1000.json`。

## 7. 复现 RTX 4090 性能表

### 7.1 测量口径

默认参数已对齐 UniLab 默认 G1 walk FlashSAC：

```text
obs / critic obs / action = 98 / 101 / 29
batch_size = 2048
updates_per_step = 2
policy_frequency = 2
actor hidden/blocks = 128 / 2
critic hidden/blocks = 256 / 2
atoms = 101
AMP = BF16
```

每个 case 先 warmup 10 轮，再计时 50 轮。warmup 不计入时间，因此结果是稳态吞吐，不包含首次编译/capture 开销。

### 7.2 三次重复

确保没有其他 GPU 任务，然后运行：

```bash
for i in 1 2 3; do
  uv run experiments/benchmark_flash_sac.py \
    --device cuda \
    --rounds 50 \
    --warmup 10 \
    --output "results/production_bf16_samples_run${i}.json"
done
```

脚本依次测量 eager immediate、eager deferred、loss-only compile、full compile、manual Graph、loss compile + Graph 和 full compile + Graph。

### 7.3 计算中位数

```bash
uv run experiments/summarize_results.py \
  results/production_bf16_samples_run1.json \
  results/production_bf16_samples_run2.json \
  results/production_bf16_samples_run3.json \
  --output results/production_bf16_samples_summary.json
```

本次最关键的预期中位数：

```text
deferred eager:        13.221 ms/round (median 13.201, p90 13.724, p95 13.863)
full-objective compile: 4.873 ms/round (median 4.916, p90 5.368, p95 5.764)
hybrid:                 4.455 ms/round (median 4.289, p90 4.957, p95 5.000)
```

不同温度、功耗上限和后台负载会导致小幅波动，应比较三次中位数，不要只看单次最优数字。

## 8. 可选 FP32/TF32 实验

BF16 是生产默认。只有在禁用 AMP 的 FP32 实验中，`matmul_precision=high` 才应解释为 TF32 性能档：

```bash
uv run experiments/benchmark_flash_sac.py \
  --device cuda --no-use-amp --matmul-precision highest \
  --output results/fp32_highest.json

uv run experiments/benchmark_flash_sac.py \
  --device cuda --no-use-amp --matmul-precision high \
  --output results/fp32_tf32.json
```

不要把 BF16 基准的 `highest/high` 称为 TF32 对比，因为主体 matmul 已在 BF16 中执行。

## 9. 应用和回退 patch

### 9.1 `unilab_rl`

先在目标 checkout 做 dry-run：

```bash
cd /path/to/unilab_rl
patch --dry-run -p1 < /home/pc823/桌面/unilabsim/flash_sac_optimization/patches/flash_sac_optimization.patch
```

无报错后应用：

```bash
patch -p1 < /home/pc823/桌面/unilabsim/flash_sac_optimization/patches/flash_sac_optimization.patch
```

该 patch 包含 learner、categorical TD projection、builder 透传和测试，是唯一需要应用的 `unilab_rl` 补丁。

### 9.2 UniLab owner YAML

```bash
cd /path/to/UniLab
patch --dry-run -p1 < /home/pc823/桌面/unilabsim/flash_sac_optimization/patches/unilab_flash_sac_config.patch
patch -p1 < /home/pc823/桌面/unilabsim/flash_sac_optimization/patches/unilab_flash_sac_config.patch
```

该 patch 只增加：

```yaml
compile_full_objectives: true
```

并保持手工 Graph 默认关闭。

若 patch 已由 Git 记录，请使用正常 Git revert 流程回退；不要对包含其他未提交工作的目录执行破坏性 reset。

## 10. 常见问题

### `torch.cuda.is_available()` 为 false

先检查 `nvidia-smi`、容器 GPU 映射和 PyTorch wheel 是否带 CUDA。CPU 路径可运行功能测试，但不能复现本报告的 CUDA Graph/Inductor 结果。

### 首次运行很慢

这是 Inductor 编译和 Graph capture 的 cold-start 成本。正式基准已用 10 轮 warmup 排除它。

### FP16 为什么没有进入手工 Graph

FP16 使用 GradScaler，当前实现对手工 Graph fail closed，自动走非手工 Graph 路径。生产默认 `auto` 在 CUDA 上解析为 BF16，不需 GradScaler，可正常进入 hybrid。

### 结果是否代表端到端训练提速

不是。本脚本只测 learner update，batch 已在 GPU。完整训练还受 env step、collector、replay pipeline、IPC 和日志影响。
