# UniLab FlashSAC 性能优化

本仓库保存 FlashSAC 的 full-objective compile 优化、补丁、测试和可复现实验脚本。优化目标是在不改变算法更新次数和数学定义的前提下，减少 GPU kernel launch、CPU/GPU 同步、无效梯度和 metrics 拷贝开销。

详细原理、完整路径定义和全部实验数据见 [FLASH_SAC_OPTIMIZATION.md](FLASH_SAC_OPTIMIZATION.md)。

## 做了什么

- 将 actor/critic 的完整 objective 放进 `torch.compile`，扩大 Inductor 的融合范围。
- actor 更新时冻结 critic 参数，只保留 action 到 critic 输出的梯度路径。
- finite guard 留在 GPU，避免每次 optimizer step 都同步到 CPU。
- metrics 延迟到一轮 learner update 末尾统一 D2H。
- 使用固定 metrics buffer，避免 Inductor CUDA Graph 或手工 CUDA Graph replay 覆盖返回值。
- 正确处理 Inductor 内部 CUDA Graph 与外层手工 CUDA Graph：纯 compile 路径允许 Inductor Graph，手工 Graph 路径关闭嵌套 capture。

推荐默认配置：

```yaml
algo_params:
  use_compile: true
  compile_full_objectives: true
  use_cuda_graph_critic: false
  use_cuda_graph_actor: false
```

手工 CUDA Graph 只建议用于 batch 和输入 shape 固定、追求极限速度且能承担更复杂 graph 生命周期管理的场景。

## 结果摘要

两种 update 口径均使用 `policy_frequency=2`：

- 1 actor / 2 critic：每 iter 有 2 次 critic、1 次 actor、1 次 temperature update。
- 4 actor / 8 critic：每 iter 有 8 次 critic、4 次 actor、4 次 temperature update。

### 真实 MuJoCo 训练

RTX 4090、G1 Walk Flat、256 env、batch 256、1000 iter，统计最后 500 iter。单位为 ms：

| update 口径 | 路径 | learner mean | median | p90 | p95 | learner 降幅 | iter mean |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 actor / 2 critic | deferred eager | 10.118 | 9.965 | 10.760 | 11.092 | — | 11.009 |
| 1 actor / 2 critic | full-objective compile | **4.323** | **4.397** | **4.874** | **5.088** | **57.3%** | **5.341** |
| 4 actor / 8 critic | deferred eager | 36.885 | 36.183 | 39.010 | 40.171 | — | 37.760 |
| 4 actor / 8 critic | full-objective compile | **13.304** | **13.145** | **14.344** | **14.888** | **63.9%** | **14.235** |

这里的 `learner` 是一个 iter 内全部 update 的总耗时，不是单次 critic 或 actor update。`iter` 是完整 runner iteration 墙钟时间。collector 与 learner 采用 double buffer，会发生重叠，因此 `iter != learner + collector`。

update 数量扩大 4 倍时，真实训练的优化后 learner mean 从 4.323 ms 增加到 13.304 ms，只增加 **3.08 倍**。原因是每 iter 固定开销只支付一次，并且 metrics D2H 和最终 CUDA synchronize 也只做一次；详细定量分析见优化报告。

## 复现实验

### 1. 目录和环境

假设目录为：

```text
/path/to/UniLab
/path/to/UniLab-FlashSAC-Optmization
```

UniLab 环境需能导入 `torch`、`unisim`、MuJoCo 和 TensorBoard。先检查：

```bash
cd /path/to/UniLab
uv run --project /path/to/UniLab --no-sync -- python -c \
  'import torch, unisim; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(unisim.__file__)'
```

不要直接执行 `uv run /path/to/UniLab-FlashSAC-Optmization/...`。这种写法可能让 uv 选择优化仓库自己的 `.venv`，导致 `ModuleNotFoundError: unisim`。下面所有命令都显式使用 UniLab 项目环境。

### 2. 真实物理训练

下面命令真正调用 UniLab 的 `train_flashsac.py` 并创建 MuJoCo 环境，同时保留生产 Rich 实时面板。默认训练 1000 iter，统计最后 500 iter：

```bash
uv run --project /path/to/UniLab --no-sync -- python \
  /path/to/UniLab-FlashSAC-Optmization/experiments/benchmark_flashsac_training.py \
  --unilab-root /path/to/UniLab \
  --backend mujoco --iterations 1000 --summary-last 500 \
  --num-envs 256 --batch-size 256 --replay-buffer-n 32 \
  --learning-starts 8 \
  --updates-per-step 2 --policy-frequency 2 \
  --compile \
  --output /path/to/UniLab-FlashSAC-Optmization/results/physical_u2_compile.json
```

四组对照只需修改以下参数：

| 实验 | 参数 |
|---|---|
| 1 actor / 2 critic，优化后 | `--updates-per-step 2 --policy-frequency 2 --compile` |
| 1 actor / 2 critic，deferred eager | `--updates-per-step 2 --policy-frequency 2 --no-compile` |
| 4 actor / 8 critic，优化后 | `--updates-per-step 8 --policy-frequency 2 --compile` |
| 4 actor / 8 critic，deferred eager | `--updates-per-step 8 --policy-frequency 2 --no-compile` |

每组使用不同的 `--output` 文件名。JSON 中：

- `rows_all`：全部共同 timing 样本；
- `rows`：最后 500 个参与统计的样本；
- `learner_train_ms`、`collector_cycle_ms`、`iter_ms`：mean、median、p90、p95；
- `training_config`：每 iter 的 critic、actor、temperature update 数量。

### 3. 数值和稳定性检查

```bash
uv run --project /path/to/UniLab --no-sync -- python experiments/validate_numerics.py
uv run --project /path/to/UniLab --no-sync -- python experiments/validate_replay_stability.py
```
