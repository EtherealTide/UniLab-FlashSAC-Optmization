# FlashSAC 复现步骤

## 1. 准备

假设目录如下：

```text
/path/to/UniLab/
/path/to/UniLab-FlashSAC-Optmization/
```

进入 UniLab 环境并确认 GPU：

```bash
cd /path/to/UniLab
uv run python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

`torch.cuda.is_available()` 必须为 `True`。UniLab 需要安装 MuJoCo 或 Motrix 对应 extra。

## 2. 真实物理训练和实时面板

```bash
cd /path/to/UniLab
uv run /path/to/UniLab-FlashSAC-Optmization/experiments/benchmark_flashsac_training.py \
  --unilab-root /path/to/UniLab \
  --backend mujoco \
  --task g1_walk_flat \
  --num-envs 256 \
  --batch-size 256 \
  --replay-buffer-n 32 \
  --learning-starts 8 \
  --updates-per-step 2 \
  --output /path/to/UniLab-FlashSAC-Optmization/results/physical_training/summary.json
```

未指定时默认训练 1000 iter，并只汇总最后 500 条 timing。终端会实时显示生产 FlashSAC 的 Rich 面板，包括 iteration、Steps/s、learner、collector、loss 和 reward。训练日志位于 `results/physical_training/<timestamp>/`。

如果要测试 Motrix：

```bash
... --backend motrix
```

如果当前环境没有通过安装包提供优化版 `unilab_rl`，可以显式指定 checkout：

```bash
... --uni-rl-src /path/to/unilab_rl-flashsac-optimization/src
```

## 3. update 数量和计时口径

默认参数为：

```text
updates_per_step = 2
policy_frequency = 2
```

因此每个训练 iter 包含 2 次 critic update、1 次 actor update 和 1 次 temperature update。`learner_train_ms`/面板中的 `Learner` 是这些 learner update 在一个 iter 内的合计；`iter_ms`/`Iter Wall` 是完整 iteration 墙钟时间。微基准表中的 4.873 ms 也是相同 update 结构下的一整个 learner round，优化前同口径为 13.221 ms，不是 4 ms/次 update。

## 4. 统计窗口和编译冷启动

默认的最后 500 iter 窗口已经避开启动阶段。可以显式改变窗口：

```bash
... --summary-last 500
```

若还要额外排除开头样本，可加 `--skip-first 4`；筛选顺序是先排除开头，再取最后 N 条。

输出 JSON 中：

- `rows_all`：所有共同 timing 样本；
- `rows`：参与最后窗口统计的样本；
- `training_config`：每 iter 的 critic/actor/temperature update 数量；
- `learner_train_ms`、`collector_cycle_ms`、`iter_ms`：各自的 mean、median、p90、p95。

## 5. 查看 TensorBoard

```bash
uv run tensorboard --logdir /path/to/UniLab-FlashSAC-Optmization/results/physical_training
```

## 6. learner 微基准（可选）

原有纯 learner 实验仍可运行：

```bash
cd /path/to/UniLab-FlashSAC-Optmization
uv run experiments/benchmark_flash_sac.py --device cuda --rounds 50 --warmup 10
```

它不启动物理引擎，只用于隔离 learner 优化；端到端结论应以第 2 节的真实训练计时为准。

## 7. Hydra 路径错误

如果看到：

```text
LexerNoViableAltException: training.log_dir=/home/...
```

说明使用了旧脚本。当前脚本已经把 `training.log_dir` 作为带引号的 Hydra override 传入；请从本仓库重新运行 `experiments/benchmark_flashsac_training.py`，不要手工删除引号。
