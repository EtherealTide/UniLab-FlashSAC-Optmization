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
  --iterations 20 \
  --num-envs 256 \
  --batch-size 256 \
  --replay-buffer-n 32 \
  --learning-starts 8 \
  --updates-per-step 2 \
  --output /path/to/UniLab-FlashSAC-Optmization/results/physical_training/summary.json
```

终端会实时显示生产 FlashSAC 的 Rich 面板，包括 iteration、Steps/s、learner、collector、loss 和 reward。训练日志位于 `results/physical_training/<timestamp>/`。

如果要测试 Motrix：

```bash
... --backend motrix
```

如果当前环境没有通过安装包提供优化版 `unilab_rl`，可以显式指定 checkout：

```bash
... --uni-rl-src /path/to/unilab_rl-flashsac-optimization/src
```

## 3. 编译冷启动处理

full-objective compile 的前几轮可能包含编译/capture 时间。保留完整训练过程但只用稳态样本统计：

```bash
... --skip-first 4
```

输出 JSON 中：

- `rows_all`：所有共同 timing 样本；
- `rows`：去掉 `skip-first` 后参与统计的样本；
- `learner_train_ms`、`collector_cycle_ms`、`iter_ms`：各自的 mean、median、p90、p95。

## 4. 查看 TensorBoard

```bash
uv run tensorboard --logdir /path/to/UniLab-FlashSAC-Optmization/results/physical_training
```

## 5. learner 微基准（可选）

原有纯 learner 实验仍可运行：

```bash
cd /path/to/UniLab-FlashSAC-Optmization
uv run experiments/benchmark_flash_sac.py --device cuda --rounds 50 --warmup 10
```

它不启动物理引擎，只用于隔离 learner 优化；端到端结论应以第 2 节的真实训练计时为准。

## 6. Hydra 路径错误

如果看到：

```text
LexerNoViableAltException: training.log_dir=/home/...
```

说明使用了旧脚本。当前脚本已经把 `training.log_dir` 作为带引号的 Hydra override 传入；请从本仓库重新运行 `experiments/benchmark_flashsac_training.py`，不要手工删除引号。
