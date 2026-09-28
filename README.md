# UniLab FlashSAC Optimization

This repository contains the FlashSAC full-objective compile optimization, regression artifacts, and a real physics-backed training timing tool. It does not include the separate Triton categorical-target experiment.

The RTX 4090 learner benchmark is summarized in [README_zh.md](README_zh.md). The important production setting is:

```yaml
algo_params:
  use_compile: true
  compile_full_objectives: true
```

For an end-to-end run, execute the benchmark from a UniLab environment:

```bash
cd /path/to/UniLab
uv run /path/to/UniLab-FlashSAC-Optmization/experiments/benchmark_flashsac_training.py \
  --unilab-root /path/to/UniLab \
  --backend mujoco --iterations 20 --num-envs 256 \
  --output /path/to/UniLab-FlashSAC-Optmization/results/physical_training/summary.json
```

The child process keeps the production Rich dashboard visible in real time. The JSON contains per-iteration learner, collector, total iteration, and reward values plus mean, median, p90, and p95. Use `--skip-first 4` to exclude compile warm-up from the summary while retaining all rows in `rows_all`.

See [REPRODUCE.md](REPRODUCE.md) for setup and troubleshooting, and [REPORT.md](REPORT.md) for the concise Chinese report.
