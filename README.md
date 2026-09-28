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
  --backend mujoco --num-envs 256 \
  --output /path/to/UniLab-FlashSAC-Optmization/results/physical_training/summary.json
```

The default run is 1000 iterations and summarizes the last 500 timing rows with mean, median, p90, and p95 while retaining all rows in `rows_all`. The child process keeps the production Rich dashboard visible in real time. Use `--summary-last N` to change the window or `--skip-first N` for an additional leading exclusion.

With the defaults, one iteration contains two critic updates plus one actor and one temperature update. The reported 4.873 ms learner microbenchmark is one complete learner round, not one update; the comparable pre-optimization result is 13.221 ms.

See [REPRODUCE.md](REPRODUCE.md) for setup and troubleshooting, and [REPORT.md](REPORT.md) for the concise Chinese report.
