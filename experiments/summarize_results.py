"""Summarize repeated FlashSAC benchmark JSON files.

The primary distribution statistics are pooled over steady-state round samples
from all input runs. Per-run means are also retained so run-to-run variance is
visible in the report.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    grouped: dict[tuple[str, bool, bool, bool, bool], list[dict[str, object]]] = defaultdict(list)
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload["results"]:
            key = (
                str(payload.get("matmul_precision", "highest")),
                bool(result.get("deferred_metrics", False)),
                bool(result.get("compile", False)),
                bool(result.get("compile_full_objectives", False)),
                bool(result.get("cuda_graphs", False)),
            )
            samples = result.get("samples_ms_per_round")
            normalized = (
                [float(value) for value in samples]
                if isinstance(samples, list)
                else [float(result["ms_per_round"])]
            )
            grouped[key].append(
                {
                    "mean_ms_per_round": float(result["ms_per_round"]),
                    "samples_ms_per_round": normalized,
                }
            )

    summary = []
    for (
        precision,
        deferred,
        compile_methods,
        full_objectives,
        cuda_graphs,
    ), runs in sorted(grouped.items()):
        samples = [
            sample
            for run in runs
            for sample in run["samples_ms_per_round"]  # type: ignore[index]
        ]
        run_means = [float(run["mean_ms_per_round"]) for run in runs]
        summary.append(
            {
                "matmul_precision": precision,
                "deferred_metrics": deferred,
                "compile": compile_methods,
                "compile_full_objectives": full_objectives,
                "cuda_graphs": cuda_graphs,
                "runs": len(runs),
                "samples": len(samples),
                "mean_ms_per_round": statistics.fmean(samples),
                "median_ms_per_round": percentile(samples, 0.50),
                "p90_ms_per_round": percentile(samples, 0.90),
                "p95_ms_per_round": percentile(samples, 0.95),
                "run_mean_average_ms": statistics.fmean(run_means),
                "run_mean_stdev_ms": (statistics.stdev(run_means) if len(run_means) > 1 else 0.0),
                "run_means_ms": run_means,
            }
        )
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
