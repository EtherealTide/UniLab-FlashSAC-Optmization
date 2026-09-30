#!/usr/bin/env python3
"""Run a real FlashSAC training loop and report learner/collector timing.

Unlike the collector-only benchmark, this command launches the production
``train_flashsac.py`` entrypoint, constructs the selected physics backend, and
reads the timing scalars emitted by the real off-policy runner.  The child
process output is forwarded live so the user can see training progress.

Example (MuJoCo, run from a UniLab environment):

    uv run /path/to/UniLab-FlashSAC-Optmization/experiments/benchmark_flashsac_training.py \
        --unilab-root /path/to/UniLab --backend mujoco

The same command works with ``--backend motrix`` when the Motrix extra is
installed.  The project's own ``src/uni_rl`` is used by default; use
``--uni-rl-src`` to test another local unilab-rl checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import pty
import select
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR: Path | None = None
TRAIN_SCRIPT: Path | None = None
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "results" / "physical_training"


def get_device_info_dict() -> dict[str, object]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        return {
            "cuda": True,
            "gpu": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "torch": torch.__version__,
        }
    except Exception as error:
        return {"cuda": False, "error": f"{type(error).__name__}: {error}"}


@dataclass(frozen=True)
class Scalar:
    step: int
    value: float


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("no timing samples were emitted")
    return {
        "count": len(samples),
        "mean_ms": fmean(samples),
        "median_ms": _percentile(samples, 0.50),
        "p90_ms": _percentile(samples, 0.90),
        "p95_ms": _percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _event_file(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("events.out.tfevents.*"))
    if not candidates:
        raise RuntimeError(f"no TensorBoard event file found under {run_dir}")
    return candidates[-1]


def _read_scalars(run_dir: Path, tag: str, *, scale: float = 1.0) -> list[Scalar]:
    from tensorboard.backend.event_processing import event_accumulator

    accumulator = event_accumulator.EventAccumulator(str(_event_file(run_dir)))
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return [Scalar(int(event.step), float(event.value) * scale) for event in accumulator.Scalars(tag)]


def _read_first_scalars(
    run_dir: Path, tags: Sequence[tuple[str, float]]
) -> list[Scalar]:
    """Read the first timing spelling present in old or current runners."""
    for tag, scale in tags:
        values = _read_scalars(run_dir, tag, scale=scale)
        if values:
            return values
    return []


def _by_step(samples: Sequence[Scalar]) -> dict[int, float]:
    return {sample.step: sample.value for sample in samples}


def parse_timing(
    run_dir: Path,
    *,
    skip_first: int = 0,
    summary_last: int | None = 500,
) -> dict[str, object]:
    """Read per-iteration learner, collector, wall and reward timings."""
    learner = _read_first_scalars(
        run_dir,
        (("timing/learner_train_ms", 1.0), ("Perf/learning_time", 1000.0)),
    )
    collector = _read_first_scalars(
        run_dir,
        (("perf/collector_cycle_ms", 1.0),),
    )
    wall = _read_first_scalars(
        run_dir,
        (("perf/iter_ms", 1.0), ("Perf/iteration_time", 1000.0)),
    )
    reward = _read_first_scalars(
        run_dir,
        (("reward/mean", 1.0), ("Train/mean_reward", 1.0)),
    )
    if not learner:
        raise RuntimeError("training log has no timing/learner_train_ms samples")
    if not collector:
        # Older runners do not have the aggregate scalar, but emit the three
        # mutually exclusive collector phases.  Reconstruct the cycle exactly.
        phase_tags = (
            ("timing/collector_env_step_ms", 1.0),
            ("timing/collector_replay_ms", 1.0),
            ("timing/collector_bookkeeping_ms", 1.0),
        )
        phases = [_by_step(_read_scalars(run_dir, tag, scale=scale)) for tag, scale in phase_tags]
        # New runner records the three mutually exclusive cycle phases without
        # an aggregate scalar.  They are already in milliseconds.
        if not any(phases):
            phases = [
                _by_step(_read_scalars(run_dir, f"Perf/collector_{name}_ms"))
                for name in ("inference_request", "learner_action_wait", "env_step", "replay_write")
            ]
        steps = sorted(set().union(*(phase.keys() for phase in phases)))
        collector = [Scalar(step, sum(phase.get(step, 0.0) for phase in phases)) for step in steps]
    if not collector:
        raise RuntimeError("training log has no collector timing samples")

    learner_by_step = _by_step(learner)
    collector_by_step = _by_step(collector)
    wall_by_step = _by_step(wall)
    reward_by_step = _by_step(reward)
    steps = sorted(set(learner_by_step) & set(collector_by_step))
    rows_all = [
        {
            "step": step,
            "learner_train_ms": learner_by_step[step],
            "collector_cycle_ms": collector_by_step[step],
            "iter_ms": wall_by_step.get(step),
            "reward": reward_by_step.get(step),
        }
        for step in steps
    ]
    if skip_first < 0:
        raise ValueError("skip_first must be non-negative")
    if summary_last is not None and summary_last <= 0:
        raise ValueError("summary_last must be positive")
    eligible_rows = rows_all[skip_first:]
    rows = eligible_rows[-summary_last:] if summary_last is not None else eligible_rows
    if not rows:
        raise RuntimeError(
            f"learner and collector timing series have fewer than {skip_first + 1} common steps"
        )
    return {
        "num_samples": len(rows),
        "num_samples_all": len(rows_all),
        "skip_first": skip_first,
        "summary_last": summary_last,
        "summary_first_step": rows[0]["step"],
        "summary_last_step": rows[-1]["step"],
        "rows_all": rows_all,
        "learner_train_ms": summarize(row["learner_train_ms"] for row in rows),
        "collector_cycle_ms": summarize(row["collector_cycle_ms"] for row in rows),
        "iter_ms": summarize(row["iter_ms"] for row in rows if row["iter_ms"] is not None),
        "rows": rows,
    }


def _command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    if TRAIN_SCRIPT is None:
        raise RuntimeError("main() must resolve --unilab-root before building the command")
    task = f"{args.task}/{args.backend}"
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        f"task={task}",
        "training.no_play=true",
        # Hydra's basic override grammar treats an unquoted absolute path as
        # invalid.  Keep the quotes inside argv so the child Hydra parser sees
        # one string value without relying on shell escaping.
        f"training.log_dir='{run_dir}'",
        # unilab_rl >= 1.4 owns this runner setting. ``++`` keeps the benchmark
        # compatible with UniLab checkouts from before and after the owner YAML
        # added the field, while retaining one timing sample per iteration.
        "++training.log_interval=1",
        f"algo.max_iterations={args.iterations}",
        "algo.save_interval=1000000",
    ]
    optional_overrides = (
        ("algo.num_envs", args.num_envs),
        ("algo.batch_size", args.batch_size),
        ("algo.replay_buffer_n", args.replay_buffer_n),
        ("algo.learning_starts", args.learning_starts),
        ("algo.updates_per_step", args.updates_per_step),
        ("algo.policy_frequency", args.policy_frequency),
    )
    command.extend(f"{name}={value}" for name, value in optional_overrides if value is not None)
    if args.compile is not None:
        enabled = str(args.compile).lower()
        command.extend(
            (
                f"algo.algo_params.use_compile={enabled}",
                f"algo.algo_params.compile_full_objectives={enabled}",
            )
        )
    command.extend(args.extra_override)
    return command


def _run(command: Sequence[str], *, env: dict[str, str]) -> None:
    if ROOT_DIR is None:
        raise RuntimeError("main() must resolve --unilab-root before launching training")
    print("$ " + " ".join(command), flush=True)
    if os.name == "posix":
        # Rich detects a TTY.  A normal PIPE makes the production runner
        # suppress its live dashboard and only emit the final screen; a PTY
        # preserves the same realtime CLI display as direct invocation.
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
            list(command),
            cwd=ROOT_DIR,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)
        try:
            while True:
                readable, _, _ = select.select([master_fd], [], [], 0.1)
                if readable:
                    try:
                        data = os.read(master_fd, 8192)
                    except OSError:
                        data = b""
                    if data:
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                    elif process.poll() is not None:
                        break
                if process.poll() is not None and not readable:
                    break
        finally:
            os.close(master_fd)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, list(command))
        return

    process = subprocess.Popen(
        list(command),
        cwd=ROOT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))


def _print_report(report: dict[str, object]) -> None:
    training_config = report["training_config"]
    assert isinstance(training_config, dict)
    print("\nTraining/update configuration:")
    print(f"  configured iterations: {int(training_config['iterations'])}")
    print(
        "  summary window: last "
        f"{int(report['num_samples'])} timing rows "
        f"(steps {int(report['summary_first_step'])}..{int(report['summary_last_step'])})"
    )
    print(
        "  updates per iteration: "
        f"{int(training_config['critic_updates_per_iteration'])} critic + "
        f"{int(training_config['actor_updates_per_iteration'])} actor + "
        f"{int(training_config['temperature_updates_per_iteration'])} temperature"
    )
    rows = report["rows"]
    assert isinstance(rows, list)
    print("\nPer-iteration timing (ms):")
    print(f"{'step':>8} {'learner':>12} {'collector':>12} {'iter':>12} {'reward':>12}")
    for row in rows:
        assert isinstance(row, dict)
        reward = row.get("reward")
        reward_text = f"{float(reward):12.4f}" if reward is not None else f"{'-':>12}"
        iter_time = row.get("iter_ms")
        iter_text = f"{float(iter_time):12.3f}" if iter_time is not None else f"{'-':>12}"
        print(
            f"{int(row['step']):8d} {float(row['learner_train_ms']):12.3f} "
            f"{float(row['collector_cycle_ms']):12.3f} {iter_text} {reward_text}"
        )
    print("\nSummary (ms):")
    print(f"{'phase':<18} {'mean':>10} {'median':>10} {'p90':>10} {'p95':>10} {'n':>6}")
    for name in ("learner_train_ms", "collector_cycle_ms", "iter_ms"):
        stats = report[name]
        assert isinstance(stats, dict)
        print(
            f"{name:<18} {float(stats['mean_ms']):10.3f} {float(stats['median_ms']):10.3f} "
            f"{float(stats['p90_ms']):10.3f} {float(stats['p95_ms']):10.3f} {int(stats['count']):6d}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mujoco", "motrix"), default="mujoco")
    parser.add_argument("--task", default="g1_walk_flat")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--replay-buffer-n", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--updates-per-step", type=int, default=None)
    parser.add_argument("--policy-frequency", type=int, default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--unilab-root",
        type=Path,
        default=None,
        help="UniLab checkout containing src/unilab/scripts/train_flashsac.py.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--keep-run", action="store_true")
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Exclude the first N timing rows before selecting the summary window.",
    )
    parser.add_argument(
        "--summary-last",
        type=int,
        default=500,
        help="Summarize only the last N eligible timing rows (default: 500).",
    )
    parser.add_argument("--uni-rl-src", type=Path, default=None)
    parser.add_argument("--extra-override", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    global ROOT_DIR, TRAIN_SCRIPT
    args = parse_args(argv)
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.updates_per_step is not None and args.updates_per_step <= 0:
        raise ValueError("--updates-per-step must be positive")
    if args.policy_frequency is not None and args.policy_frequency <= 0:
        raise ValueError("--policy-frequency must be positive")
    if args.skip_first < 0:
        raise ValueError("--skip-first must be non-negative")
    if args.summary_last <= 0:
        raise ValueError("--summary-last must be positive")
    unilab_root = (
        args.unilab_root.resolve()
        if args.unilab_root is not None
        else (PROJECT_DIR.parent / "UniLab").resolve()
    )
    train_script = unilab_root / "src" / "unilab" / "scripts" / "train_flashsac.py"
    if not train_script.is_file():
        raise FileNotFoundError(
            f"cannot find {train_script}; pass --unilab-root pointing at a UniLab checkout"
        )
    ROOT_DIR = unilab_root
    TRAIN_SCRIPT = train_script
    if args.run_dir is None:
        run_dir = DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    source_paths = [str(PROJECT_DIR / "src"), str(ROOT_DIR / "src")]
    if args.uni_rl_src is not None:
        source_paths.insert(0, str(args.uni_rl_src.resolve()))
    env["PYTHONPATH"] = os.pathsep.join(source_paths + [env.get("PYTHONPATH", "")])
    command = _command(args, run_dir)
    _run(command, env=env)
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.is_file():
        raise RuntimeError(f"training did not emit {run_config_path}")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    algo_config = run_config["config"]["algo"]
    report = parse_timing(
        run_dir,
        skip_first=args.skip_first,
        summary_last=args.summary_last,
    )
    updates_per_step = int(algo_config["updates_per_step"])
    policy_frequency = int(algo_config["policy_frequency"])
    actor_updates = (updates_per_step - 1) // policy_frequency + 1
    training_config = {
        "iterations": args.iterations,
        "summary_last": args.summary_last,
        "num_envs": int(algo_config["num_envs"]),
        "batch_size": int(algo_config["batch_size"]),
        "replay_buffer_n": int(algo_config["replay_buffer_n"]),
        "learning_starts": int(algo_config["learning_starts"]),
        "updates_per_step": updates_per_step,
        "policy_frequency": policy_frequency,
        "critic_updates_per_iteration": updates_per_step,
        "actor_updates_per_iteration": actor_updates,
        "temperature_updates_per_iteration": actor_updates,
    }
    payload = {
        "command": command,
        "run_dir": str(run_dir),
        "backend": args.backend,
        "task": args.task,
        "device": get_device_info_dict(),
        "training_config": training_config,
        **report,
    }
    _print_report(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved JSON: {args.output}")
    if not args.keep_run and args.output is None:
        # Keep the run by default when --output is requested; otherwise leave
        # it available for TensorBoard inspection because it contains the
        # actual physics-backed training artifacts.
        print(f"Training logs: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
