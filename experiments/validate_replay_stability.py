"""Stress the optimized FlashSAC hybrid path for replay and memory stability."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uni_rl.algos.flash_sac.learner import FlashSACLearner


def make_batch(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randn(batch_size, 98, device="cuda"),
        "critic": torch.randn(batch_size, 101, device="cuda"),
        "actions": torch.tanh(torch.randn(batch_size, 29, device="cuda")),
        "rewards": torch.randn(batch_size, device="cuda"),
        "next_obs": torch.randn(batch_size, 98, device="cuda"),
        "next_critic": torch.randn(batch_size, 101, device="cuda"),
        "dones": torch.zeros(batch_size, device="cuda"),
        "truncated": torch.zeros(batch_size, device="cuda"),
    }


def optimizer_steps(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        int(state["step"].item())
        for state in optimizer.state.values()
        if isinstance(state.get("step"), torch.Tensor)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--policy-frequency", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")

    torch.manual_seed(1234)
    learner = FlashSACLearner(
        obs_dim=98,
        critic_obs_dim=101,
        action_dim=29,
        actor_hidden_dim=128,
        critic_hidden_dim=256,
        actor_num_blocks=2,
        critic_num_blocks=2,
        num_atoms=101,
        device="cuda",
        use_amp=True,
        amp_dtype="auto",
        use_compile=True,
        compile_full_objectives=True,
        use_cuda_graph_critic=True,
        use_cuda_graph_actor=True,
    )
    batches = [make_batch(args.batch_size) for _ in range(4)]

    # First calls compile, capture, and replay each graph once.
    learner.update_critic_cuda_graph(batches[0], read_metrics=False)
    learner.update_actor_cuda_graph(batches[0], read_metrics=False)
    torch.cuda.synchronize()
    allocated_after_capture = torch.cuda.memory_allocated()
    reserved_after_capture = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    final_metrics: dict[str, float] = {}
    for step in range(1, args.steps):
        batch = batches[step % len(batches)]
        read_critic = step == args.steps - 1
        final_metrics.update(learner.update_critic_cuda_graph(batch, read_metrics=read_critic))
        if step % args.policy_frequency == 0:
            learner.update_actor_cuda_graph(batch, read_metrics=False)
    final_metrics.update(learner.read_deferred_actor_metrics())
    torch.cuda.synchronize()

    actor_updates = 1 + (args.steps - 1) // args.policy_frequency
    payload = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "steps": args.steps,
        "actor_updates": actor_updates,
        "batch_size": args.batch_size,
        "policy_frequency": args.policy_frequency,
        "allocated_after_capture_bytes": allocated_after_capture,
        "allocated_after_replays_bytes": torch.cuda.memory_allocated(),
        "reserved_after_capture_bytes": reserved_after_capture,
        "reserved_after_replays_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_during_replays_bytes": torch.cuda.max_memory_allocated(),
        "critic_optimizer_steps": sorted(optimizer_steps(learner.critic_optimizer)),
        "actor_optimizer_steps": sorted(optimizer_steps(learner.actor_optimizer)),
        "temperature_optimizer_steps": sorted(optimizer_steps(learner.temperature_optimizer)),
        "final_metrics": final_metrics,
    }
    checks = {
        "metrics_finite": all(math.isfinite(value) for value in final_metrics.values()),
        "metrics_not_overwritten": final_metrics["reward_scale_std"] > 0.0
        and final_metrics["temperature"] > 0.0,
        "critic_step_count": payload["critic_optimizer_steps"] == [args.steps],
        "actor_step_count": payload["actor_optimizer_steps"] == [actor_updates],
        "temperature_step_count": payload["temperature_optimizer_steps"] == [actor_updates],
        "allocated_memory_stable": payload["allocated_after_replays_bytes"]
        == allocated_after_capture,
        "reserved_memory_stable": payload["reserved_after_replays_bytes"] == reserved_after_capture,
    }
    payload["checks"] = checks
    payload["passed"] = all(checks.values())
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not payload["passed"]:
        raise SystemExit("FlashSAC replay stability validation failed")


if __name__ == "__main__":
    main()
