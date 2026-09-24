"""Compare FlashSAC updates across eager, compile, and CUDA Graph paths."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uni_rl.algos.flash_sac.learner import FlashSACLearner


def make_learner(
    *,
    compile_methods: bool,
    compile_full: bool,
    graphs: bool,
    obs_dim: int,
    critic_obs_dim: int,
    action_dim: int,
    use_amp: bool,
    amp_dtype: str,
) -> FlashSACLearner:
    return FlashSACLearner(
        obs_dim=obs_dim,
        action_dim=action_dim,
        critic_obs_dim=critic_obs_dim,
        actor_hidden_dim=128,
        critic_hidden_dim=256,
        actor_num_blocks=2,
        critic_num_blocks=2,
        num_atoms=101,
        device="cuda",
        normalize_reward=False,
        use_compile=compile_methods,
        compile_full_objectives=compile_full,
        use_cuda_graph_critic=graphs,
        use_cuda_graph_actor=graphs,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )


def make_batch(
    batch_size: int,
    *,
    obs_dim: int,
    critic_obs_dim: int,
    action_dim: int,
) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randn(batch_size, obs_dim, device="cuda"),
        "critic": torch.randn(batch_size, critic_obs_dim, device="cuda"),
        "actions": torch.tanh(torch.randn(batch_size, action_dim, device="cuda")),
        "rewards": torch.randn(batch_size, device="cuda"),
        "next_obs": torch.randn(batch_size, obs_dim, device="cuda"),
        "next_critic": torch.randn(batch_size, critic_obs_dim, device="cuda"),
        "dones": torch.zeros(batch_size, device="cuda"),
        "truncated": torch.zeros(batch_size, device="cuda"),
    }


def update(learner: FlashSACLearner, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    critic = learner.update_critic_cuda_graph(batch)
    actor = learner.update_actor_cuda_graph(batch)
    if not learner.cuda_graph_critic_captures_target_update:
        learner.soft_update_target()
    torch.cuda.synchronize()
    return critic | actor


def state_groups(learner: FlashSACLearner) -> dict[str, torch.Tensor]:
    parameters: list[torch.Tensor] = []
    buffers: list[torch.Tensor] = []
    optimizer_state: list[torch.Tensor] = []
    for module in (learner.actor, learner.critic, learner.target_critic, learner.temperature):
        parameters.extend(value.detach().reshape(-1).float() for value in module.parameters())
        buffers.extend(value.detach().reshape(-1).float() for value in module.buffers())
    for optimizer in (
        learner.actor_optimizer,
        learner.critic_optimizer,
        learner.temperature_optimizer,
    ):
        for state in optimizer.state.values():
            optimizer_state.extend(
                value.detach().reshape(-1).float()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            )
    return {
        "parameters": torch.cat(parameters),
        "buffers": torch.cat(buffers),
        "optimizer_state": torch.cat(optimizer_state),
    }


def difference(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual - expected).abs()
    return {
        "max_abs": float(delta.max().cpu()),
        "mean_abs": float(delta.mean().cpu()),
    }


def update_counters(learner: FlashSACLearner) -> dict[str, list[float]]:
    optimizer_steps: list[float] = []
    for optimizer in (
        learner.actor_optimizer,
        learner.critic_optimizer,
        learner.temperature_optimizer,
    ):
        optimizer_steps.extend(
            float(state["step"].item())
            for state in optimizer.state.values()
            if isinstance(state.get("step"), torch.Tensor)
        )
    return {
        "optimizer_steps": optimizer_steps,
        "scheduler_epochs": [
            float(learner.actor_scheduler.last_epoch),
            float(learner.critic_scheduler.last_epoch),
            float(learner.temperature_scheduler.last_epoch),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="highest")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--obs-dim", type=int, default=98)
    parser.add_argument("--critic-obs-dim", type=int, default=101)
    parser.add_argument("--action-dim", type=int, default=29)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("auto", "fp16", "bf16"), default="auto")
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.manual_seed(1234)
    learner_kwargs = {
        "obs_dim": args.obs_dim,
        "critic_obs_dim": args.critic_obs_dim,
        "action_dim": args.action_dim,
        "use_amp": args.use_amp,
        "amp_dtype": args.amp_dtype,
    }
    eager = make_learner(
        compile_methods=False,
        compile_full=False,
        graphs=False,
        **learner_kwargs,
    )
    initial = copy.deepcopy(eager.get_state_dict())
    learners = {
        "eager": eager,
        "loss_compile": make_learner(
            compile_methods=True, compile_full=False, graphs=False, **learner_kwargs
        ),
        "full_compile": make_learner(
            compile_methods=True, compile_full=True, graphs=False, **learner_kwargs
        ),
        "manual_graph": make_learner(
            compile_methods=False, compile_full=False, graphs=True, **learner_kwargs
        ),
        "hybrid": make_learner(
            compile_methods=True, compile_full=True, graphs=True, **learner_kwargs
        ),
    }
    for name, learner in learners.items():
        if name != "eager":
            learner.load_state_dict(copy.deepcopy(initial))
    batches = [
        make_batch(
            args.batch_size,
            obs_dim=args.obs_dim,
            critic_obs_dim=args.critic_obs_dim,
            action_dim=args.action_dim,
        )
        for _ in range(args.batch_count)
    ]

    results: dict[str, dict[str, float]] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    counters: dict[str, dict[str, list[float]]] = {}
    for name, learner in learners.items():
        torch.cuda.manual_seed_all(9876)
        for step in range(args.steps):
            results[name] = update(learner, batches[step % len(batches)])
        states[name] = state_groups(learner)
        counters[name] = update_counters(learner)

    reference = states["eager"]
    comparisons = {}
    for name in learners.keys() - {"eager"}:
        comparisons[name] = {
            group: difference(states[name][group], reference[group]) for group in reference
        } | {
            "metric_max_abs": max(
                abs(results[name][key] - results["eager"][key]) for key in results["eager"]
            ),
        }
    hybrid_parameter_diff = difference(
        states["hybrid"]["parameters"], states["full_compile"]["parameters"]
    )
    hybrid_buffer_diff = difference(states["hybrid"]["buffers"], states["full_compile"]["buffers"])
    hybrid_optimizer_diff = difference(
        states["hybrid"]["optimizer_state"], states["full_compile"]["optimizer_state"]
    )
    hybrid_metric_diff = max(
        abs(results["hybrid"][key] - results["full_compile"][key])
        for key in results["full_compile"]
    )
    hybrid_comparison = {
        "parameters": hybrid_parameter_diff,
        "buffers": hybrid_buffer_diff,
        "optimizer_state": hybrid_optimizer_diff,
        "metric_max_abs": hybrid_metric_diff,
    }
    thresholds = {
        "metric_max_abs": 5.0e-2,
        "parameter_mean_abs": 5.0e-4,
        "buffer_mean_abs": 5.0e-4,
        "optimizer_state_mean_abs": 5.0e-5,
    }
    checks = {
        "all_metrics_finite": all(
            math.isfinite(value) for metrics in results.values() for value in metrics.values()
        ),
        "metric_outputs_not_overwritten": all(
            metrics["reward_scale_std"] > 0.0 and metrics["temperature"] > 0.0
            for metrics in results.values()
        ),
        "optimizer_steps_match": all(
            bool(counter["optimizer_steps"])
            and all(step == float(args.steps) for step in counter["optimizer_steps"])
            for counter in counters.values()
        ),
        "scheduler_steps_match": all(
            all(epoch == args.steps for epoch in counter["scheduler_epochs"])
            for counter in counters.values()
        ),
        "hybrid_metric_close": hybrid_metric_diff <= thresholds["metric_max_abs"],
        "hybrid_parameter_mean_close": hybrid_parameter_diff["mean_abs"]
        <= thresholds["parameter_mean_abs"],
        "hybrid_buffer_mean_close": hybrid_buffer_diff["mean_abs"] <= thresholds["buffer_mean_abs"],
        "hybrid_optimizer_mean_close": hybrid_optimizer_diff["mean_abs"]
        <= thresholds["optimizer_state_mean_abs"],
    }
    payload = {
        "matmul_precision": args.matmul_precision,
        "steps": args.steps,
        "batch_count": args.batch_count,
        "batch_size": args.batch_size,
        "obs_dim": args.obs_dim,
        "critic_obs_dim": args.critic_obs_dim,
        "action_dim": args.action_dim,
        "use_amp": args.use_amp,
        "amp_dtype": args.amp_dtype,
        "metrics": results,
        "update_counters": counters,
        "comparisons_to_eager": comparisons,
        "hybrid_comparison_to_full_compile": hybrid_comparison,
        "validation": {
            "thresholds": thresholds,
            "checks": checks,
            "passed": all(checks.values()),
        },
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.check and not all(checks.values()):
        raise SystemExit("FlashSAC numerical validation failed")


if __name__ == "__main__":
    main()
