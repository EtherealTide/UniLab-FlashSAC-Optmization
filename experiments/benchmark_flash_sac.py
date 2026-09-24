"""Reproducible FlashSAC eager/deferred/CUDA-Graph microbenchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uni_rl.algos.flash_sac.learner import FlashSACLearner


def make_batch(
    batch_size: int,
    device: str,
    *,
    obs_dim: int,
    critic_obs_dim: int,
    action_dim: int,
) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randn(batch_size, obs_dim, device=device),
        "critic": torch.randn(batch_size, critic_obs_dim, device=device),
        "actions": torch.tanh(torch.randn(batch_size, action_dim, device=device)),
        "rewards": torch.randn(batch_size, device=device),
        "next_obs": torch.randn(batch_size, obs_dim, device=device),
        "next_critic": torch.randn(batch_size, critic_obs_dim, device=device),
        "dones": torch.zeros(batch_size, device=device),
        "truncated": torch.zeros(batch_size, device=device),
    }


def run_case(
    *,
    device: str,
    deferred: bool,
    cuda_graphs: bool,
    compile_methods: bool,
    compile_full_objectives: bool,
    rounds: int,
    updates: int,
    warmup: int,
    batch_size: int,
    actor_hidden_dim: int,
    critic_hidden_dim: int,
    actor_num_blocks: int,
    critic_num_blocks: int,
    num_atoms: int,
    obs_dim: int,
    critic_obs_dim: int,
    action_dim: int,
    policy_frequency: int,
    use_amp: bool,
    amp_dtype: str,
) -> dict[str, object]:
    learner = FlashSACLearner(
        obs_dim=obs_dim,
        action_dim=action_dim,
        critic_obs_dim=critic_obs_dim,
        actor_hidden_dim=actor_hidden_dim,
        critic_hidden_dim=critic_hidden_dim,
        actor_num_blocks=actor_num_blocks,
        critic_num_blocks=critic_num_blocks,
        num_atoms=num_atoms,
        device=device,
        use_compile=compile_methods,
        compile_full_objectives=compile_full_objectives,
        use_cuda_graph_critic=cuda_graphs,
        use_cuda_graph_actor=cuda_graphs,
        use_cuda_graph_critic_packed_staging=cuda_graphs,
        use_cuda_graph_actor_packed_staging=cuda_graphs,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    batches = [
        make_batch(
            batch_size,
            device,
            obs_dim=obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
        )
        for _ in range(updates)
    ]

    def one_round(read_metrics: bool) -> None:
        actor_updated = False
        for update_idx, batch in enumerate(batches):
            critic_read_metrics = read_metrics or update_idx == updates - 1
            learner.update_critic_cuda_graph(batch, read_metrics=critic_read_metrics)
            if update_idx % policy_frequency == 0:
                learner.update_actor_cuda_graph(batch, read_metrics=read_metrics)
                actor_updated = True
            if not learner.cuda_graph_critic_captures_target_update:
                learner.soft_update_target()
        if not read_metrics and actor_updated:
            learner.read_deferred_actor_metrics()
        if device == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):
        one_round(not deferred)
    if device == "cuda":
        torch.cuda.synchronize()
    samples_ms: list[float] = []
    for _ in range(rounds):
        start = time.perf_counter()
        one_round(not deferred)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    sorted_samples = sorted(samples_ms)

    def percentile(q: float) -> float:
        index = (len(sorted_samples) - 1) * q
        lower = int(index)
        upper = min(lower + 1, len(sorted_samples) - 1)
        fraction = index - lower
        return sorted_samples[lower] + fraction * (sorted_samples[upper] - sorted_samples[lower])

    mean_ms = sum(samples_ms) / len(samples_ms)
    return {
        "device": device,
        "deferred_metrics": deferred,
        "cuda_graphs": cuda_graphs,
        "compile": compile_methods,
        "compile_full_objectives": compile_full_objectives,
        "rounds": rounds,
        "updates": updates,
        "batch_size": batch_size,
        "actor_hidden_dim": actor_hidden_dim,
        "critic_hidden_dim": critic_hidden_dim,
        "actor_num_blocks": actor_num_blocks,
        "critic_num_blocks": critic_num_blocks,
        "num_atoms": num_atoms,
        "obs_dim": obs_dim,
        "critic_obs_dim": critic_obs_dim,
        "action_dim": action_dim,
        "policy_frequency": policy_frequency,
        "use_amp": use_amp,
        "amp_dtype": amp_dtype,
        # ``ms_per_round`` remains the mean for compatibility with older JSON.
        "ms_per_round": mean_ms,
        "mean_ms_per_round": mean_ms,
        "median_ms_per_round": percentile(0.50),
        "p90_ms_per_round": percentile(0.90),
        "p95_ms_per_round": percentile(0.95),
        "samples_ms_per_round": samples_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--policy-frequency", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--actor-hidden-dim", type=int, default=128)
    parser.add_argument("--critic-hidden-dim", type=int, default=256)
    parser.add_argument("--actor-num-blocks", type=int, default=2)
    parser.add_argument("--critic-num-blocks", type=int, default=2)
    parser.add_argument("--num-atoms", type=int, default=101)
    parser.add_argument("--obs-dim", type=int, default=98)
    parser.add_argument("--critic-obs-dim", type=int, default=101)
    parser.add_argument("--action-dim", type=int, default=29)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("auto", "fp16", "bf16"), default="auto")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    torch.set_float32_matmul_precision(args.matmul_precision)

    network_kwargs = {
        "actor_hidden_dim": args.actor_hidden_dim,
        "critic_hidden_dim": args.critic_hidden_dim,
        "actor_num_blocks": args.actor_num_blocks,
        "critic_num_blocks": args.critic_num_blocks,
        "num_atoms": args.num_atoms,
        "obs_dim": args.obs_dim,
        "critic_obs_dim": args.critic_obs_dim,
        "action_dim": args.action_dim,
        "policy_frequency": args.policy_frequency,
        "use_amp": args.use_amp,
        "amp_dtype": args.amp_dtype,
    }

    results = [
        run_case(
            device=args.device,
            deferred=False,
            cuda_graphs=False,
            compile_methods=False,
            compile_full_objectives=False,
            rounds=args.rounds,
            updates=args.updates,
            warmup=args.warmup,
            batch_size=args.batch_size,
            **network_kwargs,
        ),
        run_case(
            device=args.device,
            deferred=True,
            cuda_graphs=False,
            compile_methods=False,
            compile_full_objectives=False,
            rounds=args.rounds,
            updates=args.updates,
            warmup=args.warmup,
            batch_size=args.batch_size,
            **network_kwargs,
        ),
    ]
    if args.device == "cuda":
        results.extend(
            [
                run_case(
                    device=args.device,
                    deferred=True,
                    cuda_graphs=False,
                    compile_methods=True,
                    compile_full_objectives=False,
                    rounds=args.rounds,
                    updates=args.updates,
                    warmup=args.warmup,
                    batch_size=args.batch_size,
                    **network_kwargs,
                ),
                run_case(
                    device=args.device,
                    deferred=True,
                    cuda_graphs=False,
                    compile_methods=True,
                    compile_full_objectives=True,
                    rounds=args.rounds,
                    updates=args.updates,
                    warmup=args.warmup,
                    batch_size=args.batch_size,
                    **network_kwargs,
                ),
                run_case(
                    device=args.device,
                    deferred=True,
                    cuda_graphs=True,
                    compile_methods=False,
                    compile_full_objectives=False,
                    rounds=args.rounds,
                    updates=args.updates,
                    warmup=args.warmup,
                    batch_size=args.batch_size,
                    **network_kwargs,
                ),
                run_case(
                    device=args.device,
                    deferred=True,
                    cuda_graphs=True,
                    compile_methods=True,
                    compile_full_objectives=False,
                    rounds=args.rounds,
                    updates=args.updates,
                    warmup=args.warmup,
                    batch_size=args.batch_size,
                    **network_kwargs,
                ),
                run_case(
                    device=args.device,
                    deferred=True,
                    cuda_graphs=True,
                    compile_methods=True,
                    compile_full_objectives=True,
                    rounds=args.rounds,
                    updates=args.updates,
                    warmup=args.warmup,
                    batch_size=args.batch_size,
                    **network_kwargs,
                ),
            ]
        )
    payload = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "matmul_precision": args.matmul_precision,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
