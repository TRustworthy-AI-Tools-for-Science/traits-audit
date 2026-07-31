"""Committee acquisition demo — runs the 9 trained agents under uniform-pick aggregation.

Loads previously-trained policies from ``--models-dir``, runs one deployment
episode on the Forrester oracle using uniform random pick across agents, and
runs the existing AuditPipeline on the resulting trajectory.

For comparison, also runs an LCB baseline on a matched seed.

Usage:
    ta-committee-demo --models-dir _results/committee_v0/models --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from traits_audit import AuditHook, AuditPipeline
from traits_audit._example import BootstrapSurrogate, oracle as forrester_oracle, lcb
from traits_audit.checks import (
    CalibrationErrorCheck,
    ConformalCoverageCheck,
    CRPSCheck,
    NegativeLogLikelihoodCheck,
    PITUniformityCheck,
    IntervalScoreCheck,
    IntervalCoverageCheck,
    VarianceAlignmentCheck,
    UncertaintyEvolutionCheck,
    UncertaintyAnomalyCheck,
    VarianceErrorCorrelationCheck,
)
from traits_audit.committee.agents import CommitteeAgent, make_agent
from traits_audit.committee.aggregate import uniform_random_pick
from traits_audit.committee.env import CommitteeEnv, DEFAULT_EPISODE_LENGTH
from traits_audit.committee.rewards import REWARD_REGISTRY


def _build_pipeline() -> AuditPipeline:
    """Match the audit pipeline used in the existing Forrester demo."""
    return AuditPipeline(
        checks=[
            CalibrationErrorCheck(threshold=0.15),
            ConformalCoverageCheck(target_coverage=0.9, max_q_ratio=1.5),
            CRPSCheck(),
            NegativeLogLikelihoodCheck(),
            PITUniformityCheck(),
            IntervalScoreCheck(),
            IntervalCoverageCheck(expected_coverage=0.683, tolerance=0.15),
            VarianceAlignmentCheck(tolerance=0.5),
            UncertaintyEvolutionCheck(slope_threshold=-0.05),
            UncertaintyAnomalyCheck(z_threshold=3.0),
            VarianceErrorCorrelationCheck(min_correlation=0.1),
        ],
        verbose=False,
    )


def _load_committee(models_dir: Path, env_kwargs: Optional[dict] = None) -> list[CommitteeAgent]:
    """Load all 9 trained agents. Each gets a fresh inference-only env."""
    agents: list[CommitteeAgent] = []
    for name in REWARD_REGISTRY:
        # Build a fresh env (the loaded model only needs the spaces, not the
        # original training-time env state).
        agent = make_agent(name=name, env_kwargs=env_kwargs)
        from stable_baselines3 import SAC
        model_path = models_dir / f"{name}.zip"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model for {name!r} not found at {model_path}. "
                f"Run `ta-committee-train` first."
            )
        agent.model = SAC.load(str(model_path), env=agent.env)
        agents.append(agent)
    return agents


def _run_committee_episode(
    agents: list[CommitteeAgent],
    seed: int,
    episode_length: int,
) -> tuple[AuditHook, dict]:
    """Run one committee acquisition episode and audit the trajectory.

    Uses the first agent's env as the shared dynamics — all proposals are
    evaluated against the same state. Uniform-pick aggregation selects one
    proposal per step.
    """
    env = agents[0].env  # share dynamics
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 7919)

    pipeline = _build_pipeline()
    hook = AuditHook(pipeline)
    selection_counts = {a.name: 0 for a in agents}
    proposals_per_step: list[np.ndarray] = []

    for _ in range(episode_length):
        x_chosen, selected_idx, all_proposals = uniform_random_pick(
            agents, obs, rng, deterministic=True
        )
        selection_counts[agents[selected_idx].name] += 1
        proposals_per_step.append(all_proposals)

        action = np.array([x_chosen], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        y_q = info["y_q"]
        mu_q = info["mu_q"]
        sigma_q = info["sigma_q"]
        hook.on_step(
            y_true=y_q,
            y_pred_mean=mu_q,
            y_pred_std=sigma_q,
            uncertainty=sigma_q,
            abs_error=abs(y_q - mu_q),
            acquisition_score=float(mu_q - 2.0 * sigma_q),
            dataset_size=float(env.unwrapped.x_obs.size),
        )
        if terminated or truncated:
            break

    hook.on_end()
    info = {
        "selection_counts": selection_counts,
        "proposals_per_step": np.asarray(proposals_per_step),
        "x_obs": env.unwrapped.x_obs.copy(),
        "y_obs": env.unwrapped.y_obs.copy(),
    }
    return hook, info


def _run_lcb_baseline(seed: int, episode_length: int) -> AuditHook:
    """LCB baseline matching the existing demo loop."""
    oracle_rng = np.random.default_rng(seed)
    surrogate_rng = np.random.default_rng(seed + 2**31)
    surrogate = BootstrapSurrogate(
        degree=5, n_estimators=30, std_scale=0.7, rng=surrogate_rng,
    )
    pool = np.linspace(0, 1, 300)
    x_obs = oracle_rng.uniform(0, 1, size=20)
    y_obs = forrester_oracle(x_obs, oracle_rng)
    pipeline = _build_pipeline()
    hook = AuditHook(pipeline)
    for _ in range(episode_length):
        surrogate.fit(x_obs, y_obs)
        mu_pool, sigma_pool = surrogate.predict(pool)
        idx = lcb(mu_pool, sigma_pool)
        x_q = float(pool[idx])
        y_q = float(forrester_oracle(np.array([x_q]), oracle_rng)[0])
        mu_q = float(mu_pool[idx])
        sigma_q = float(sigma_pool[idx])
        x_obs = np.append(x_obs, x_q)
        y_obs = np.append(y_obs, y_q)
        hook.on_step(
            y_true=y_q,
            y_pred_mean=mu_q,
            y_pred_std=sigma_q,
            uncertainty=sigma_q,
            abs_error=abs(y_q - mu_q),
            acquisition_score=float(mu_q - 2.0 * sigma_q),
            dataset_size=float(x_obs.size),
        )
    hook.on_end()
    return hook


def _print_audit_comparison(
    committee_report,
    lcb_report,
    selection_counts: dict[str, int],
) -> None:
    names = [r.name for r in committee_report.results]
    name_w = max(len(n) for n in names) + 2
    col_w = 18
    sep = "─" * name_w + "┼" + "─" * col_w + "┼" + "─" * col_w
    print()
    print("=" * (name_w + col_w * 2 + 2))
    print(" AUDIT COMPARISON · committee vs LCB")
    print("=" * (name_w + col_w * 2 + 2))
    print(" " * name_w + "│" + " committee".ljust(col_w) + "│" + " LCB".ljust(col_w))
    print(sep)
    for i, n in enumerate(names):
        cr = committee_report.results[i]
        lr = lcb_report.results[i]
        def cell(r):
            if r.value is None:
                return ("PASS" if r.passed else "FAIL")
            return f"{'PASS' if r.passed else 'FAIL'} {r.value:.4f}"
        print(f" {n:<{name_w-1}}│ {cell(cr):<{col_w-1}}│ {cell(lr):<{col_w-1}}")
    print(sep)
    print(f" {'Overall':<{name_w-1}}│"
          f" {'PASS' if committee_report.passed else 'FAIL':<{col_w-1}}│"
          f" {'PASS' if lcb_report.passed else 'FAIL':<{col_w-1}}")
    print("=" * (name_w + col_w * 2 + 2))
    print("\nCommittee selection counts:")
    total = sum(selection_counts.values())
    for name, count in sorted(selection_counts.items(), key=lambda kv: -kv[1]):
        frac = count / total if total else 0
        bar = "█" * int(frac * 40)
        print(f"  {name:<24} {count:>4}  {bar} {frac:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("_results/committee_v0/models"),
        help="Where the trained policies live (from ta-committee-train).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the deployment episode + LCB baseline.",
    )
    parser.add_argument(
        "--episode-length", type=int, default=DEFAULT_EPISODE_LENGTH,
        help="Acquisition steps in the deployment episode (default: 100).",
    )
    parser.add_argument(
        "--skip-lcb", action="store_true",
        help="Skip the LCB baseline comparison.",
    )
    args = parser.parse_args()

    if not args.models_dir.exists():
        raise FileNotFoundError(
            f"No models at {args.models_dir} — run `ta-committee-train --episodes 100 "
            f"--output-dir {args.models_dir}` first."
        )

    print(f"Loading 9 agents from {args.models_dir} ...")
    agents = _load_committee(args.models_dir)

    print(f"Running committee deployment episode (seed={args.seed}, "
          f"length={args.episode_length}) ...")
    committee_hook, info = _run_committee_episode(
        agents, seed=args.seed, episode_length=args.episode_length,
    )
    committee_report = committee_hook.report

    lcb_report = None
    if not args.skip_lcb:
        print(f"Running LCB baseline (seed={args.seed}) ...")
        lcb_hook = _run_lcb_baseline(args.seed, args.episode_length)
        lcb_report = lcb_hook.report

    if lcb_report is not None:
        _print_audit_comparison(committee_report, lcb_report, info["selection_counts"])
    else:
        print("\nCommittee audit:")
        for r in committee_report.results:
            label = "PASS" if r.passed else "FAIL"
            val = f" ({r.value:.4f})" if r.value is not None else ""
            print(f"  {r.name:<24} {label}{val}")


if __name__ == "__main__":
    main()
