"""Solo training entry point — trains all 9 committee agents in parallel.

Each agent gets its own SAC instance, its own env (with its own reward
function), and its own running z-score normalizer. No shared trajectory.

Two usage patterns:

  Workstation (parallel within one process):
      ta-committee-train --episodes 100 --log-dir _results/committee_v0/smoke

  HPC (one (agent, seed) per job-array task):
      ta-committee-train --agents CRPS --seeds 0 --serial \\
          --episodes 5000 --buffer-size 500000 --learning-starts 2000 \\
          --log-dir $SCRATCH/committee_v0/run01

  Multi-seed across several agents on workstation:
      ta-committee-train --agents CRPS NLL --seeds 0 1 2 \\
          --episodes 500 --log-dir _results/committee_v0/run01

Output layout for ``--log-dir <D>``:
    D/models/{agent}.zip                   final model (single-seed mode)
    D/models/{agent}_seed{i}.zip           final model (multi-seed mode)
    D/models/{agent}.normalizer.npz        running z-score stats
    D/checkpoints/{agent}/step_*.zip       periodic checkpoints (if --ckpt-freq)
    D/tb/{agent}_seed{i}/                  TensorBoard event files (if --tensorboard)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from traits_audit.committee.agents import CommitteeAgent, make_agent
from traits_audit.committee.env import DEFAULT_EPISODE_LENGTH
from traits_audit.committee.rewards import REWARD_REGISTRY


def train_one_agent(
    name: str,
    total_timesteps: int,
    models_dir: Path,
    seed: int,
    tensorboard_log: Optional[str],
    suffix: str = "",
    sac_kwargs: Optional[dict] = None,
    ckpt_freq: int = 0,
    ckpt_dir: Optional[Path] = None,
) -> tuple[str, Path]:
    """Worker function: build, train, and save a single agent.

    Parameters
    ----------
    name : str
        Agent registry key (e.g. ``"CRPS"``).
    total_timesteps : int
        SAC ``learn`` budget.
    models_dir : Path
        Where the final ``.zip`` lands.
    seed : int
        Per-job seed; controls env reset, SAC init, and the random warm-start.
    tensorboard_log : str, optional
        Per-(agent, seed) TB directory (set to ``f"{root_tb}/{name}{suffix}"``
        by ``main``).
    suffix : str
        Appended to the saved filename (e.g. ``"_seed3"``). Empty in
        single-seed mode for backwards compatibility.
    sac_kwargs : dict
        Forwarded to ``CommitteeAgent.build_model``.
    ckpt_freq : int
        If > 0, save a checkpoint every ``ckpt_freq`` timesteps.
    ckpt_dir : Path, optional
        Checkpoint root. Required if ``ckpt_freq > 0``.

    Designed to be called via ProcessPoolExecutor; instantiates SB3 lazily
    so we don't pickle SAC across the process boundary.
    """
    agent = make_agent(
        name=name,
        seed=seed,
        tensorboard_log=tensorboard_log,
    )
    agent.build_model(verbose=0, **(sac_kwargs or {}))

    callback = None
    if ckpt_freq > 0 and ckpt_dir is not None:
        from stable_baselines3.common.callbacks import CheckpointCallback
        save_root = ckpt_dir / f"{name}{suffix}"
        save_root.mkdir(parents=True, exist_ok=True)
        callback = CheckpointCallback(
            save_freq=ckpt_freq,
            save_path=str(save_root),
            name_prefix="step",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

    agent.learn(
        total_timesteps=total_timesteps,
        progress_bar=False,
        callback=callback,
    )

    save_name = name + suffix
    agent_for_save = CommitteeAgent(
        name=save_name,
        reward_computer=agent.reward_computer,
        env=agent.env,
        normalizer=agent.normalizer,
        model=agent.model,
        seed=seed,
        tensorboard_log=tensorboard_log,
    )
    agent_for_save.save(models_dir)
    return save_name, models_dir / f"{save_name}.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes per agent (default: 100 = smoke run; "
             "full run uses 500–10000).",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=DEFAULT_EPISODE_LENGTH,
        help="Steps per episode (default: 100).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("_results/committee_v0/run"),
        help="Run root. Models land at {log-dir}/models/, checkpoints at "
             "{log-dir}/checkpoints/, TensorBoard at {log-dir}/tb/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="DEPRECATED — use --log-dir. If provided, models save directly "
             "into --output-dir/ with no models/ subdir (legacy v0 layout).",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Skip TensorBoard logging (default: enabled at {log-dir}/tb/).",
    )
    parser.add_argument(
        "--ckpt-freq",
        type=int,
        default=0,
        help="Save a checkpoint every N timesteps (default: 0 = disabled). "
             "Useful for long HPC runs where you may want intermediate models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Single base seed (default: 0). Ignored if --seeds is given.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Multi-seed mode: train each agent once per seed. Outputs named "
             "{agent}_seed{i}.zip. Overrides --seed.",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        choices=list(REWARD_REGISTRY),
        help="Subset of agents to train (default: all 9).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel workers (default: min(n_jobs, cpu_count)).",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Train serially in this process (useful for debugging and HPC "
             "job-array mode where each task is a single (agent, seed)).",
    )
    # -- SAC hyperparams (defaults sized for workstation smoke runs;
    #    bump for HPC-scale via flags). -----------------------------
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000,
                        help="SAC replay buffer (default: 100k; bump to 500k+ for HPC).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=200,
                        help="SAC random-action warmup (default: 200; bump to 2k for HPC).")
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    selected = args.agents if args.agents else list(REWARD_REGISTRY)
    total_timesteps = args.episodes * args.episode_length

    # -- Output layout. Two paths: new --log-dir (recommended) vs legacy
    #    --output-dir (kept for backwards compat with the workstation smoke).
    if args.output_dir is not None:
        models_dir = args.output_dir
        ckpt_dir = None
        tb_root = None
    else:
        models_dir = args.log_dir / "models"
        ckpt_dir = args.log_dir / "checkpoints" if args.ckpt_freq > 0 else None
        tb_root = None if args.no_tensorboard else str(args.log_dir / "tb")

    models_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if tb_root is not None:
        Path(tb_root).mkdir(parents=True, exist_ok=True)

    # Build job list: each entry is (agent_name, seed, save_suffix).
    if args.seeds is not None:
        jobs = [
            (name, seed, f"_seed{seed}")
            for name in selected
            for seed in args.seeds
        ]
        seed_msg = f"seeds={args.seeds}"
    else:
        jobs = [
            (name, args.seed + i, "")
            for i, name in enumerate(selected)
        ]
        seed_msg = f"base seed={args.seed} (single-seed mode)"

    sac_kwargs = dict(
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        gamma=args.gamma,
    )

    print(f"Training {len(jobs)} jobs ({len(selected)} agents × "
          f"{len(args.seeds) if args.seeds else 1} seed(s)) · "
          f"{args.episodes} episodes × {args.episode_length} steps = "
          f"{total_timesteps:,} timesteps each")
    print(f"  {seed_msg}")
    print(f"  SAC: lr={args.learning_rate}, buffer={args.buffer_size:,}, "
          f"batch={args.batch_size}, learning_starts={args.learning_starts}, "
          f"gamma={args.gamma}")
    print(f"  Models: {models_dir}")
    if ckpt_dir is not None:
        print(f"  Checkpoints: {ckpt_dir} (every {args.ckpt_freq:,} steps)")
    if tb_root is not None:
        print(f"  TensorBoard: {tb_root}")
    print()

    def tb_for(name: str, suffix: str) -> Optional[str]:
        if tb_root is None:
            return None
        return str(Path(tb_root) / f"{name}{suffix}")

    if args.serial:
        for i, (name, seed, suffix) in enumerate(jobs):
            label = f"{name}{suffix}"
            print(f"  [{i+1}/{len(jobs)}] training {label} (seed={seed}) ...")
            train_one_agent(
                name=name,
                total_timesteps=total_timesteps,
                models_dir=models_dir,
                seed=seed,
                tensorboard_log=tb_for(name, suffix),
                suffix=suffix,
                sac_kwargs=sac_kwargs,
                ckpt_freq=args.ckpt_freq,
                ckpt_dir=ckpt_dir,
            )
            print(f"  [{i+1}/{len(jobs)}] {label} saved")
        return

    max_workers = args.max_workers or min(len(jobs), 8)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                train_one_agent,
                name=name,
                total_timesteps=total_timesteps,
                models_dir=models_dir,
                seed=seed,
                tensorboard_log=tb_for(name, suffix),
                suffix=suffix,
                sac_kwargs=sac_kwargs,
                ckpt_freq=args.ckpt_freq,
                ckpt_dir=ckpt_dir,
            ): f"{name}{suffix}"
            for (name, seed, suffix) in jobs
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                trained_name, path = future.result()
                print(f"  ✓ {trained_name} → {path}")
            except Exception as exc:
                print(f"  ✗ {label} failed: {exc!r}")
                raise

    print(f"\nAll {len(jobs)} jobs trained. Models in {models_dir}")


if __name__ == "__main__":
    main()
