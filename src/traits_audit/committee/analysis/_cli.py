"""``ta-committee-analyze`` — post-training analyses for committee v0.

Subcommands:

    corr-random
        9x9 reward correlation matrix on uniform-random rollouts.
        No trained models needed — fast sanity check / baseline.

    corr-trained
        Same matrix but on rollouts from each trained policy (all 9 agents
        x N seeds). The "9 wearing 3-4 costumes" test lives here.

    density
        Headline 9-panel query-density figure vs predicted_styles.md.

    regret
        Simple-regret curves + paired Wilcoxon test vs best-solo.

Default output directory: ``_results/committee_v0/`` to live alongside
``predicted_styles.md``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _add_corr_random(sub):
    p = sub.add_parser(
        "corr-random",
        help="9x9 reward correlation matrix on uniform-random rollouts.",
    )
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--episode-length", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path,
                   default=Path("_results/committee_v0"))
    p.add_argument("--cluster-threshold", type=float, default=0.5,
                   help="1 - |rho| distance threshold for clustering.")


def _add_corr_trained(sub):
    p = sub.add_parser(
        "corr-trained",
        help="9x9 reward correlation matrix on trained-policy rollouts.",
    )
    p.add_argument("--models-dir", type=Path,
                   default=Path("runs/committee_v0_5M/models"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-episodes-per-seed", type=int, default=50,
                   help="Plan calls for 50 episodes per (agent, seed).")
    p.add_argument("--episode-length", type=int, default=100)
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for episode seed selection.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("_results/committee_v0"))
    p.add_argument("--cluster-threshold", type=float, default=0.5)


def _add_density(sub):
    p = sub.add_parser(
        "density",
        help="Per-agent query density figure (the headline result).",
    )
    p.add_argument("--models-dir", type=Path,
                   default=Path("runs/committee_v0_5M/models"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-episodes-per-seed", type=int, default=50)
    p.add_argument("--episode-length", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bins", type=int, default=30)
    p.add_argument("--output-dir", type=Path,
                   default=Path("_results/committee_v0"))


def _add_regret(sub):
    p = sub.add_parser(
        "regret",
        help="Simple-regret curves + paired Wilcoxon test vs best-solo.",
    )
    p.add_argument("--models-dir", type=Path,
                   default=Path("runs/committee_v0_5M/models"))
    p.add_argument("--committee-solo-seed", type=int, default=0,
                   help="Which training seed to use for committee + solos.")
    p.add_argument("--n-episode-seeds", type=int, default=20)
    p.add_argument("--episode-length", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path,
                   default=Path("_results/committee_v0"))


def _add_thread_b(sub):
    p = sub.add_parser(
        "thread-b",
        help="Votes-as-features bake-off: LCB/max-sigma +/- committee votes.",
    )
    p.add_argument("--models-dir", type=Path,
                   default=Path("runs/committee_v0_5M/models"))
    p.add_argument("--committee-solo-seed", type=int, default=0)
    p.add_argument("--n-episode-seeds", type=int, default=20)
    p.add_argument("--episode-length", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vote-weight", type=float, default=1.0)
    p.add_argument("--output-dir", type=Path,
                   default=Path("_results/committee_v1_threadB"))
    p.add_argument("--skip-ablation", action="store_true",
                   help="Skip leave-one-out ablation (9x extra rollouts).")


def _add_learning_curves(sub):
    p = sub.add_parser(
        "learning-curves",
        help="3x3 SAC training-reward curves for the 9 committee members.",
    )
    p.add_argument("--tb-dir", type=Path,
                   default=Path("runs/committee_v0_5M/tb"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--tag", type=str, default="rollout/ep_rew_mean")
    p.add_argument("--smoothing", type=float, default=0.6,
                   help="EMA alpha; TB default is 0.6.")
    p.add_argument("--n-grid", type=int, default=500)
    p.add_argument("--output-dir", type=Path,
                   default=Path("_results/committee_v0"))


def _add_thread_a(sub):
    p = sub.add_parser(
        "thread-a",
        help="QBC aggregator bake-off vs the v0 uniform-pick committee.",
    )
    p.add_argument("--models-dir", type=Path,
                   default=Path("runs/committee_v0_5M/models"))
    p.add_argument("--correlation-csv", type=Path,
                   default=Path("_results/committee_v0/correlation_trained.csv"))
    p.add_argument("--regret-json", type=Path,
                   default=Path("_results/committee_v0/regret_test.json"))
    p.add_argument("--committee-solo-seed", type=int, default=0)
    p.add_argument("--n-episode-seeds", type=int, default=20)
    p.add_argument("--episode-length", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path,
                   default=Path("_results/committee_v1_threadA"))


def _run_corr_random(args) -> None:
    from traits_audit.committee.analysis.correlation import (
        cluster_agents,
        random_rollout_correlation,
        render_heatmap,
        write_csv,
    )

    print(f"[corr-random] {args.n_episodes} episodes x {args.episode_length} steps "
          f"(seed={args.seed})")
    result = random_rollout_correlation(
        n_episodes=args.n_episodes,
        episode_length=args.episode_length,
        seed=args.seed,
    )
    labels, n_clusters = cluster_agents(result.matrix, threshold=args.cluster_threshold)
    print(f"[corr-random] clusters at threshold={args.cluster_threshold}: "
          f"{n_clusters}")
    for cid in sorted(set(labels.tolist())):
        members = [n for n, l in zip(result.agent_names, labels) if l == cid]
        print(f"  cluster {cid}: {', '.join(members)}")

    out = args.output_dir
    write_csv(result, out / "correlation_random.csv")
    render_heatmap(
        result,
        title=f"Reward correlation - random rollouts ({args.n_episodes} eps)",
        output_path=out / "correlation_random.png",
    )
    render_heatmap(
        result,
        title=f"Reward correlation - random rollouts, cluster-ordered",
        output_path=out / "correlation_random_clustered.png",
        cluster_labels=labels,
    )
    np.save(out / "correlation_random_labels.npy", labels)
    print(f"[corr-random] wrote outputs to {out}/")


def _run_corr_trained(args) -> None:
    from traits_audit.committee.analysis.correlation import (
        cluster_agents,
        render_heatmap,
        trained_policy_correlation,
        write_csv,
    )

    print(f"[corr-trained] models={args.models_dir} seeds={args.seeds} "
          f"{args.n_episodes_per_seed} ep/seed x {args.episode_length} steps")
    result = trained_policy_correlation(
        models_dir=args.models_dir,
        seeds=args.seeds,
        n_episodes_per_seed=args.n_episodes_per_seed,
        episode_length=args.episode_length,
        rng_seed=args.seed,
    )
    labels, n_clusters = cluster_agents(result.matrix, threshold=args.cluster_threshold)
    print(f"[corr-trained] clusters at threshold={args.cluster_threshold}: "
          f"{n_clusters}")
    for cid in sorted(set(labels.tolist())):
        members = [n for n, l in zip(result.agent_names, labels) if l == cid]
        print(f"  cluster {cid}: {', '.join(members)}")

    out = args.output_dir
    write_csv(result, out / "correlation_trained.csv")
    render_heatmap(
        result,
        title=f"Reward correlation - trained policies ({result.n_episodes} eps)",
        output_path=out / "correlation_trained.png",
    )
    render_heatmap(
        result,
        title=f"Reward correlation - trained policies, cluster-ordered",
        output_path=out / "correlation_trained_clustered.png",
        cluster_labels=labels,
    )
    np.save(out / "correlation_trained_labels.npy", labels)
    print(f"[corr-trained] wrote outputs to {out}/")


def _run_density(args) -> None:
    from traits_audit.committee.analysis.density import (
        render_headline_figure,
        run_density_rollouts,
        write_density_csv,
    )

    print(f"[density] models={args.models_dir} seeds={args.seeds} "
          f"{args.n_episodes_per_seed} ep/seed x {args.episode_length} steps")
    result = run_density_rollouts(
        models_dir=args.models_dir,
        seeds=args.seeds,
        n_episodes_per_seed=args.n_episodes_per_seed,
        episode_length=args.episode_length,
        rng_seed=args.seed,
    )
    out = args.output_dir
    write_density_csv(result, out / "query_density.csv")
    render_headline_figure(
        result,
        output_path=out / "query_density_headline.png",
        n_bins=args.n_bins,
    )
    print(f"[density] wrote outputs to {out}/")


def _run_regret(args) -> None:
    import json
    from traits_audit.committee.analysis.regret import (
        paired_test,
        render_regret_figure,
        run_regret,
        write_regret_csv,
    )

    print(f"[regret] models={args.models_dir} solo-seed={args.committee_solo_seed} "
          f"{args.n_episode_seeds} ep-seeds x {args.episode_length} steps")
    result = run_regret(
        models_dir=args.models_dir,
        seeds=[args.committee_solo_seed],
        n_episode_seeds=args.n_episode_seeds,
        episode_length=args.episode_length,
        rng_seed=args.seed,
        committee_solo_seed=args.committee_solo_seed,
    )
    test = paired_test(result)
    print(f"[regret] committee vs best-solo ({test['best_solo']}): "
          f"committee={test['committee_mean']:.4f}  "
          f"best-solo={test['best_solo_mean']:.4f}  "
          f"p={test['p_value']:.4f}")

    out = args.output_dir
    write_regret_csv(result, out / "regret.csv")
    render_regret_figure(result, out / "regret.png")
    (out / "regret_test.json").write_text(json.dumps(test, indent=2) + "\n")
    print(f"[regret] wrote outputs to {out}/")


def _run_learning_curves(args) -> None:
    from traits_audit.committee.analysis.learning_curves import (
        load_learning_curves, render_learning_curves_figure,
        write_learning_curves_csv,
    )

    print(f"[learning-curves] tb={args.tb_dir} seeds={args.seeds} tag={args.tag}")
    result = load_learning_curves(
        tb_dir=args.tb_dir, seeds=args.seeds,
        tag=args.tag, smoothing_alpha=args.smoothing, n_grid=args.n_grid,
    )
    out = args.output_dir
    render_learning_curves_figure(
        result, output_path=out / "learning_curves.png",
        title_tag=args.tag,
    )
    write_learning_curves_csv(result, out / "learning_curves.csv")
    print(f"[learning-curves] wrote outputs to {out}/")


def _run_thread_b(args) -> None:
    import json
    from traits_audit.committee.analysis.thread_regret import (
        paired_terminal_test, run_thread_b, run_thread_b_ablation,
        write_thread_csv,
    )
    from traits_audit.committee.analysis.thread_figures import (
        render_ablation, render_b1,
    )

    print(f"[thread-b] models={args.models_dir} solo-seed={args.committee_solo_seed} "
          f"{args.n_episode_seeds} ep-seeds x {args.episode_length} steps "
          f"vote_weight={args.vote_weight}")
    result, voter = run_thread_b(
        models_dir=args.models_dir,
        n_episode_seeds=args.n_episode_seeds,
        episode_length=args.episode_length,
        rng_seed=args.seed,
        committee_solo_seed=args.committee_solo_seed,
        vote_weight=args.vote_weight,
    )
    out = args.output_dir
    write_thread_csv(result, out / "thread_b_regret.csv")
    tests = render_b1(result, out / "b1_regret_paired.png")
    (out / "thread_b_tests.json").write_text(json.dumps(tests, indent=2) + "\n")
    print(f"[thread-b] LCB+votes vs LCB: p={tests['LCB+votes_vs_LCB']['p_value']:.2e}")
    print(f"[thread-b] max-sigma+votes vs max-sigma: "
          f"p={tests['MaxSigma+votes_vs_MaxSigma']['p_value']:.2e}")

    if not args.skip_ablation:
        for target_policy, out_name, csv_name in [
            ("LCB+votes",       "b2_ablation_lcb.png",       "thread_b_ablation_lcb.csv"),
            ("max-sigma+votes", "b2_ablation_maxsigma.png",  "thread_b_ablation_maxsigma.csv"),
        ]:
            print(f"[thread-b] running leave-one-out ablation for {target_policy} ...")
            ablation = run_thread_b_ablation(
                voter,
                n_episode_seeds=args.n_episode_seeds,
                episode_length=args.episode_length,
                rng_seed=args.seed,
                vote_weight=args.vote_weight,
                policy=target_policy,
            )
            baseline_terminal = result.per_policy_regret[target_policy][:, -1]
            render_ablation(
                ablation, baseline_terminal,
                agent_names=voter.agent_names,
                output_path=out / out_name,
                headline=target_policy,
            )
            ablation_rows = ["dropped_agent,episode_seed,terminal_regret"]
            for name, arr in ablation.items():
                for es, v in zip(result.seeds, arr):
                    ablation_rows.append(f"{name},{es},{v:.6f}")
            (out / csv_name).write_text("\n".join(ablation_rows) + "\n")
    print(f"[thread-b] wrote outputs to {out}/")


def _run_thread_a(args) -> None:
    import json
    from traits_audit.committee.analysis.thread_regret import (
        run_thread_a, write_thread_csv,
    )
    from traits_audit.committee.analysis.thread_figures import (
        render_a1, render_a2, render_a3,
    )

    print(f"[thread-a] models={args.models_dir} corr={args.correlation_csv} "
          f"regret={args.regret_json} {args.n_episode_seeds} ep-seeds")
    result, voter, indep_w, invreg_w = run_thread_a(
        models_dir=args.models_dir,
        correlation_csv=args.correlation_csv,
        regret_json=args.regret_json,
        n_episode_seeds=args.n_episode_seeds,
        episode_length=args.episode_length,
        rng_seed=args.seed,
        committee_solo_seed=args.committee_solo_seed,
    )
    out = args.output_dir
    write_thread_csv(result, out / "thread_a_regret.csv")
    tests = render_a1(result, out / "a1_aggregator_bakeoff.png")
    (out / "thread_a_tests.json").write_text(json.dumps(tests, indent=2) + "\n")

    # Pull solo terminal SR from the same regret_json used for inv-reg weights.
    solo_data = json.loads(args.regret_json.read_text())["solo_means"]
    solo_terminal = {a: float(solo_data[f"solo:{a}"]) for a in voter.agent_names}
    render_a2(indep_w, invreg_w, solo_terminal, out / "a2_weight_vs_regret.png")
    render_a3(result, out / "a3_disagreement.png")

    weights_payload = {
        "independence": indep_w,
        "inverse_regret": invreg_w,
    }
    (out / "thread_a_weights.json").write_text(
        json.dumps(weights_payload, indent=2) + "\n"
    )

    print("[thread-a] significance vs best-solo:")
    for name, t in tests.items():
        print(f"  {name}: mean={t['a_mean']:.4f}  p={t['p_value']:.2e}")
    print(f"[thread-a] wrote outputs to {out}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ta-committee-analyze",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_corr_random(sub)
    _add_corr_trained(sub)
    _add_density(sub)
    _add_regret(sub)
    _add_thread_b(sub)
    _add_thread_a(sub)
    _add_learning_curves(sub)

    args = parser.parse_args()
    if args.cmd == "corr-random":
        _run_corr_random(args)
    elif args.cmd == "corr-trained":
        _run_corr_trained(args)
    elif args.cmd == "density":
        _run_density(args)
    elif args.cmd == "regret":
        _run_regret(args)
    elif args.cmd == "thread-b":
        _run_thread_b(args)
    elif args.cmd == "thread-a":
        _run_thread_a(args)
    elif args.cmd == "learning-curves":
        _run_learning_curves(args)
    else:
        parser.error(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
