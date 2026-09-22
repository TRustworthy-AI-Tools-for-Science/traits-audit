"""``ta-committee-analyze`` — post-training analyses for committee v0.

Subcommands:

    corr-random
        KxK reward correlation matrix on uniform-random rollouts.
        No trained models needed — fast sanity check / baseline.

    corr-trained
        Same matrix but on rollouts from each trained policy (all K agents
        x N seeds). The "K wearing 3-4 costumes" test lives here.

    density
        Headline per-agent query-density figure vs predicted_styles.md
        (Forrester only) or per-dimension marginals (other problems).

    regret
        Simple-regret curves + paired Wilcoxon test vs best-solo.

Every subcommand except ``learning-curves`` (which just reads TensorBoard
scalars, and so is problem-agnostic anyway) takes ``--problem`` (default:
``forrester``; see ``committee/problems.py`` for the others) to run on a
different benchmark. It must match what ``--models-dir`` was trained on.

Default output directory: ``_results/committee_v0/`` to live alongside
``predicted_styles.md``. For another problem, pass a different
``--output-dir`` (e.g. ``_results/committee_branin-currin``) — nothing here
does that for you automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from traits_audit.committee.problems import PROBLEMS


def _add_corr_random(sub):
    p = sub.add_parser(
        "corr-random",
        help="KxK reward correlation matrix on uniform-random rollouts.",
    )
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--episode-length", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--problem", choices=list(PROBLEMS), default="forrester")
    p.add_argument("--output-dir", type=Path,
                   default=Path("_results/committee_v0"))
    p.add_argument("--cluster-threshold", type=float, default=0.5,
                   help="1 - |rho| distance threshold for clustering.")


def _add_corr_trained(sub):
    p = sub.add_parser(
        "corr-trained",
        help="KxK reward correlation matrix on trained-policy rollouts.",
    )
    p.add_argument("--models-dir", type=Path,
                   default=Path("runs/committee_v0_5M/models"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-episodes-per-seed", type=int, default=50,
                   help="Plan calls for 50 episodes per (agent, seed).")
    p.add_argument("--episode-length", type=int, default=100)
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for episode seed selection.")
    p.add_argument("--problem", choices=list(PROBLEMS), default="forrester",
                   help="Must match what --models-dir was trained on.")
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
    p.add_argument("--problem", choices=list(PROBLEMS), default="forrester",
                   help="Must match what --models-dir was trained on.")
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
    p.add_argument("--problem", choices=list(PROBLEMS), default="forrester",
                   help="Must match what --models-dir was trained on.")
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
    p.add_argument("--problem", choices=list(PROBLEMS), default="forrester",
                   help="Must match what --models-dir was trained on.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("_results/committee_v1_threadB"))
    p.add_argument("--skip-ablation", action="store_true",
                   help="Skip leave-one-out ablation (15x extra rollouts).")
    p.add_argument("--only", nargs="+", metavar="POLICY",
                   help="Run only these policies (one shard of an array job).")
    p.add_argument("--ablate-agents", nargs="+", metavar="AGENT",
                   help="Ablate only these agents (one shard of an array job).")
    p.add_argument("--shard-tag", type=str, default="",
                   help="Suffix for this shard's CSVs; also forces shard mode "
                        "(CSV only, no figures). Use `thread-gather` after.")


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
    p.add_argument("--problem", choices=list(PROBLEMS), default="forrester",
                   help="Must match what --models-dir was trained on.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("_results/committee_v1_threadA"))
    p.add_argument("--only", nargs="+", metavar="POLICY",
                   help="Run only these policies (one shard of an array job).")
    p.add_argument("--shard-tag", type=str, default="",
                   help="Suffix for this shard's CSVs; also forces shard mode "
                        "(CSV only, no figures). Use `thread-gather` after.")


def _add_thread_gather(sub):
    p = sub.add_parser(
        "thread-gather",
        help="Merge sharded thread-a/thread-b CSVs and render their figures.",
    )
    p.add_argument("thread", choices=["a", "b"])
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory holding the shard CSVs; figures land here.")
    p.add_argument("--best-solo", type=str, default=None,
                   help="Best-solo agent for thread-a's reference. Defaults to "
                        "the value recorded in thread_a_weights.json.")


def _run_thread_gather(args) -> None:
    """Rebuild a whole-run result from shard CSVs and render the figures.

    The shards already hold every number; this step is pure rendering, so it
    is cheap and needs no trained models.
    """
    import json
    from traits_audit.committee.analysis.thread_regret import read_thread_csv
    from traits_audit.committee.analysis.thread_figures import (
        render_a1, render_a2, render_a3, render_ablation, render_b1,
    )

    out = args.output_dir
    stem = f"thread_{args.thread}_regret"
    shards = sorted(out.glob(f"{stem}*.csv"))
    if not shards:
        raise SystemExit(f"no {stem}*.csv shards found under {out}/")
    print(f"[gather] merging {len(shards)} shard(s): "
          f"{', '.join(p.name for p in shards)}")
    result = read_thread_csv(shards)
    print(f"[gather] {len(result.per_policy_regret)} policies x "
          f"{len(result.seeds)} seeds x {result.episode_length} steps")

    if args.thread == "b":
        tests = render_b1(result, out / "b1_regret_paired.png")
        (out / "thread_b_tests.json").write_text(json.dumps(tests, indent=2) + "\n")
        for name, t in tests.items():
            print(f"  {name}: p={t['p_value']:.2e}")

        for target_policy, fig_name, csv_stem in ABLATION_TARGETS:
            ab_shards = sorted(out.glob(f"{csv_stem}*.csv"))
            if not ab_shards:
                continue
            # Merge the per-agent terminal regrets across ablation shards.
            merged: dict[str, list[float]] = {}
            for path in ab_shards:
                for line in path.read_text().splitlines()[1:]:
                    agent, _es, v = line.split(",")
                    merged.setdefault(agent, []).append(float(v))
            ablation = {k: np.asarray(v) for k, v in merged.items()}
            render_ablation(
                ablation,
                baseline_terminal=result.per_policy_regret[target_policy][:, -1],
                agent_names=sorted(ablation),
                output_path=out / fig_name,
                headline=target_policy,
            )
            print(f"[gather] {fig_name}: {len(ablation)} agents")
    else:
        weights_path = out / "thread_a_weights.json"
        payload = json.loads(weights_path.read_text()) if weights_path.exists() else {}
        best_solo = args.best_solo or payload.get("best_solo")
        if best_solo is None:
            raise SystemExit(
                "thread-a gather needs --best-solo (no thread_a_weights.json)"
            )
        reference = f"best-solo:{best_solo}"
        tests = render_a1(result, out / "a1_aggregator_bakeoff.png",
                          reference=reference)
        (out / "thread_a_tests.json").write_text(json.dumps(tests, indent=2) + "\n")
        for name, t in tests.items():
            print(f"  {name}: mean={t['a_mean']:.4f}  p={t['p_value']:.2e}")

        indep_w, invreg_w = payload.get("independence"), payload.get("inverse_regret")
        if indep_w and invreg_w:
            # Solo terminal SR = 1/weight, the inverse of how invreg was built.
            solo_terminal = {a: 1.0 / w for a, w in invreg_w.items()}
            render_a2(indep_w, invreg_w, solo_terminal,
                      out / "a2_weight_vs_regret.png")
        render_a3(result, out / "a3_disagreement.png")

    print(f"[gather] wrote figures to {out}/")


def _run_corr_random(args) -> None:
    from traits_audit.committee.analysis.correlation import (
        cluster_agents,
        random_rollout_correlation,
        render_heatmap,
        write_csv,
    )

    print(f"[corr-random] problem={args.problem} {args.n_episodes} episodes x "
          f"{args.episode_length} steps (seed={args.seed})")
    result = random_rollout_correlation(
        n_episodes=args.n_episodes,
        episode_length=args.episode_length,
        seed=args.seed,
        problem=args.problem,
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

    print(f"[corr-trained] problem={args.problem} models={args.models_dir} "
          f"seeds={args.seeds} {args.n_episodes_per_seed} ep/seed x "
          f"{args.episode_length} steps")
    result = trained_policy_correlation(
        models_dir=args.models_dir,
        seeds=args.seeds,
        n_episodes_per_seed=args.n_episodes_per_seed,
        episode_length=args.episode_length,
        rng_seed=args.seed,
        problem=args.problem,
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
    from traits_audit.committee.problems import get_problem

    problem = get_problem(args.problem)
    print(f"[density] problem={args.problem} models={args.models_dir} "
          f"seeds={args.seeds} {args.n_episodes_per_seed} ep/seed x "
          f"{args.episode_length} steps")
    result = run_density_rollouts(
        models_dir=args.models_dir,
        seeds=args.seeds,
        n_episodes_per_seed=args.n_episodes_per_seed,
        episode_length=args.episode_length,
        rng_seed=args.seed,
        problem=problem,
    )
    out = args.output_dir
    write_density_csv(result, out / "query_density.csv", dim=problem.dim)
    render_headline_figure(
        result,
        output_path=out / "query_density_headline.png",
        n_bins=args.n_bins,
        problem=problem,
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

    print(f"[regret] problem={args.problem} models={args.models_dir} "
          f"solo-seed={args.committee_solo_seed} "
          f"{args.n_episode_seeds} ep-seeds x {args.episode_length} steps")
    result = run_regret(
        models_dir=args.models_dir,
        seeds=[args.committee_solo_seed],
        n_episode_seeds=args.n_episode_seeds,
        episode_length=args.episode_length,
        rng_seed=args.seed,
        committee_solo_seed=args.committee_solo_seed,
        problem=args.problem,
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


ABLATION_TARGETS = [
    # (policy, figure name, csv stem)
    ("LCB+votes",       "b2_ablation_lcb.png",      "thread_b_ablation_lcb"),
    ("max-sigma+votes", "b2_ablation_maxsigma.png", "thread_b_ablation_maxsigma"),
]


def _run_thread_b_shard_ablations(args, voter, result, tag, ablate_only) -> None:
    """Leave-one-out ablation for this shard's agents; CSV only, no figure.

    The full-committee baseline each bar is measured against lives in the
    thread_b_regret CSVs, so it is resolved at gather time rather than here
    — a shard that doesn't happen to run the target policy can't know it.
    """
    from traits_audit.committee.analysis.thread_regret import (
        run_thread_b_ablation,
    )

    if args.skip_ablation:
        return
    for target_policy, _fig, csv_stem in ABLATION_TARGETS:
        print(f"[thread-b] ablation shard for {target_policy} "
              f"(agents: {ablate_only or 'all'}) ...")
        ablation = run_thread_b_ablation(
            voter,
            n_episode_seeds=args.n_episode_seeds,
            episode_length=args.episode_length,
            rng_seed=args.seed,
            vote_weight=args.vote_weight,
            policy=target_policy,
            problem=args.problem,
            only_agents=ablate_only,
        )
        rows = ["dropped_agent,episode_seed,terminal_regret"]
        for name, arr in ablation.items():
            for es, v in zip(result.seeds, arr):
                rows.append(f"{name},{es},{v:.6f}")
        (args.output_dir / f"{csv_stem}{tag}.csv").write_text(
            "\n".join(rows) + "\n"
        )


def _run_thread_b(args) -> None:
    import json
    from traits_audit.committee.analysis.thread_regret import (
        paired_terminal_test, run_thread_b, run_thread_b_ablation,
        write_thread_csv,
    )
    from traits_audit.committee.analysis.thread_figures import (
        render_ablation, render_b1,
    )

    print(f"[thread-b] problem={args.problem} models={args.models_dir} "
          f"solo-seed={args.committee_solo_seed} "
          f"{args.n_episode_seeds} ep-seeds x {args.episode_length} steps "
          f"vote_weight={args.vote_weight}")

    only = args.only or None
    ablate_only = args.ablate_agents or None
    # A shard computes a slice and writes only CSVs; `thread-gather` then
    # renders the figures once every shard has landed.
    sharded = bool(only or ablate_only or args.shard_tag)
    tag = f"_{args.shard_tag}" if args.shard_tag else ""

    result, voter = run_thread_b(
        models_dir=args.models_dir,
        n_episode_seeds=args.n_episode_seeds,
        episode_length=args.episode_length,
        rng_seed=args.seed,
        committee_solo_seed=args.committee_solo_seed,
        vote_weight=args.vote_weight,
        problem=args.problem,
        only=only,
    )
    out = args.output_dir
    write_thread_csv(result, out / f"thread_b_regret{tag}.csv")

    if sharded:
        _run_thread_b_shard_ablations(args, voter, result, tag, ablate_only)
        print(f"[thread-b] shard outputs written to {out}/ "
              f"— run `thread-gather` once all shards finish")
        return

    tests = render_b1(result, out / "b1_regret_paired.png")
    (out / "thread_b_tests.json").write_text(json.dumps(tests, indent=2) + "\n")
    print(f"[thread-b] LCB+votes vs LCB: p={tests['LCB+votes_vs_LCB']['p_value']:.2e}")
    print(f"[thread-b] max-sigma+votes vs max-sigma: "
          f"p={tests['MaxSigma+votes_vs_MaxSigma']['p_value']:.2e}")

    if not args.skip_ablation:
        for target_policy, out_name, csv_stem in ABLATION_TARGETS:
            csv_name = f"{csv_stem}.csv"
            print(f"[thread-b] running leave-one-out ablation for {target_policy} ...")
            ablation = run_thread_b_ablation(
                voter,
                n_episode_seeds=args.n_episode_seeds,
                episode_length=args.episode_length,
                rng_seed=args.seed,
                vote_weight=args.vote_weight,
                policy=target_policy,
                problem=args.problem,
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

    print(f"[thread-a] problem={args.problem} models={args.models_dir} "
          f"corr={args.correlation_csv} "
          f"regret={args.regret_json} {args.n_episode_seeds} ep-seeds")
    only = args.only or None
    tag = f"_{args.shard_tag}" if args.shard_tag else ""
    result, voter, indep_w, invreg_w, best_solo = run_thread_a(
        models_dir=args.models_dir,
        correlation_csv=args.correlation_csv,
        regret_json=args.regret_json,
        n_episode_seeds=args.n_episode_seeds,
        episode_length=args.episode_length,
        rng_seed=args.seed,
        committee_solo_seed=args.committee_solo_seed,
        problem=args.problem,
        only=only,
    )
    out = args.output_dir
    write_thread_csv(result, out / f"thread_a_regret{tag}.csv")

    if only or args.shard_tag:
        # Weights are cheap and identical across shards; write them so the
        # gather step doesn't need the models loaded again.
        (out / "thread_a_weights.json").write_text(json.dumps({
            "independence": indep_w,
            "inverse_regret": invreg_w,
            "best_solo": best_solo,
        }, indent=2) + "\n")
        print(f"[thread-a] shard outputs written to {out}/ "
              f"— run `thread-gather` once all shards finish")
        return

    tests = render_a1(result, out / "a1_aggregator_bakeoff.png",
                      reference=f"best-solo:{best_solo}")
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
    _add_thread_gather(sub)
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
    elif args.cmd == "thread-gather":
        _run_thread_gather(args)
    elif args.cmd == "learning-curves":
        _run_learning_curves(args)
    else:
        parser.error(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
