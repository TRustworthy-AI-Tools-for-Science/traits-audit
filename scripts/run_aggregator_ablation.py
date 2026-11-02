"""Leave-one-out ablation of committee members, per aggregator rule.

The existing Thread B ablation drops a member from a vote-feature vector handed
to a base rule (LCB / max-sigma). This drops a member from the *aggregator
itself*, for each of the five committee rules, so we can see whether any single
member carries a rule's behaviour.

Writes one tidy CSV per (problem, rule):
    _results/<dir>/threadA/thread_a_ablation_<rule>.csv
with the same schema the Thread B ablations use:
    dropped_agent,episode_seed,terminal_regret

Run one problem per process so the three can go in parallel:
    python scripts/run_aggregator_ablation.py forrester
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from traits_audit.committee.analysis.thread_regret import run_thread_b_ablation
from traits_audit.committee.analysis.votes import CommitteeVoter
from traits_audit.committee.problems import get_problem

# Forrester's models live under committee_v0_5M even though its figures go to
# _results/committee_v1 -- the HPC scripts special-case this, so we do too.
PROBLEMS = {
    "forrester": ("runs/committee_v0_5M/models", "_results/committee_v1"),
    "branin-currin-mo": ("runs/committee_branin-currin-mo_5M/models",
                         "_results/committee_branin-currin-mo"),
    "color": ("runs/committee_color_5M/models", "_results/committee_color"),
}

RULES = [
    ("committee:uniform", None),
    ("committee:agree", None),
    ("committee:disagree", None),
    ("committee:weighted-indep", "independence"),
    ("committee:weighted-invreg", "inverse_regret"),
]

N_SEEDS = 20


def main(problem_name: str) -> None:
    models_dir, out_dir = PROBLEMS[problem_name]
    problem = get_problem(problem_name)
    voter = CommitteeVoter(REPO / models_dir)
    weights_all = json.loads(
        (REPO / out_dir / "threadA" / "thread_a_weights.json").read_text()
    )
    dest = REPO / out_dir / "threadA"
    dest.mkdir(parents=True, exist_ok=True)

    for rule, wkey in RULES:
        t0 = time.time()
        weights = weights_all[wkey] if wkey else None
        terminal = run_thread_b_ablation(
            voter, n_episode_seeds=N_SEEDS, policy=rule,
            problem=problem, weights=weights, rng_seed=0,
        )
        # Write the real RNG seeds, not 0..n-1: the renderer pairs these rows
        # against the baseline regret CSV by sorting on episode_seed, and a
        # positional index sorts into a different order than the true seeds,
        # silently comparing each drop against an unrelated episode.
        seed_rng = np.random.default_rng(0)
        seeds = [int(seed_rng.integers(0, 2**31 - 1)) for _ in range(N_SEEDS)]
        rows = ["dropped_agent,episode_seed,terminal_regret"]
        for agent, arr in terminal.items():
            for i, val in enumerate(np.asarray(arr).ravel()):
                rows.append(f"{agent},{seeds[i]},{val:.6f}")
        slug = rule.replace("committee:", "")
        path = dest / f"thread_a_ablation_{slug}.csv"
        path.write_text("\n".join(rows) + "\n")
        print(f"[{problem_name}] {rule:28s} -> {path.name} "
              f"({len(terminal)} agents, {time.time() - t0:.0f} s)",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
