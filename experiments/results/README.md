# traits-audit-results — paper reproduction code

Code that generates every result in *Automating Uncertainty Auditing During
Autonomous Scientific Discovery for the Human-in-the-loop* (NMI 2026, in
preparation). It is built on the generic
[`traits-audit`](../../packages/traits-audit) package, which you need
only if you want to audit your *own* active-learning loop.

This directory holds **code only**. Results (figures, CSVs, trained models)
are never committed. They are published, together with a frozen copy of
this code, as a Zenodo deposit (DOI: *TBD on release*).

## Setup

From the repository root:

```bash
uv sync --all-extras          # everything (needs torch: Linux, Windows, or Apple-silicon macOS)
uv sync --extra mobo          # or only what one case study needs
```

| Extra       | Needed for                                    |
|-------------|-----------------------------------------------|
| *(none)*    | Forrester calibration (`ta-cal-demo`)         |
| `sdl`       | SDL colour-matching digital twin              |
| `mobo`      | Branin-Currin MOBO                            |
| `pybamm`    | Battery digital twin                          |
| `camd`      | CAMD materials-stability screening            |
| `committee` | RL committee steering study                   |
| `all`       | all of the above                              |

## Case studies → commands

Every demo writes to `_results/<demo>/` under the working directory (ignored
by git).

| Paper case study                                         | Command |
|----------------------------------------------------------|---------|
| 1. SOBO, Forrester, four calibration conditions          | `ta-cal-demo` |
| 2. SOBO, SDL colour-mixing digital twin                  | `ta-sdl-demo` |
| 3. MOBO, Branin-Currin                                   | `ta-mobo-demo` |
| 4. MOBO, battery digital twin (PyBaMM)                   | `ta-pybamm-demo` |
| 5. RL: audit metrics as a steering signal                | see below |
| Supplementary: CAMD materials stability                  | `ta-camd-demo` |

Each command takes `--help`. Record the exact command lines used for the paper
alongside the Zenodo deposit.

### Case study 5: committee RL

Training is expensive (15 agents × 5 seeds × 3 problems) and is meant for a
SLURM cluster. See [`hpc/README.md`](hpc/README.md) for the job-array
templates. Locally:

```bash
ta-committee-train --agents CRPS --seeds 0 --episodes 50    # one smoke-sized agent
```

Once training is done, regenerate the figures:

```bash
experiments/results/scripts/make_all_committee_figures.sh   _results/committee_v1  runs/committee_v0_5M/models
experiments/results/scripts/make_problem_figures.sh color   _results/committee_color
experiments/results/scripts/make_problem_figures.sh branin-currin
```

`thread-a` reads `regret_test.json` and `correlation_trained.csv` from an
earlier stage. Because results are no longer committed, they no longer
arrive on the cluster via `git pull`: run the `regret` / `corr-trained`
stages first (the scripts do this in order) or copy those files over.

## Layout

```
src/results/
  cal_demo.py  sdl_demo.py  mobo_demo.py  pybamm_demo.py  camd_demo.py
  viz.py               figure code shared by the demos
  mechanism_check.py   Gramian mechanism-check printout
  committee/           RL environment, rewards, agents, analysis CLI
  policies/            state-space active-learning policies (numpy + traits-audit only)
scripts/               figure drivers, ablations, state-space steering study
hpc/                   SLURM templates
tests/
```

### State-space steering (paper: "Novel Active Learning Policies")

`results.policies` learns the uncertainty-state dynamics
x<sub>t+1</sub> = A x<sub>t</sub> + B u<sub>t</sub> online and uses them to
choose the next query. Two standalone scripts apply the policies to the
Forrester problem:

```bash
python experiments/results/scripts/steer_state_space.py --out _results/steering
python experiments/results/scripts/analyze_steering.py  --in  _results/steering
```

`steer_state_space.py` follows the experimental plan's conventions:

- Metrics are evaluated on a fixed reference grid (§3.3).
- Metrics are z-scored against the warm-start baseline (§3.1).
- A random-probe fraction of the budget is reserved (§3.4).
- The metrics are split into a **steering** subset, which the policy sees,
  and an **audit-only** subset, which it never sees (§6.1). Choose the split
  with `--steering`.

It compares Random, LCB, StateSpace, BayesianDynamics and their
health-steered `audit-*` variants, with seeds paired across policies.

`analyze_steering.py` tests the four §6.1 predictions against the random
policy on the same seeds:

1. Steered metrics improve.
2. Audit-only metrics don't.
3. Cross-group co-movement breaks.
4. A drifts once the loop is closed.

It writes `steering_summary.json` and `steering_summary.png`.

## Tests

```bash
uv run pytest experiments/results/tests
```

Tests whose optional dependencies are missing are skipped, not failed.

### Golden outputs

`tests/test_golden.py` runs each case study at smoke size with a fixed seed and
compares every audit report, every MLflow metric and the list of output files
against `tests/golden/<case>.json`. Refactors of the paper code must keep these
unchanged. To record a case for the first time, or to accept a deliberate change:

```bash
uv run pytest experiments/results/tests/test_golden.py --update-golden
```

Record on a machine with the case's extras installed. Torch has no wheels for
Intel macOS, and `camd` needs `pymatgen` to build, so the SDL, MOBO, PyBaMM and
CAMD goldens need Linux or Apple-silicon macOS.
