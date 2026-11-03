# Plan: split demos from paper results, refactor `viz.py` and the case studies

## Goals

1. **`demos/`** is a new top-level folder for anyone. It holds short, readable scripts that show
   how to put `traits-audit` into an experiment. They power the documentation.
2. **`experiments/results/`** is only for the NMI 2026 paper. Its code may stay heavy and
   specialised, but it should be organised so a reviewer can follow it.
3. Paper outputs must not change. Every refactor step in `results` must preserve behaviour.

## Current state (what we're fixing)

| File | Lines | Problem |
|---|---|---|
| `results/viz.py` | 2,102 | Mixes four things: generic audit plots, domain plots (CIE colour, materials, Pareto), analysis code (`run_lyapunov_analysis`, `run_dmdc_lyapunov_analysis`) and global rcParams applied on import. The demos import private helpers (`_save`, `_fig_metric_correlations`, `_fig_check_grid`). |
| `results/cal_demo.py` | 1,192 | A 360-line `_run_scenario`. It runs about 30 checks, replication arms, refit sweeps, provenance and MLflow all at once. |
| `results/{sdl,mobo,pybamm,camd}_demo.py` | 730–1,130 | One 450–650-line `run()` each. Repeated scaffolding: MLflow setup, `_predict`, `_make_pipeline`, the Lyapunov block and figure saving. |
| `docs/demo_*.rst` | 530–660 | These are paper case-study write-ups shown as the library's "case studies". About 6.8 MB of committed PNGs come from heavy runs that docs CI can't reproduce. |

A novice can't learn "how do I add traits-audit to my loop" from any of these. The pattern
is buried under paper-specific machinery. The only example a novice can read today is
`traits_audit/_quickstart.py` (97 lines).

## Target layout

```
demos/                                   # for everyone; rendered into the docs
  README.md                              # what each demo teaches + how to run
  01_minimal_loop.py                     # hook.on_step / on_end, print the report
  02_detecting_miscalibration.py         # same loop, over/under-confident surrogate → checks fail
  03_bayesian_optimization.py            # BoTorch GP + acquisition, check_every=N intermediate reports (needs botorch)
  04_plotting_a_report.py                # traits_audit.plotting: check grid, uncertainty evolution
  05_stability_of_the_loop.py            # trajectory → DMDc / Lyapunov checks, what ρ(A) means
  06_writing_a_custom_check.py           # subclass AuditCheck
  07_logging_to_mlflow.py                # MLflowLogger (needs traits-audit[mlflow])
  08_existing_framework_callback.py      # callback-slot pattern with an ask/tell optimiser (optional extra)

experiments/results/src/results/         # paper only
  common/
    pipeline.py      # shared pipeline builders (one place for thresholds)
    tracking.py      # MLflow run setup / artifact logging
    surrogates.py    # _predict / fit wrappers shared across BoTorch case studies
    lyapunov.py      # run_lyapunov_analysis, run_dmdc_lyapunov_analysis (moved out of viz)
  figures/
    style.py         # paper rcParams, applied through a context manager, not on import
    audit.py         # paper-specific audit figures (state heatmap, metric correlations, scenario panels)
    stability.py     # poles, stability contours, Lyapunov evolution
    pareto.py        # Pareto frontier / scenarios
    color.py         # CIE background + trajectory (also used by committee.analysis.density)
    materials.py     # exploration campaign, discovery rate (CAMD)
  case_studies/
    forrester_calibration.py   # was cal_demo.py
    sdl_color.py               # was sdl_demo.py
    mobo_branin_currin.py      # was mobo_demo.py
    pybamm_battery.py          # was pybamm_demo.py
    camd_stability.py          # was camd_demo.py
  committee/  policies/        # unchanged
```

Generic plots move into the library as a public, tested module, `traits_audit.plotting`.
They are the plots that take only an `AuditReport` or the hook history, so the demos can
call them in one line:

- `plot_check_grid(stage_reports)`, from `_fig_check_grid`, `check_grid_figures`, `_result_status` and `_CHECK_ABBREV`
- `plot_uncertainty_evolution(history)` and `plot_audit_evolution(stage_reports)`
- `plot_calibration_curve(result)`, from `_fig_calibration_curve`

The library versions use matplotlib only (decided 2 Nov 2026). The paper's check grid is
Plotly, so it stays in `results` but now draws on the library's shared status logic
(`result_status`, `CHECK_ABBREVIATIONS`, `split_trackable_checks`, `filter_stage_reports`).
Paper figures stay byte-identical.

## Rules for a demo script

Every file in `demos/` must:

- **Be one file, read top to bottom**, with numbered `# Step N:` comments. Define no
  classes unless a class is the point (for example, demo 06).
- **Be about 60–150 lines**, with no private imports (`_foo`) and nothing from `results`.
- **Use only core dependencies** (numpy, scipy, scikit-learn, matplotlib), except demos 03
  (BoTorch), 07 (MLflow) and 08 (an ask/tell framework). Those state their extra on the
  first docstring line and skip cleanly without it.
- **Avoid argparse.** Settings are module-level constants (`N_STEPS = 40`, `SEED = 0`)
  that a reader can edit.
- **Run in under 10 seconds** so the docs build can execute every demo.
- **Show the audit as an add-on to an existing loop.** The loop must make sense without
  the hook, and the hook lines should be clearly marked (`# ← traits-audit`).
- **End with what to look for**: a printed report and, where useful, one figure.

## Docs

- **Generate the docs pages from `demos/` with `sphinx-gallery`** (add it to the
  `traits-audit[docs]` extra). The docs build runs each script and creates its figures, so
  no demo PNGs are committed. A broken demo then fails the `-W` docs build in CI, which
  doubles as a smoke test.
- **Add a "Demos" toctree (the gallery) before the existing "Case studies" toctree.**
  The case-study pages stay on the docs site: they show how the package is used on real
  problems and explain what the results mean.
- **Rename the case-study pages from `demo_*.rst` to `case_study_*.rst`** so they aren't
  confused with the gallery, and add redirects for the old URLs. Update their commands to
  `ta-paper-*` and their module paths to `results.case_studies.*`. Their committed figures
  in `docs/_static/` stay, because CI can't regenerate them; rename the folders to
  `case_study_*/` to match.
- **Track the overlap with the paper.** Some of these results will be folded into the
  paper. When that happens, each affected page should cite the paper for the result and
  keep the usage walk-through.
- **Keep `ta-demo`** (the library's `_quickstart.py`) for pip users. Make
  `demos/01_minimal_loop.py` the same example so there is one canonical first example.

## Phases

Each phase is its own PR and leaves `main` green. That means library tests, experiment
tests and `make -C docs html SPHINXOPTS=-W` all pass.

### Phase 0: freeze a baseline for the paper code
- Run every case study at smoke size with fixed seeds. Use the existing `test_demos.py`
  arguments, plus the CAMD, PyBaMM and SDL extras on a machine that has them. Record:
  - the list of output files
  - each final `AuditReport` (`report.summary()` plus per-check values) and the hook history (`hook.save_history`)
  - key scalars (hypervolume, ρ(A), fraction stable)
- Store them under `experiments/results/tests/golden/`, and add
  `test_golden.py` to compare against them (with a tolerance on floats). This test guards
  every later phase.

### Phase 1: `traits_audit.plotting` (library)
- Move the generic plots listed above into the library and give them public names and
  NumPy docstrings. Add tests in the style of `tests/test_stability_viz_*.py`, which run
  headless and check figure output.
- Point `results` at the new functions and delete the moved code from `viz.py`.
- Add an API page at `docs/api/plotting.rst`.

### Phase 2: `demos/` and the gallery
- Write demos 01–08 against the public API only. Configure sphinx-gallery
  (`examples_dirs="../demos"`, `gallery_dirs="demos"`) and add the docs toctree entry.
- Make the CI docs job install `traits-audit[docs,mlflow]` plus `botorch`, so demos 03 and
  07 execute in CI. Demo 08's extra is optional: the demo skips cleanly if it is missing.
  Pin a CPU-only torch wheel in the docs job to keep install time down, and check that
  demo 03 still runs in under 10 seconds.
- Add `demos/README.md` and link it from the root README and the quickstart.

### Phase 3: break up `results/viz.py`
- Split it into the `figures/` modules above. Move the two `run_*_lyapunov_analysis`
  functions into `common/lyapunov.py`. Replace the import-time `_RCPARAMS` side effect
  with `figures.style.paper_style()`, a context manager or decorator.
- Make the helpers the case studies call public (`_save` becomes `save_figure`,
  `_fig_metric_correlations` becomes `plot_metric_correlations`).
- Update `committee.analysis.density` and `test_viz_status.py`.
- The golden test must pass without changes.

### Phase 4: restructure the case studies
- Move `*_demo.py` into `case_studies/`. Split each `run()` into stages with clear inputs
  and outputs: `setup`, `run_loop`, `audit`, `analyze`, `make_figures`. Have `run()` call
  them in order.
- Pull the shared scaffolding into `common/`: MLflow setup, the `_predict` and fit
  wrappers, and the Lyapunov block (currently repeated in the SDL, MOBO and PyBaMM studies).
- Split `cal_demo._run_scenario` along its existing section comments: replication arm,
  refit sweeps, provenance, stage attribution, Lyapunov/DMDc.
- Rename the commands in `[project.scripts]` to `ta-paper-cal`, `ta-paper-sdl`,
  `ta-paper-mobo`, `ta-paper-pybamm` and `ta-paper-camd`. Update the README command table,
  `test_demos.py` (rename it to `test_case_studies.py`), the HPC scripts and the deferred
  Zenodo command record. The golden test must pass without changes.

### Phase 5: update the case-study pages and clean up
- Rename and update the case-study pages as described under Docs.
- Update the changelog and `hook_pipeline_map.md`. Grep for stale references to
  `ta-*-demo`, `results.viz`, `*_demo.py` and `demo_*.rst`.

## Decisions (2 Nov 2026)

1. **Case-study write-ups stay on the docs site.** They document how to use the package
   and what the results mean. Some results will also be folded into the paper.
2. **The paper commands are renamed to `ta-paper-*`.**
3. **The docs use sphinx-gallery** to turn `demos/` into pages.
4. **Demo 03 uses BoTorch.** The docs CI job installs it.
5. **`traits_audit.plotting` is matplotlib-only.** The paper keeps its Plotly check grid.
