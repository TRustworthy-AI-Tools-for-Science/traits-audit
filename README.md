# traits-audit

This repository contains two things:

| Directory | What it is | Who it is for |
|---|---|---|
| [`packages/traits-audit`](packages/traits-audit) | **The `traits-audit` library**: a generic, passive uncertainty-audit pipeline for any active-learning loop. Published to PyPI. | Anyone who wants to audit the uncertainty of their own AL / BO / SDL campaign. |
| [`experiments/results`](experiments/results) | **Paper reproduction code**: the case studies and RL steering study behind the NMI 2026 paper, built on the library. | Reviewers and readers reproducing the paper's results. |

Documentation for the library lives in [`docs/`](docs) and is published at
**https://trustworthy-ai-tools-for-science.github.io/traits-audit/**.

## Using the library

```bash
pip install traits-audit
ta-demo
```

See [`packages/traits-audit/README.md`](packages/traits-audit/README.md).

## Developing / reproducing the paper

The repository is a [uv](https://docs.astral.sh/uv/) workspace:

```bash
uv sync                                    # library + experiment (core) + dev tools
uv sync --all-extras                       # + every case study's heavy dependencies
uv run pytest packages/traits-audit/tests  # library tests
uv run pytest experiments/results/tests    # experiment tests
```

To reproduce the paper, see
[`experiments/results/README.md`](experiments/results/README.md). Results are
not tracked in git. They are archived with a frozen copy of the code on Zenodo.

## License

MIT. See [LICENSE](LICENSE).
