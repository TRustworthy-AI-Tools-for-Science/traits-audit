"""Post-training analysis for committee-RL v0.

Four deliverables from the v0 plan, each in its own module:

- ``rollouts``    — shared rollout engine. Random-action and trained-policy
                    episodes over the Forrester env, capturing the per-step
                    info needed to score every reward computer offline.
- ``correlation`` — 9x9 cross-agent reward correlation matrix (random and
                    trained), clustering, heatmap rendering.
- ``density``     — per-agent query density distributions (the headline
                    9-panel figure vs predicted_styles.md).
- ``regret``      — simple-regret curves (random / max-sigma / LCB / 9 solo /
                    committee) with paired statistical test.

CLI entry point ``ta-committee-analyze`` lives in ``_cli``.
"""
