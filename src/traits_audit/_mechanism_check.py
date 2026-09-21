"""Internal helper: print the controllability-Gramian mechanism-check block
shared by the ta-camd / ta-pybamm / ta-sdl demos.

Not part of the public API — this is demo-output formatting (the same role
``_viz.py`` plays for plots), not a new ``AuditCheck``. The controllability
mechanism (does the Gramian eigenvalue ratio separate a genuine reducible
epistemic component from a fixed aleatoric floor?) is this paper's own
diagnostic, run as part of a demo/experiment, not shipped as a pluggable
audit metric — see ``_paper/paper1_logical_pitfalls.md`` Categories 1 and 2
for why the eigenvalue ratio (not eigenvector alignment) is the headline
statistic reported here, and why a short/poorly-identified trajectory must be
flagged rather than silently trusted.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .trajectory import TrajectoryDMDcResult


def print_mechanism_check(
    result: TrajectoryDMDcResult,
    label: str,
    aleatoric_indices: Sequence[int] | None = None,
) -> None:
    """Print one controllability-Gramian mechanism-check block.

    Parameters
    ----------
    result : TrajectoryDMDcResult
        Output of :func:`traits_audit.trajectory.analyze_trajectory`.
    label : str
        Short run label, e.g. ``"real split"`` or ``"two-epistemic null"``.
    aleatoric_indices : sequence of int, optional
        0-based indices *within the uncertainty block* (i.e. relative to the
        first uncertainty column of the augmented state, not the full
        augmented state) of the components that are the irreducible
        floor. Only used to report the secondary alignment statistic; omit
        (or pass ``None``) for a null run where no component is aleatoric.
    """
    print(f"\n=== Mechanism check: {label} ===")

    ratio = result.cond_Wc
    ci = result.cond_Wc_ci
    ci_str = f"  (95% CI {ci[0]:.3g}-{ci[1]:.3g})" if ci is not None else "  (CI unavailable)"
    print(f"  eigenvalue ratio cond(Wc) [HEADLINE statistic]: {ratio:.3g}{ci_str}")
    print(
        f"  rho(A_state)={result.rho_A_state:.3g}  "
        f"rho(A_unc)={result.rho_A_unc:.3g}  "
        f"rho(A_joint)={result.rho_A_joint:.3g}"
    )

    if not result.min_length_ok:
        print(
            "  [WARNING] trajectory shorter than the T >= 4d+1 estimability "
            "boundary -- this ratio is under-powered, interpret with caution."
        )
    if np.isfinite(ratio) and ratio > 1e6:
        print(
            "  [NOTE] very large ratio -- check action diversity before "
            "over-interpreting; a poorly-identified B (low protocol variance) "
            "can inflate cond(Wc) even without a genuine reducible/irreducible "
            "split."
        )

    if aleatoric_indices:
        n_unc = result.uncertainty_dim
        n_state = result.Wc_eigenvectors.shape[0] - n_unc
        least_controllable = result.Wc_eigenvectors[:, -1]
        al_mass = float(np.linalg.norm(
            [least_controllable[n_state + i] for i in aleatoric_indices]
        ))
        print(
            f"  [secondary, non-discriminating] aleatoric-axis alignment of the "
            f"least-controllable direction: {al_mass:.2f} (saturates near 1 "
            "whether or not a real split exists -- the ratio above, not this "
            "number, is the statistic that actually separates them)"
        )

    print(
        "  Note: the Gramian marks an UNCONTROLLABLE / non-contracting "
        "direction -- it does not, by itself, prove that direction is "
        "'aleatoric uncertainty' rather than e.g. a poorly-explored one."
    )
