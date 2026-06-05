"""AuditPipeline: runs a list of AuditChecks and aggregates results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import AuditCheck, AuditReport


class AuditPipeline:
    """
    Runs a registered list of :class:`~traits_audit.base.AuditCheck` instances
    against accumulated step history and aggregates results into an
    :class:`~traits_audit.base.AuditReport`.

    A failed check does **not** stop subsequent checks — all checks always run.

    Parameters
    ----------
    checks : list[AuditCheck]
        Ordered list of checks to execute.
    verbose : bool
        If ``True``, prints each check result as it completes.

    Notes
    -----
    ``run`` is called by :class:`~traits_audit.hook.AuditHook`; you do not
    normally call it directly.  However, you may call it with any ``history``
    list and kwargs for offline / post-hoc analysis.
    """

    def __init__(self, checks: list[AuditCheck], verbose: bool = False):
        self.checks = checks
        self.verbose = verbose

    def run(
        self,
        history: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AuditReport:
        """
        Execute all checks.

        Parameters
        ----------
        history : list[dict]
            Accumulated step data from :meth:`~traits_audit.hook.AuditHook.on_step`.
            Each dict corresponds to one loop step.
        metadata : dict, optional
            Stored verbatim in the report (experiment name, timestamp, etc.).
        **kwargs
            Forwarded unchanged to every check.  Use named arguments so each
            check can select only what it needs.

        Returns
        -------
        AuditReport
        """
        report = AuditReport(metadata=metadata or {})
        for check in self.checks:
            result = check.run(history, **kwargs)
            report.results.append(result)
            if self.verbose:
                tag = "PASS" if result.passed else "FAIL"
                val = f" ({result.value:.4f})" if result.value is not None else ""
                print(f"[{tag}] {result.name}{val}: {result.message}")
        return report

    def save(
        self,
        report: AuditReport,
        path: str | Path,
        merge: bool = True,
    ) -> None:
        """
        Persist a report to JSON.

        Parameters
        ----------
        merge : bool
            If ``True`` and the file already exists, append new results rather
            than overwriting.
        """
        path = Path(path)
        data = report.to_dict()
        if merge and path.exists():
            existing = json.loads(path.read_text())
            # Replace by check name so repeated saves don't accumulate duplicates.
            existing_by_name = {r["name"]: r for r in existing.get("results", [])}
            for r in data["results"]:
                existing_by_name[r["name"]] = r
            merged_results = list(existing_by_name.values())
            existing["results"] = merged_results
            # Recompute summary fields from the merged result set.
            existing["n_passed"] = sum(1 for r in merged_results if r.get("passed"))
            existing["n_failed"] = sum(1 for r in merged_results if not r.get("passed"))
            existing["passed"] = existing["n_failed"] == 0
            data = existing
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
