Writing custom checks
=====================

Subclass :class:`~traits_audit.base.AuditCheck` and implement three
things: :attr:`~traits_audit.base.AuditCheck.name`,
:attr:`~traits_audit.base.AuditCheck.category`, and
:meth:`~traits_audit.base.AuditCheck.run`.

.. code-block:: python

   from traits_audit.base import AuditCheck, AuditCategory, AuditResult

   class MyCheck(AuditCheck):

       @property
       def name(self) -> str:
           return "MyCheck"

       @property
       def category(self) -> AuditCategory:
           return AuditCategory.EPISTEMIC

       def run(self, history: list[dict], **kwargs) -> AuditResult:
           ...

:meth:`~traits_audit.base.AuditCheck.run` receives:

- **history** — ``list[dict]``, one dict per :meth:`~traits_audit.hook.AuditHook.on_step` call.
- **\*\*kwargs** — named arrays from :meth:`~traits_audit.hook.AuditHook.on_end` or :meth:`~traits_audit.pipeline.AuditPipeline.run`.

A check should always return an :class:`~traits_audit.base.AuditResult`
— never raise.  If required data is missing, return ``passed=True`` with a
skip message so the pipeline remains non-blocking.


Example 1 — batch check (kwargs only)
--------------------------------------

Uses predictions assembled after the loop.  The loop does not need to
record anything extra via ``on_step``.

.. code-block:: python

   import numpy as np
   from traits_audit.base import AuditCheck, AuditCategory, AuditResult

   class ENCECheck(AuditCheck):
       """Expected Normalised Calibration Error (Levi et al. 2022).

       Compares root-mean-variance (RMV) to RMSE in equal-frequency bins.

       Parameters
       ----------
       n_bins : int
           Number of equal-frequency bins (default: 10).
       threshold : float
           Maximum acceptable ENCE (default: 0.1).

       Required kwargs
       ---------------
       ``y_true``, ``y_pred_mean``, ``y_pred_std``
       """

       def __init__(self, n_bins: int = 10, threshold: float = 0.1):
           self.n_bins    = n_bins
           self.threshold = threshold

       @property
       def name(self) -> str:
           return f"ENCE(bins={self.n_bins})"

       @property
       def category(self) -> AuditCategory:
           return AuditCategory.ALEATORIC_MODEL

       def run(self, history, *, y_true=None, y_pred_mean=None,
               y_pred_std=None, **kwargs):
           if any(v is None for v in (y_true, y_pred_mean, y_pred_std)):
               return AuditResult(
                   name=self.name, passed=True, category=self.category,
                   message="Skipped — y_true / y_pred_mean / y_pred_std not provided.",
               )

           y_true = np.asarray(y_true).ravel()
           mu     = np.asarray(y_pred_mean).ravel()
           sigma  = np.asarray(y_pred_std).ravel()

           order = np.argsort(sigma)
           bins  = np.array_split(order, self.n_bins)
           terms = []
           for idx in bins:
               rmv  = float(np.sqrt(np.mean(sigma[idx] ** 2)))
               rmse = float(np.sqrt(np.mean((y_true[idx] - mu[idx]) ** 2)))
               terms.append(abs(rmv - rmse) / (rmv + 1e-12))

           ence = float(np.mean(terms))
           return AuditResult(
               name=self.name,
               passed=ence <= self.threshold,
               category=self.category,
               value=ence,
               threshold=self.threshold,
               message=f"ENCE = {ence:.4f}",
               details={"bin_terms": terms},
           )

Wire it up — no changes to the loop body:

.. code-block:: python

   pipeline = AuditPipeline([ENCECheck(n_bins=10, threshold=0.1)])
   hook     = AuditHook(pipeline)

   for step in my_loop:
       hook.on_step(...)    # no extra keys needed for this check

   report = hook.on_end(y_true=y_test, y_pred_mean=mu_test, y_pred_std=sigma_test)


Example 2 — sequence check (history only)
------------------------------------------

Reads a value recorded at every step.  No batch arrays needed.

.. code-block:: python

   class PredictiveEntropyDecayCheck(AuditCheck):
       """Passes if predictive entropy drops by at least ``min_relative_drop``
       from the first step to the last.

       Parameters
       ----------
       min_relative_drop : float
           Minimum required fractional entropy reduction (default: 0.1).

       Required history key
       --------------------
       ``entropy`` — record per step via ``hook.on_step(entropy=...)``.
       """

       def __init__(self, min_relative_drop: float = 0.1):
           self.min_relative_drop = min_relative_drop

       @property
       def name(self) -> str:
           return "PredictiveEntropyDecay"

       @property
       def category(self) -> AuditCategory:
           return AuditCategory.EPISTEMIC

       def run(self, history, **kwargs):
           entropies = [h["entropy"] for h in history if "entropy" in h]

           if len(entropies) < 2:
               return AuditResult(
                   name=self.name, passed=True, category=self.category,
                   message="Skipped — fewer than 2 steps with 'entropy'.",
               )

           initial, final = float(entropies[0]), float(entropies[-1])
           drop = (initial - final) / (initial + 1e-12)

           return AuditResult(
               name=self.name,
               passed=drop >= self.min_relative_drop,
               category=self.category,
               value=drop,
               threshold=self.min_relative_drop,
               message=f"Entropy drop {drop:.1%}  ({initial:.4f} → {final:.4f})",
           )

The loop records ``entropy`` each step:

.. code-block:: python

   import numpy as np

   hook = AuditHook(pipeline)
   for step in my_loop:
       mu, sigma = model.predict_with_uncertainty(X_candidates)
       hook.on_step(
           uncertainty=float(sigma.mean()),
           entropy=float(0.5 * np.log(2 * np.pi * np.e * sigma ** 2).mean()),
       )

   report = hook.on_end()   # no batch kwargs needed for this check


Example 3 — mixed (history with kwargs fallback)
-------------------------------------------------

Accepts data from either route so the check also works offline (post-hoc,
without a live hook).

.. code-block:: python

   class MaxUncertaintyFractionCheck(AuditCheck):
       """Flags if more than ``threshold`` fraction of steps had uncertainty
       above ``cap``.

       Parameters
       ----------
       cap : float
           Absolute upper limit on acceptable uncertainty.
       threshold : float
           Maximum acceptable fraction of steps above ``cap`` (default: 0.1).

       Accepts
       -------
       ``uncertainties`` kwarg (batch array) **or** ``uncertainty`` per-step
       history key.
       """

       def __init__(self, cap: float, threshold: float = 0.1):
           self.cap       = cap
           self.threshold = threshold

       @property
       def name(self) -> str:
           return f"MaxUncertaintyFraction(cap={self.cap})"

       @property
       def category(self) -> AuditCategory:
           return AuditCategory.ALEATORIC_IRREDUCIBLE

       def run(self, history, *, uncertainties=None, **kwargs):
           if uncertainties is not None:
               u = np.asarray(uncertainties, dtype=float)
           else:
               vals = [h["uncertainty"] for h in history if "uncertainty" in h]
               if not vals:
                   return AuditResult(
                       name=self.name, passed=True, category=self.category,
                       message="Skipped — no uncertainty data.",
                   )
               u = np.asarray(vals, dtype=float)

           fraction = float((u > self.cap).mean())
           return AuditResult(
               name=self.name,
               passed=fraction <= self.threshold,
               category=self.category,
               value=fraction,
               threshold=self.threshold,
               message=f"{fraction:.1%} of steps exceeded cap ({self.cap})",
               details={"n_exceeded": int((u > self.cap).sum())},
           )


Registering mixed checks
------------------------

Checks are order-independent.  Register all of them in one pipeline
regardless of which data route they use — the pipeline passes both
``history`` and ``**kwargs`` to every check.

.. code-block:: python

   pipeline = AuditPipeline(
       checks=[
           CalibrationErrorCheck(threshold=0.1),     # batch
           ENCECheck(n_bins=10),                      # batch
           UncertaintyEvolutionCheck(),               # sequence
           PredictiveEntropyDecayCheck(),             # sequence
           MaxUncertaintyFractionCheck(cap=2.0),      # mixed
       ],
       verbose=True,
   )

   hook = AuditHook(pipeline)

   for step in my_loop:
       mu, sigma = model.predict_with_uncertainty(X)
       hook.on_step(
           uncertainty=float(sigma.mean()),
           entropy=float(0.5 * np.log(2 * np.pi * np.e * sigma ** 2).mean()),
       )

   report = hook.on_end(
       y_true=y_test,
       y_pred_mean=mu_test,
       y_pred_std=sigma_test,
   )

   print(report.summary())
   pipeline.save(report, "audit.json")


AuditResult reference
----------------------

.. autoclass:: traits_audit.base.AuditResult
   :members:
   :no-index:

.. autoclass:: traits_audit.base.AuditReport
   :members:
   :no-index:

.. autoclass:: traits_audit.base.AuditCheck
   :members:
   :no-index:
