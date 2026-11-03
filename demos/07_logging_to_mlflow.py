"""
Logging the audit to MLflow
===========================

*Needs: mlflow (``pip install "traits-audit[mlflow]"``).*

If you already track experiments in MLflow, give the hook an
:class:`~traits_audit.mlflow_logger.MLflowLogger`. Every ``on_step`` value
becomes an MLflow metric, every audit report is stored as JSON, and checks
that can draw themselves (CalibrationError) are saved as figures. Nothing
else in the loop changes.
"""

# %%
# Step 1: point MLflow at a tracking store
# ----------------------------------------
# This demo writes to a throwaway SQLite file in a temporary folder. In your
# own project, use your usual tracking server or ``mlflow.db`` file.

import sys
import tempfile
from pathlib import Path

try:
    import mlflow
except ImportError:
    sys.exit('This demo needs mlflow: pip install "traits-audit[mlflow]"')

import matplotlib.pyplot as plt
import numpy as np
from traits_audit import AuditHook, AuditPipeline
from traits_audit.checks import CalibrationErrorCheck, IntervalCoverageCheck, NegativeLogLikelihoodCheck
from traits_audit.mlflow_logger import MLflowLogger

N_STEPS = 50
CHECK_EVERY = 10
NOISE = 0.1
SEED = 0

tracking_db = Path(tempfile.mkdtemp()) / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
mlflow.set_experiment("traits-audit-demo")

# %%
# Step 2: run the loop inside an MLflow run
# -----------------------------------------
# The only new line is ``logger=MLflowLogger()``. The logger attaches to the
# active run. The loop is the simulated one from demo 06: a model whose error
# shrinks as it sees more data, reporting an honest standard deviation.

pipeline = AuditPipeline([
    CalibrationErrorCheck(threshold=0.15),
    IntervalCoverageCheck(expected_coverage=0.683, tolerance=0.15),
    NegativeLogLikelihoodCheck(),
])
rng = np.random.default_rng(SEED)

with mlflow.start_run(run_name="audited-loop") as run:
    hook = AuditHook(pipeline, check_every=CHECK_EVERY, logger=MLflowLogger())
    for step in range(N_STEPS):
        sigma = np.hypot(0.5 / np.sqrt(step + 1), NOISE)
        y_pred = rng.normal()
        y_true = y_pred + rng.normal(0, sigma)
        hook.on_step(y_true=y_true, y_pred_mean=y_pred, y_pred_std=sigma, uncertainty=sigma)
    report = hook.on_end()

print(report.summary())

# %%
# Step 3: what ended up in MLflow
# -------------------------------
# Metrics are named ``audit/step/<key>`` for each ``on_step`` value,
# ``audit/intermediate/<check>`` for the reports every ``CHECK_EVERY`` steps,
# and ``audit/final/<check>`` for the final report. Reports and figures are
# artifacts under ``audit/``.

client = mlflow.MlflowClient()
run_id = run.info.run_id
print("Some metric keys:", sorted(client.get_run(run_id).data.metrics)[:6])
print("Artifacts:", [a.path for a in client.list_artifacts(run_id, "audit")])

# %%
# Step 4: read a metric back
# --------------------------
# The intermediate CalibrationError values form a history you can plot here or
# browse in the MLflow UI with
# ``mlflow ui --backend-store-uri sqlite:///path/to/mlflow.db``.

ce = client.get_metric_history(run_id, "audit/intermediate/CalibrationError")
fig, ax = plt.subplots(figsize=(4.5, 3))
ax.plot([m.step for m in ce], [m.value for m in ce], marker="o")
ax.axhline(0.15, color="grey", ls="--", lw=1, label="threshold")
ax.set(xlabel="step", ylabel="CalibrationError", title="Read back from MLflow")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
