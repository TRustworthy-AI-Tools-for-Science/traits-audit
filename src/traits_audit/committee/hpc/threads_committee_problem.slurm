#!/bin/bash
# =============================================================================
# Thread A / Thread B bake-offs, parallelised as a SLURM array.
#
# Serially these take hours: thread-b alone is 5 policies x 20 seeds x 100
# steps plus two 15-agent leave-one-out ablations (~7k rollouts), and each
# rollout refits a bootstrap surrogate at every step. The work splits
# cleanly by policy and by ablated agent, so each array task runs one slice
# and writes its own CSV shard; a gather step then renders the figures.
#
# Sharding is exact, not approximate: episode seeds are drawn before any
# policy filtering, so every shard evaluates the SAME 20 episodes and the
# merged result is byte-identical to a single-process run (asserted in
# tests/test_committee_threads_ndim.py). The paired Wilcoxon tests stay valid.
#
# Submit one problem at a time, after its training array and its `regret`
# step have finished (thread-a reads regret_test.json for its best-solo
# reference):
#
#     sbatch src/traits_audit/committee/hpc/threads_committee_problem.slurm forrester
#     sbatch src/traits_audit/committee/hpc/threads_committee_problem.slurm branin-currin
#     sbatch src/traits_audit/committee/hpc/threads_committee_problem.slurm color
#
# Then, once the array clears (the dependency makes this automatic):
#
#     sbatch --dependency=afterok:<ARRAY_JOBID> \
#            src/traits_audit/committee/hpc/gather_threads.slurm forrester
# =============================================================================

#SBATCH --job-name=committee_threads
#SBATCH --account=YOUR_ACCOUNT         # <-- replace with your Alliance allocation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-16                   # 2 bake-offs + 15 ablation shards
#SBATCH --output=logs/slurm/threads_%A_%a.out
#SBATCH --error=logs/slurm/threads_%A_%a.err

PROBLEM="${1:?usage: sbatch threads_committee_problem.slurm <forrester|branin-currin|color>}"

# ---------------------------------------------------------------------------
# Paths — edit these for your cluster
# ---------------------------------------------------------------------------
PROJECT_DIR="/scratch/$USER/traits-audit"            # edit locally, don't push
VENV_ACTIVATE="$HOME/envs/traits-audit/bin/activate" # edit locally, don't push

# Forrester's 15-agent run lives under committee_v0_5M/ but its figures go to
# committee_v1/; the other problems use matching names.
case "$PROBLEM" in
    forrester) MODELS="$PROJECT_DIR/runs/committee_v0_5M/models"
               RESULTS="$PROJECT_DIR/_results/committee_v1" ;;
    *)         MODELS="$PROJECT_DIR/runs/committee_${PROBLEM}_5M/models"
               RESULTS="$PROJECT_DIR/_results/committee_${PROBLEM}" ;;
esac

# ---------------------------------------------------------------------------
# Task layout — 17 tasks (--array=0-16), sized to the 15-agent registry:
#   0        thread-b policy bake-off      (5 policies, no ablation)
#   1        thread-a aggregator bake-off  (7 policies)
#   2-16     thread-b ablation, one agent each, both target policies
#
# If the reward registry grows, extend AGENTS below and bump --array to
# 0-(N_AGENTS+1). The script hard-fails on a mismatch rather than silently
# skipping an agent, which would leave a gap in the ablation figure.
# ---------------------------------------------------------------------------
AGENTS=(CRPS NLL IntervalScore CalibrationError KuleshovCalibration ENCE \
        CalibrationError1Std ConformalCoverage PITUniformity IntervalCoverage \
        VarianceAlignment VarErrCorrelation UncertaintyEvolution \
        UncertaintyAnomaly MahalanobisOOD)
N_AGENTS=${#AGENTS[@]}

cd "$PROJECT_DIR"
mkdir -p logs/slurm "$RESULTS/threadA" "$RESULTS/threadB"

module load StdEnv/2023
module load python/3.11
source "$VENV_ACTIVATE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

set -eo pipefail

TID=$SLURM_ARRAY_TASK_ID
echo "============================================================"
echo "Threads job: $SLURM_JOB_ID  task $TID  on $(hostname)  at $(date)"
echo "Problem:     $PROBLEM"
echo "Models:      $MODELS"
echo "============================================================"

if [ "$TID" -eq 0 ]; then
    # Policy bake-off only; the ablations are tasks 2..N_AGENTS+1.
    ta-committee-analyze thread-b \
        --problem "$PROBLEM" --models-dir "$MODELS" \
        --output-dir "$RESULTS/threadB" \
        --shard-tag "policies" --skip-ablation

elif [ "$TID" -eq 1 ]; then
    ta-committee-analyze thread-a \
        --problem "$PROBLEM" --models-dir "$MODELS" \
        --correlation-csv "$RESULTS/correlation_trained.csv" \
        --regret-json     "$RESULTS/regret_test.json" \
        --output-dir "$RESULTS/threadA" \
        --shard-tag "all"

elif [ "$TID" -lt $(( N_AGENTS + 2 )) ]; then
    AGENT=${AGENTS[$(( TID - 2 ))]}
    echo "Ablating agent: $AGENT"
    # --only with a single cheap policy keeps the shard's own bake-off
    # trivial; the ablation is what this task is really for. The shard's
    # regret CSV is still written and merges harmlessly at gather time.
    ta-committee-analyze thread-b \
        --problem "$PROBLEM" --models-dir "$MODELS" \
        --output-dir "$RESULTS/threadB" \
        --only "LCB" \
        --ablate-agents "$AGENT" \
        --shard-tag "ablate_${AGENT}"

else
    echo "ERROR: task $TID exceeds the $(( N_AGENTS + 2 ))-task layout for a" \
         "${N_AGENTS}-agent registry. Fix --array to 0-$(( N_AGENTS + 1 ))." >&2
    exit 1
fi

echo "finished at $(date)"
