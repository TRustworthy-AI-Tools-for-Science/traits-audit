#!/bin/bash
# =============================================================================
# Committee-RL on another benchmark — all 15 agents × 5 seeds = 75 tasks.
#
# Submit (problem name is the first argument):
#     sbatch src/traits_audit/committee/hpc/train_committee_problem.slurm branin-currin
#     sbatch src/traits_audit/committee/hpc/train_committee_problem.slurm color
#
# Each problem writes to its own run dir, runs/committee_<problem>_5M/, with
# the same layout as the Forrester runs (models/, checkpoints/, tb/).
#
# branin-currin-mo audits BOTH objectives: 932-dim state, two surrogates
# refit per step. The replay buffer holds obs + next_obs, so it is roughly
# double the single-objective arm's -- hence 12G rather than 8G. Expect
# ~2x the wall-clock per step from the second surrogate fit.
#
# Resources are sized for the larger states (479 / 932 / 1092 dims vs Forrester's
# 414): the replay buffer holds obs + next_obs, ~4.4 GB for color at 500k.
# Slowest tasks (MahalanobisOOD) measured ~2x a CRPS task; 12 h leaves >2x
# headroom on both problems.
# =============================================================================

#SBATCH --job-name=traits_committee_problem
#SBATCH --account=YOUR_ACCOUNT         # <-- replace with your Alliance allocation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --array=0-74                   # 15 agents × 5 seeds = 75 tasks
#SBATCH --output=logs/slurm/committee_problem_%A_%a.out
#SBATCH --error=logs/slurm/committee_problem_%A_%a.err

PROBLEM="${1:?usage: sbatch train_committee_problem.slurm <branin-currin|color>}"

# ---------------------------------------------------------------------------
# Paths — edit these for your cluster
# ---------------------------------------------------------------------------
PROJECT_DIR="/scratch/$USER/traits-audit"          # edit locally, don't push
VENV_ACTIVATE="$HOME/envs/traits-audit/bin/activate" # edit locally, don't push
LOG_DIR="$PROJECT_DIR/runs/committee_${PROBLEM}_5M"

# ---------------------------------------------------------------------------
# (Agent, seed) grid — must match --array=0-74 above. Same 15 agents, in the
# same order, as the reward registry.
# ---------------------------------------------------------------------------
AGENTS=(CRPS NLL IntervalScore CalibrationError KuleshovCalibration ENCE \
        CalibrationError1Std ConformalCoverage PITUniformity IntervalCoverage \
        VarianceAlignment VarErrCorrelation UncertaintyEvolution \
        UncertaintyAnomaly MahalanobisOOD)
SEEDS=(0 1 2 3 4)

N_SEEDS=${#SEEDS[@]}
AGENT_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$((  SLURM_ARRAY_TASK_ID % N_SEEDS ))
AGENT=${AGENTS[$AGENT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"
mkdir -p logs/slurm "$LOG_DIR"

module load StdEnv/2023
module load python/3.11
source "$VENV_ACTIVATE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

# From here on, any failure fails the task (so SLURM reports FAILED, not
# COMPLETED, and --dependency=afterok holds dependent jobs back).
set -eo pipefail

# ---------------------------------------------------------------------------
# Training budget — matches the Forrester runs for direct comparability.
# ---------------------------------------------------------------------------
EPISODES=${EPISODES:-5000}
EPISODE_LENGTH=${EPISODE_LENGTH:-100}
BUFFER_SIZE=${BUFFER_SIZE:-500000}
LEARNING_STARTS=${LEARNING_STARTS:-2000}
CKPT_FREQ=${CKPT_FREQ:-50000}

echo "============================================================"
echo "Array job:     $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "Host:          $(hostname)    started $(date)"
echo "Problem:       $PROBLEM"
echo "Agent:         $AGENT"
echo "Seed:          $SEED"
echo "Log dir:       $LOG_DIR"
echo "Budget:        $EPISODES episodes × $EPISODE_LENGTH steps"
echo "============================================================"

ta-committee-train \
    --problem         "$PROBLEM" \
    --agents          "$AGENT" \
    --seeds           "$SEED" \
    --serial \
    --episodes        "$EPISODES" \
    --episode-length  "$EPISODE_LENGTH" \
    --buffer-size     "$BUFFER_SIZE" \
    --learning-starts "$LEARNING_STARTS" \
    --ckpt-freq       "$CKPT_FREQ" \
    --log-dir         "$LOG_DIR"

echo "finished at $(date)"
