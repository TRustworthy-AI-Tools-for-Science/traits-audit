#!/bin/bash
# =============================================================================
# Committee-RL v1 — the 6 NEW agents only (v0's 9 are reused unchanged).
#
# 6 agents × 5 seeds = 30 array tasks. Writes into the SAME LOG_DIR as v0 so
# the analysis CLI sees all 15 models in one models/ dir.
#
# Submit:
#     sbatch hpc/train_committee_v1_new6.slurm
#
# The v0 script (train_committee.slurm) trains the original 9. Keep both:
# v0 stays a reproducible artifact, this one is the v1 add-on.
# =============================================================================

#SBATCH --job-name=traits_committee_v1
#SBATCH --account=YOUR_ACCOUNT         # <-- replace with your Alliance allocation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --array=0-29                   # 6 agents × 5 seeds = 30 tasks
#SBATCH --output=logs/slurm/committee_v1_%A_%a.out
#SBATCH --error=logs/slurm/committee_v1_%A_%a.err

# ---------------------------------------------------------------------------
# Paths — edit these for your cluster
# ---------------------------------------------------------------------------
PROJECT_DIR="/scratch/$USER/traits-audit"          # edit locally, don't push
VENV_ACTIVATE="$HOME/envs/traits-audit/bin/activate" # edit locally, don't push
LOG_DIR="$PROJECT_DIR/runs/committee_v0_5M"   # same dir as v0; models merge

# ---------------------------------------------------------------------------
# (Agent, seed) grid — must match --array=0-29 above.
# 3 calibration variants + 3 different-signal personalities.
# ---------------------------------------------------------------------------
AGENTS=(KuleshovCalibration ENCE CalibrationError1Std \
        UncertaintyEvolution UncertaintyAnomaly MahalanobisOOD)
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

# ---------------------------------------------------------------------------
# Training budget — matches v0 for direct comparability.
# ---------------------------------------------------------------------------
EPISODES=${EPISODES:-5000}
EPISODE_LENGTH=${EPISODE_LENGTH:-100}
BUFFER_SIZE=${BUFFER_SIZE:-500000}
LEARNING_STARTS=${LEARNING_STARTS:-2000}
CKPT_FREQ=${CKPT_FREQ:-50000}

echo "============================================================"
echo "Array job:     $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "Host:          $(hostname)    started $(date)"
echo "Agent:         $AGENT (v1 new-member)"
echo "Seed:          $SEED"
echo "Log dir:       $LOG_DIR"
echo "Budget:        $EPISODES episodes × $EPISODE_LENGTH steps"
echo "============================================================"

ta-committee-train \
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
