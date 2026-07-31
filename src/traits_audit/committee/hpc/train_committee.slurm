#!/bin/bash
# =============================================================================
# Committee-RL v0 — solo SAC training, one (agent, seed) per array task.
#
# 9 agents × 5 seeds = 45 array tasks. Each task trains a single agent on a
# single seed and saves its model + normalizer into a shared models/ dir.
#
# Submit:
#     sbatch hpc/train_committee.slurm
#
# Monitor:
#     squeue -u $USER
#     tail -f logs/slurm/committee_<JOB>_<TASKID>.out
#
# Output layout (under $LOG_DIR):
#     models/{agent}_seed{i}.zip
#     models/{agent}_seed{i}.normalizer.npz
#     checkpoints/{agent}_seed{i}/step_*.zip
#     tb/{agent}_seed{i}/
#
# After all tasks finish, run the deployment demo (loads one seed at a time):
#     ta-committee-demo --models-dir $LOG_DIR/models
# =============================================================================

#SBATCH --job-name=traits_committee
#SBATCH --account=rrg-j3goals          # <-- replace with your Alliance allocation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --array=0-44                   # 9 agents × 5 seeds = 45 tasks
#SBATCH --output=logs/slurm/committee_%A_%a.out
#SBATCH --error=logs/slurm/committee_%A_%a.err

# ---------------------------------------------------------------------------
# Paths — edit these for your cluster
# ---------------------------------------------------------------------------
PROJECT_DIR="/scratch/$USER/traits-audit"
VENV_ACTIVATE="$HOME/envs/traits-audit/bin/activate"
LOG_DIR="$PROJECT_DIR/runs/committee_v0_5M"

# ---------------------------------------------------------------------------
# (Agent, seed) grid — must match --array=0-44 above.
# ---------------------------------------------------------------------------
AGENTS=(CRPS NLL IntervalScore CalibrationError ConformalCoverage \
        PITUniformity IntervalCoverage VarianceAlignment VarErrCorrelation)
SEEDS=(0 1 2 3 4)

# Decompose the linear array index into (agent, seed).
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

# Single-core per task — pin BLAS / OpenMP so we don't oversubscribe when
# multiple array tasks land on the same node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# CPU-only (SAC on Box(1,) actions doesn't benefit from GPU).
export CUDA_VISIBLE_DEVICES=""

# ---------------------------------------------------------------------------
# Training budget — production scale (override on the sbatch command line if
# needed, e.g. `sbatch --export=EPISODES=2000 hpc/train_committee.slurm`)
# ---------------------------------------------------------------------------
EPISODES=${EPISODES:-5000}
EPISODE_LENGTH=${EPISODE_LENGTH:-100}
BUFFER_SIZE=${BUFFER_SIZE:-500000}
LEARNING_STARTS=${LEARNING_STARTS:-2000}
CKPT_FREQ=${CKPT_FREQ:-50000}

echo "============================================================"
echo "Array job:     $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "Host:          $(hostname)    started $(date)"
echo "Agent:         $AGENT"
echo "Seed:          $SEED"
echo "Log dir:       $LOG_DIR"
echo "Budget:        $EPISODES episodes × $EPISODE_LENGTH steps"
echo "Buffer:        $BUFFER_SIZE"
echo "Learn starts:  $LEARNING_STARTS"
echo "Ckpt freq:     $CKPT_FREQ"
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
