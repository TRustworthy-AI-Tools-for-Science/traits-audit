#!/bin/bash
# =============================================================================
# Committee-RL v1 — regenerate the full figure set from a completed run.
#
# Submit after the training array clears:
#     sbatch --dependency=afterok:<TRAIN_ARRAY_JOBID> \
#            src/traits_audit/committee/hpc/analyze_committee.slurm
#
# Runs scripts/make_all_committee_figures.sh, which chains the seven
# ta-committee-analyze subcommands (regret, corr-trained, density,
# learning-curves, thread-a, thread-b, corr-random).
# =============================================================================

#SBATCH --job-name=committee_analyze
#SBATCH --account=YOUR_ACCOUNT         # <-- replace with your allocation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/analyze_%j.out
#SBATCH --error=logs/slurm/analyze_%j.err

PROJECT_DIR="/scratch/$USER/traits-audit"          # edit locally, don't push
VENV_ACTIVATE="$HOME/envs/traits-audit/bin/activate" # edit locally, don't push

cd "$PROJECT_DIR"
mkdir -p logs/slurm

module load StdEnv/2023
module load python/3.11
source "$VENV_ACTIVATE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

echo "============================================================"
echo "Analyze job:  $SLURM_JOB_ID  on $(hostname)  at $(date)"
echo "============================================================"

bash scripts/make_all_committee_figures.sh

echo "finished at $(date)"
