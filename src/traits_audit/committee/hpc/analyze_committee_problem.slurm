#!/bin/bash
# =============================================================================
# Committee-RL on another benchmark — regenerate the core figure set
# (correlation, density, regret, learning curves) from a completed run.
#
# Submit after the training array clears (problem name is the first
# argument, and must match what train_committee_problem.slurm ran):
#     sbatch --dependency=afterok:<TRAIN_ARRAY_JOBID> \
#            src/traits_audit/committee/hpc/analyze_committee_problem.slurm branin-currin
#
# Runs scripts/make_problem_figures.sh, which chains the four
# ta-committee-analyze subcommands generalized to take --problem:
# corr-random, corr-trained, density, regret, learning-curves.
# thread-a/thread-b are NOT run here — see make_problem_figures.sh.
# =============================================================================

#SBATCH --job-name=committee_analyze_problem
#SBATCH --account=YOUR_ACCOUNT         # <-- replace with your allocation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm/analyze_problem_%j.out
#SBATCH --error=logs/slurm/analyze_problem_%j.err

PROBLEM="${1:?usage: sbatch analyze_committee_problem.slurm <branin-currin|color>}"

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

set -eo pipefail

echo "============================================================"
echo "Analyze job:  $SLURM_JOB_ID  on $(hostname)  at $(date)"
echo "Problem:      $PROBLEM"
echo "============================================================"

bash scripts/make_problem_figures.sh "$PROBLEM"

echo "finished at $(date)"
