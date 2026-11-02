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
# Runs scripts/make_problem_figures.sh, which chains every
# ta-committee-analyze subcommand for this problem: corr-random,
# corr-trained, density, regret, learning-curves, thread-b, thread-a.
#
# The two thread steps dominate the runtime. To run them as a parallel
# array instead, use threads_committee_problem.slurm + gather_threads.slurm
# (see README.md) and drop them from make_problem_figures.sh.
# =============================================================================

#SBATCH --job-name=committee_analyze_problem
#SBATCH --account=YOUR_ACCOUNT         # <-- replace with your allocation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00   # thread-a/thread-b dominate; see README for the
                          # parallel array alternative if this is too long
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
