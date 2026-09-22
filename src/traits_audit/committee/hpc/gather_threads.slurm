#!/bin/bash
# =============================================================================
# Merge the thread-a / thread-b CSV shards and render their figures.
#
# Pure rendering: every number already exists in the shards, so this loads no
# trained models and finishes in seconds. Submit with a dependency on the
# array job so it fires as soon as the last shard lands:
#
#     ARRAY=$(sbatch --parsable \
#         src/traits_audit/committee/hpc/threads_committee_problem.slurm color)
#     sbatch --dependency=afterok:$ARRAY \
#         src/traits_audit/committee/hpc/gather_threads.slurm color
#
# afterok means a failed shard blocks the gather rather than producing a
# figure with a silently missing policy or ablation bar.
# =============================================================================

#SBATCH --job-name=committee_gather
#SBATCH --account=YOUR_ACCOUNT         # <-- replace with your Alliance allocation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/gather_%j.out
#SBATCH --error=logs/slurm/gather_%j.err

PROBLEM="${1:?usage: sbatch gather_threads.slurm <forrester|branin-currin|color>}"

PROJECT_DIR="/scratch/$USER/traits-audit"            # edit locally, don't push
VENV_ACTIVATE="$HOME/envs/traits-audit/bin/activate" # edit locally, don't push

case "$PROBLEM" in
    forrester) RESULTS="$PROJECT_DIR/_results/committee_v1" ;;
    *)         RESULTS="$PROJECT_DIR/_results/committee_${PROBLEM}" ;;
esac

cd "$PROJECT_DIR"
mkdir -p logs/slurm

module load StdEnv/2023
module load python/3.11
source "$VENV_ACTIVATE"

export MPLBACKEND=Agg
set -eo pipefail

echo "============================================================"
echo "Gather job: $SLURM_JOB_ID  on $(hostname)  at $(date)"
echo "Problem:    $PROBLEM   ->   $RESULTS"
echo "============================================================"

ta-committee-analyze thread-gather b --output-dir "$RESULTS/threadB"
ta-committee-analyze thread-gather a --output-dir "$RESULTS/threadA"

echo "finished at $(date)"
