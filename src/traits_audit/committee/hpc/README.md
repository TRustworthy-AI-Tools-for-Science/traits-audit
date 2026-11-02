# HPC scripts for committee-RL training

## Files in this directory

- **`train_committee.slurm`** — SLURM job-array template (9 agents × 5 seeds
  = 45 tasks). Calls `ta-committee-train` once per (agent, seed) tuple. Edit
  `--account`, `PROJECT_DIR`, `VENV_ACTIVATE`, and `LOG_DIR` at the top for
  your cluster. Submit with `sbatch hpc/train_committee.slurm`.
- **`train_committee_problem.slurm`** — the same for Branin-Currin / colour
  (15 agents × 5 seeds = 75 tasks); takes the problem name as `$1`.
- **`analyze_committee_problem.slurm`** — the serial figure set for one
  problem (correlation, density, regret, learning curves, and — since the
  threads were generalized — thread-a/thread-b too).
- **`threads_committee_problem.slurm`** + **`gather_threads.slurm`** — the
  *parallel* path for the threads specifically. See below.

## Running the Thread A/B bake-offs in parallel

Serially the threads dominate the analysis: thread-b alone is 5 policies ×
20 seeds × 100 steps plus two 15-agent leave-one-out ablations (~7k
rollouts, each refitting a bootstrap surrogate at every step). Hours per
problem. The work splits cleanly by policy and by ablated agent:

```bash
P=color   # or forrester, branin-currin
ARRAY=$(sbatch --parsable src/traits_audit/committee/hpc/threads_committee_problem.slurm $P)
sbatch --dependency=afterok:$ARRAY src/traits_audit/committee/hpc/gather_threads.slurm $P
```

17 tasks: task 0 is the thread-b policy bake-off, task 1 is thread-a, and
tasks 2–16 are one leave-one-out ablation shard per agent. Each writes its
own CSV shard; `gather_threads.slurm` merges them and renders the figures
(no models loaded, seconds to run).

**Run `regret` first.** thread-a reads `regret_test.json` to pick its
best-solo reference, so that file must exist before the array starts.

**Sharding is exact.** Episode seeds are drawn *before* policies are
filtered, so every shard evaluates the same 20 episodes and the merged
result is byte-identical to a single-process run. This is what keeps the
paired Wilcoxon tests valid; `read_thread_csv` refuses to merge shards whose
seeds disagree rather than silently producing an invalid comparison.
`afterok` is deliberate — a failed shard blocks the gather instead of
yielding a figure with a missing policy or ablation bar.

## How a single array task works

Each array task computes its (agent, seed) from `$SLURM_ARRAY_TASK_ID` and runs:

```bash
ta-committee-train \
    --agents "$AGENT" \
    --seeds "$SEED" \
    --serial \
    --episodes 5000 \
    --buffer-size 500000 \
    --learning-starts 2000 \
    --ckpt-freq 50000 \
    --log-dir "$LOG_DIR"
```

`--serial` is required: it skips the in-process `ProcessPoolExecutor` so a
single SLURM task doesn't try to spawn more workers.

`--seeds N` (one seed) produces `{agent}_seed{N}.zip` rather than the
single-seed default `{agent}.zip`. The multi-seed naming is what makes
deployment scripts able to disambiguate across array tasks.

## Output layout

For `--log-dir $LOG_DIR`, agents = `[CRPS, NLL, ...]`, seeds = `[0, 1, 2, 3, 4]`:

```
$LOG_DIR/
    models/
        CRPS_seed0.zip
        CRPS_seed0.normalizer.npz
        CRPS_seed1.zip
        ...
        VarErrCorrelation_seed4.zip
    checkpoints/
        CRPS_seed0/
            step_50000_steps.zip
            step_100000_steps.zip
            ...
    tb/
        CRPS_seed0/
            events.out.tfevents.*
        ...
```

Open TensorBoard with `tensorboard --logdir $LOG_DIR/tb` to compare all
(agent, seed) curves side-by-side.

## Deployment (after training finishes)

The current `ta-committee-demo` loads `{agent}.zip` filenames (single-seed
default). For multi-seed deployment, run the demo per seed:

```bash
for SEED in 0 1 2 3 4; do
    cp $LOG_DIR/models/CRPS_seed${SEED}.zip $LOG_DIR/models/CRPS.zip
    # ... repeat for all 9 agents ...
    ta-committee-demo --models-dir $LOG_DIR/models --seed $((SEED + 1000))
done
```

A v1 `--models-suffix` flag to `ta-committee-demo` would make this cleaner;
for now manual renaming or symlinking works.

## Resource hints (per array task)

- **CPU**: 1 core. SAC training on Box(1,) actions is single-threaded.
  Pin BLAS with `OMP_NUM_THREADS=1` etc. (handled in the SLURM script) so
  multiple array tasks landing on the same node don't oversubscribe.
- **Memory**: ~2 GB peak with `--buffer-size 500000` (replay buffer:
  500k × state dim 414 × float32 ≈ 800 MB + SB3/torch overhead).
- **Time**:
  - `--episodes 5000`: ~2 hr for most agents, ~3.5 hr for PITUniformity
    (slowest due to scipy's `kstest`).
  - `--episodes 10000`: roughly 2× the above.
  - The 6-hour `#SBATCH --time` in the template covers all 9 agents at 5000
    episodes; bump for PIT if running 10000+.

## Overriding the budget without editing the script

```bash
sbatch --export=EPISODES=10000,BUFFER_SIZE=1000000 hpc/train_committee.slurm
```

The template reads `EPISODES`, `EPISODE_LENGTH`, `BUFFER_SIZE`,
`LEARNING_STARTS`, `CKPT_FREQ` from the environment with sane defaults.

## Submitting your own grid

To use a different seed list or agent subset, edit the `AGENTS=` and
`SEEDS=` arrays at the top of the SLURM script and update `--array=0-N`
to match `n_agents × n_seeds - 1`.

Example: 3 agents × 10 seeds:

```bash
AGENTS=(CRPS PITUniformity VarErrCorrelation)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
# #SBATCH --array=0-29
```
