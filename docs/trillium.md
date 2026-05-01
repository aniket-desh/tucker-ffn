# Trillium workflow guide

How Aniket runs experiments on Trillium (SciNet, U of T) from his laptop. Written for another Claude Code instance working in any project that needs GPU access on this cluster. Mirror these conventions exactly — the workflow has been tuned around concrete failures (24h walltime burns, NumPy ABI conflicts, silent rsync failures), and any deviation is likely to reproduce one of them.

This file is intentionally **project-agnostic**. Substitute `<project>` with the actual project's directory name on Trillium (which should match the local repo's directory name) and `<venv>` with the project's venv directory under `$HOME/envs/`. Substitute `<pipeline>`, `<experiment>`, `<stage>`, `<molecule>`, etc. with whatever the host project's vocabulary is.

---

## 1. Mental model

Three machines, three roles:

1. **Local Mac** — source of truth. Code lives here. Everything is edited and committed locally.
2. **Trillium login node** (`trig-login01`) — SSH target. Runs lightweight bash, `sbatch`, `squeue`. Code is rsynced into `$SCRATCH/<project>/` on the shared filesystem.
3. **Trillium compute nodes** — only `$SCRATCH` is writable from compute. Each job activates a `~/envs/<venv>/` virtualenv that persists across jobs.

The loop is always:

```
edit locally → bash slurm/deploy.sh → (ssh + bash slurm/<run-script>.sh) → bash slurm/fetch_results.sh [tag] [--logs]
```

There is no in-place editing on Trillium. If the user says "fix the script and re-run," that means edit locally and re-deploy.

---

## 2. Connection + filesystem facts

- SSH target: `aniketrd@trillium-gpu.scinet.utoronto.ca`. Login node hostname: `trig-login01`.
- `$SCRATCH` resolves to something like `/scratch/t/trash/aniketrd` (varies by group). Always reference it as `$SCRATCH` in scripts; never hardcode.
- Code root on cluster: `$SCRATCH/<project>/`. The directory name should match the project name on the laptop.
- Logs land in `$SCRATCH/<project>/logs/`, naming `{job-name}_{jobid}.{out,err}`.
- Results land in `$SCRATCH/<project>/results/...`. They are NOT synced back automatically; see fetch script.
- Allocation: `rrg-aspuru`. Default partition: `compute_full_node`.
- Inspect job state: `scontrol show job <id>` or `sacct -j <id>`. Live queue: `squeue -u aniketrd`.

---

## 3. The three workflow scripts

### 3.1 `slurm/deploy.sh` — sync local → Trillium

Run **locally**. Does an `rsync -avz` of the project tree into `$SCRATCH/<project>/`, excluding `__pycache__`, `*.pyc`, `results/`, `logs/`, `.git/`, and any large data dirs. With `--run` it also kicks off the pipeline via SSH.

Canonical shape:

```bash
#!/bin/bash
set -e
REMOTE="aniketrd@trillium-gpu.scinet.utoronto.ca"
REMOTE_DIR="\$SCRATCH/<project>"   # NOTE: $SCRATCH is escaped; expanded on remote

ssh "$REMOTE" "mkdir -p $REMOTE_DIR"
rsync -avz \
    --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'results/' --exclude 'logs/' --exclude '.git/' \
    ./ "${REMOTE}:${REMOTE_DIR}/"

if [ "$1" = "--run" ]; then
    ssh "$REMOTE" "cd ${REMOTE_DIR} && bash slurm/<entry-script>.sh"
fi
```

Rules:
- `$SCRATCH` MUST be backslash-escaped in the local script so it expands on the remote shell. Hardcoding the full path will break on group reassignments.
- Always exclude `results/`, `logs/`, `.git/`. The user does not want laptop logs polluting the cluster, and `.git/` rsync is huge and pointless.
- Add per-project excludes for any large bundled data dirs.
- Never auto-`--run` something destructive. The flag is there because `bash deploy.sh --run` is the most common one-liner Aniket uses to ship + submit in one command.

### 3.2 `slurm/<experiment>_<vN>.sh` — generator scripts

These are the meat. They run **on Trillium** (after `ssh`), live in the project's `slurm/` dir, and emit + submit one or more chained SLURM jobs via heredocs. The naming convention is `<pipeline>_v<N>.sh` (or `<pipeline>_v<N>_<modifier>.sh` for variants like `_dt005`, `_resubmit`, `_h2`). Bumping `vN` is how Aniket tracks experiments — incremented every time the architecture, dataset, or hyperparams change in a way worth diff-able.

Anatomy (canonical):

```bash
#!/bin/bash
# =======================================================================
# <One-line title: what changed from previous version>
#
# <2–6 paragraphs explaining the motivation, what's held constant vs
#  varied from the prior version, expected outcome / decision rule.
#  This block is the user-facing "why" — write it carefully, the user
#  reads it before deciding whether to submit.>
#
# Usage: cd $SCRATCH/<project> && bash slurm/<this-script>.sh
# =======================================================================
set -euo pipefail

TAG="${1:-<default-tag>}"
DATA_DIR="results/<pipeline>/${TAG}"
DATA_PATH="${DATA_DIR}/<data-filename>"
MODEL_DIR="results/<pipeline>/${TAG}_model"

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

echo "=== <Title> ==="
echo "  tag:     ${TAG}"
echo "  dataset: ${DATA_PATH}"
echo "  model:   ${MODEL_DIR}"
echo ""

# ── Stage 1 (e.g. datagen) ───────────────────────────────────────
cat > "slurm/_<stage>_${TAG}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/<venv>/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/<project>"
export PYTHONUNBUFFERED=1
# BLAS-thread pin (only when stage uses multiprocessing.Pool; see §4)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
EOF
cat >> "slurm/_<stage>_${TAG}.sh" << EOF
python3 -m <module.path> \\
  --output ${DATA_PATH} \\
  --<arg> <value> \\
  --n_workers 16
EOF
chmod +x "slurm/_<stage>_${TAG}.sh"

JOB_STAGE1=$(sbatch --parsable \
  --partition=${PARTITION} \
  --job-name="<short-name>-${TAG}" \
  --output="logs/<short-name>_${TAG}_%j.out" \
  --error="logs/<short-name>_${TAG}_%j.err" \
  --time=24:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=16 \
  --account=${ACCOUNT} \
  "slurm/_<stage>_${TAG}.sh")
echo "[submitted] <stage>: job ${JOB_STAGE1}"

# ── Stage 2 (depends on stage 1) ─────────────────────────────────
cat > "slurm/_<next>_${TAG}.sh" << 'EOF'
... same module/venv preamble ...
EOF
cat >> "slurm/_<next>_${TAG}.sh" << EOF
python3 -m <next-module> \\
  --data_path ${DATA_PATH} \\
  ...
EOF
chmod +x "slurm/_<next>_${TAG}.sh"

JOB_STAGE2=$(sbatch --parsable \
  --dependency=afterok:${JOB_STAGE1} \
  ...
  "slurm/_<next>_${TAG}.sh")
echo "[submitted] <next>: job ${JOB_STAGE2} (after ${JOB_STAGE1})"

echo ""
echo "=== Submitted ==="
echo "  <stage>: ${JOB_STAGE1} -> <next>: ${JOB_STAGE2}"
```

Hard rules for these scripts:

1. **Two-heredoc pattern is mandatory.** First heredoc is `<< 'EOF'` (single-quoted) — preserves the literal preamble (no `$SCRATCH` expansion locally). Second heredoc is `<< EOF` (unquoted) — interpolates `${TAG}`, `${DATA_PATH}`, etc. into the python command. Never collapse to one heredoc, you will either lose `$SCRATCH` expansion on the compute node or accidentally interpolate cluster-side env vars at submission time.
2. **Write the inner job script under `slurm/_<stage>_${TAG}.sh`** — leading underscore marks it as generated and not version-controlled (gitignore `slurm/_*.sh`).
3. **`set -euo pipefail`** at the top of both the outer generator and every inner heredoc.
4. **Use `sbatch --parsable`** to capture the job ID into a variable for `--dependency=afterok:${JOB}` chaining. The standard pipeline shape is datagen → train → eval (+plot), each `afterok:` on the previous.
5. **Always set `--account=rrg-aspuru` and `--partition=compute_full_node`.**
6. **Print one `[submitted] <stage>: job ${JOB}` line per stage** and a final `=== Submitted ===` summary block. The user reads this output to copy job IDs into `squeue` / `scancel` and into the log entry he writes after a session.
7. **Header comment block is non-negotiable.** Aniket reads it before submitting. State what changed from the previous version, what's held constant, expected outcome, and (if relevant) a decision rule for what the result will tell him. Don't write generic boilerplate.
8. **Don't add features the script doesn't need.** The generator scripts are deliberately not parameterized beyond `TAG`. If a knob needs to change, bump `vN` and write a new script. This is how the user keeps an audit trail of experiments.

### 3.3 `slurm/fetch_results.sh` — pull results + logs back

Run **locally**. Two args:

```bash
bash slurm/fetch_results.sh                    # all results, no logs
bash slurm/fetch_results.sh <tag>              # one experiment's results
bash slurm/fetch_results.sh <tag> --logs       # results + logs
bash slurm/fetch_results.sh --logs             # all results + logs
```

Implementation rules (from a real bug fixed in past sessions):

- **Single rsync invocation per phase.** A previous version had multiple `rsync ... 2>/dev/null || true` calls — silently swallowed every failure. Replace with one `rsync` using `--include`/`--exclude` filters.
- **Print before/after counts** so the user can see at a glance whether anything new arrived. Per-prefix breakdown (e.g. `train_*`, `eval_*`, `data_*`) makes it obvious which job class produced output.
- **Never mask errors.** No `2>/dev/null`, no `|| true`. If rsync fails, the user needs to see it.

Canonical shape:

```bash
REMOTE="aniketrd@trillium-gpu.scinet.utoronto.ca"
REMOTE_DIR="\$SCRATCH/<project>"
TAG="${1:-}"

if [ -n "$TAG" ] && [ "$TAG" != "--logs" ]; then
    rsync -avz "${REMOTE}:${REMOTE_DIR}/results/<pipeline>/${TAG}/" \
               "${LOCAL_DIR}/results/<pipeline>/${TAG}/"
else
    rsync -avz "${REMOTE}:${REMOTE_DIR}/results/<pipeline>/" \
               "${LOCAL_DIR}/results/<pipeline>/"
fi

if [ "$FETCH_LOGS" = true ]; then
    BEFORE=$(find "${LOCAL_DIR}/logs/" -maxdepth 1 -type f | wc -l | tr -d ' ')
    rsync -avz \
        --include='<job-prefix>_*' \
        --exclude='*' \
        "${REMOTE}:${REMOTE_DIR}/logs/" \
        "${LOCAL_DIR}/logs/"
    AFTER=$(find "${LOCAL_DIR}/logs/" -maxdepth 1 -type f | wc -l | tr -d ' ')
    echo "  log files: ${BEFORE} -> ${AFTER} ($((AFTER - BEFORE)) new)"
fi
```

---

## 4. Environment rules — read this before writing any preamble

### 4.1 Standard SLURM heredoc preamble

```bash
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/<venv>/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/<project>"
export PYTHONUNBUFFERED=1
```

Every line is load-bearing:

- **`module load StdEnv/2023 python/3.11 cuda/12.2`** — minimal stack. `StdEnv/2023` is the SciNet base; `python/3.11` matches the venv Python; `cuda/12.2` is what PyTorch was built against.
- **DO NOT load `scipy-stack/2024a`.** It ships matplotlib/pandas compiled against NumPy 1.x via cvmfs and conflicts with venvs that have NumPy 2.x. Symptoms: `_ARRAY_API not found` and `numpy.dtype size changed` errors when libraries transitively import matplotlib.
- **`source "$HOME/envs/<venv>/bin/activate"`** — venvs live in `$HOME`, not `$SCRATCH`, so they persist across `$SCRATCH` purges and are shared across all jobs.
- **`export PYTHONNOUSERSITE=1` + `unset PYTHONPATH`** — together prevent inherited package paths from shadowing venv imports. Skipping these intermittently surfaces ImportErrors that look like the venv is broken when it isn't.
- **`cd "$SCRATCH/<project>"`** — every command assumes this CWD; module-style invocation (`python3 -m <pkg>.<mod>`) requires it.
- **`export PYTHONUNBUFFERED=1`** — without this, `print()` output buffers and the SLURM `.out` file looks empty for hours. Aniket relies on `tail -f logs/<job>_<id>.out` for live monitoring.

### 4.2 BLAS thread pin — REQUIRED for any multiprocessing job

If the python entry point uses `multiprocessing.Pool` (datagen scripts typically do; trainers don't), add to the preamble:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

**Why:** without scipy-stack/MKL, the venv's NumPy is OpenBLAS-backed. Without the pin, each forked worker spawns ~ncores BLAS threads and the node thrashes. Concrete past failure: a 16-worker datagen job hit 24h TimeLimit without finishing one unit of work; the same code with MKL had finished hundreds in ~11h. Match `--n_workers <N>` to the physical CPU count of the partition you're requesting.

### 4.3 SBATCH resource conventions

- **GPU jobs:** `--gpus-per-node=4 --cpus-per-task=4` (single-task, 4 GPUs, 4 CPUs). Trillium GPU nodes have 16 CPUs + 4 GPUs and a "1 GPU per 4 CPUs" usage rule, so a full-node training job is `4 GPUs × 4 CPUs = 16 CPUs` — **not 32**.
- **Datagen / multiprocessing jobs:** `--gpus-per-node=4 --cpus-per-task=16` is the de facto pattern Aniket uses (full GPU node + all 16 CPUs for workers, even when the job is CPU-only — the GPU node is what's available under his allocation). `--n_workers 16` matches.
- **CPU-only nodes** (when applicable) have 48 CPUs / 384 GB. Niagara-origin nodes have 40 physical / 80 HT / 188 GB with no local scratch.
- **Walltimes (good defaults):**
  - datagen: `--time=24:00:00` (or `48:00:00` for finer parameter grids that multiply the work)
  - train: `--time=24:00:00`
  - eval+plot: `--time=04:00:00`
- **Job names + log paths:** `--job-name="<short>-${TAG}"`, `--output="logs/<short>_${TAG}_%j.out"`, `--error="logs/<short>_${TAG}_%j.err"`. The `%j` SLURM placeholder is the job ID — keep it; the fetch script and log inspection rely on the `_<jobid>.{out,err}` suffix.
- **Account:** always `--account=rrg-aspuru`.

### 4.4 Venv bootstrap (one-time, manual on Trillium)

A `slurm/setup_env.sh` (or equivalent) should be run interactively on the login node after the first deploy:

```bash
ssh aniketrd@trillium-gpu.scinet.utoronto.ca
cd $SCRATCH/<project>
bash slurm/setup_env.sh
```

The setup script typically uses `python -m venv --system-site-packages "$HOME/envs/<venv>"` and installs the project's deps via `pip`. Note: it's OK to load `scipy-stack/2024a` *during* setup if the bootstrap needs it, but the *job-time* preamble must omit it (see §4.1). Cleanest is to skip scipy-stack everywhere.

If a job dies on `ModuleNotFoundError` for a transitive dep (concrete past case: `mpmath` from torch._dynamo → sympy), the fix is `pip install <missing>` in the venv on the login node. Then write a `<experiment>_resubmit.sh` that submits only the downstream stages (skipping completed earlier stages) — don't re-run a long datagen if its outputs are already on disk.

---

## 5. Tag + path conventions

- **`TAG`** is the experiment identifier, defaulting per script. Format: `<descriptor>_<pipeline>_v<N>[_<modifier>]`. The descriptor identifies the system being studied; `vN` increments per architectural/data change; modifier is for one-off variants.
- **Dataset path:** `results/<pipeline>/${TAG}/<data-filename>`.
- **Model output:** `results/<pipeline>/${TAG}_model/`. Trainers write checkpoints + history into this dir; eval reads from `${MODEL_DIR}/<checkpoint-filename>` and writes plots/JSON under `${MODEL_DIR}/eval/` and `${MODEL_DIR}/plots/`.
- **Reusing data across versions:** for vN where only the model changed (no datagen rerun), set `DATA_PATH` to the prior version's path and skip the datagen stage. Pattern: hard-code `DATA_PATH="results/.../<prior-tag>/<data-filename>"` and start the chain at the train sbatch (no `--dependency`).
- **Multi-seed runs:** loop over `SEEDS=(...)` in the outer generator, append `_s${SEED}` to TAG, submit train+eval pairs in parallel (no cross-seed dependency).

---

## 6. The session loop — what the user actually does

1. Edit code locally; stage a new `<experiment>_v<N>.sh` if the experiment is non-trivial.
2. `bash slurm/deploy.sh` — sync.
3. `ssh aniketrd@trillium-gpu.scinet.utoronto.ca`, then `cd $SCRATCH/<project> && bash slurm/<experiment>_v<N>.sh`. Read the `[submitted]` lines to grab job IDs.
4. Periodically: `squeue -u aniketrd` and `tail -f logs/<job>_<id>.out` to monitor.
5. When jobs complete: from local Mac, `bash slurm/fetch_results.sh <tag> --logs`.
6. Inspect locally; write a log entry; decide on next iteration.

**`bash slurm/deploy.sh --run`** is the one-shot version of steps 2–3, but it requires the deploy script to know which entry script to invoke. Aniket usually does steps 2 and 3 separately because the entry script changes per experiment.

---

## 7. Hard "don'ts" learned from incidents

- **Don't load `scipy-stack/2024a` in job-time SLURM preambles.** NumPy 1.x ABI conflict with venvs on NumPy 2.x. Symptoms: `_ARRAY_API not found`, `numpy.dtype size changed` from a transitive matplotlib import. (See §4.1.)
- **Don't use `multiprocessing.Pool` without pinning BLAS threads.** Past 24h TimeLimit with zero work units finished. (See §4.2.)
- **Don't hardcode `$SCRATCH`.** Always use the env var; group reassignments change the absolute path.
- **Don't `2>/dev/null || true` rsync calls.** Past silent fetch failures cost real debugging time.
- **Don't request `--cpus-per-task=32` on a GPU node.** GPU nodes have 16 CPUs total. Job will pend forever or fail to schedule.
- **Don't auto-amend or skip hooks.** Standard local rule, but note that on Trillium there are no hooks; just don't invent destructive shortcuts.
- **Don't edit `slurm/_*.sh` (the underscored generated files) directly.** They are regenerated by their parent `<experiment>.sh` on every run. Edit the generator instead, redeploy, re-submit.
- **Don't reuse a TAG across architectures.** `${TAG}_model/` will collide and silently overwrite checkpoints. Bump `vN` or add a modifier suffix.
- **Don't skip the header comment block.** It's the single most-read part of the script when the user comes back to a result a week later.

---

## 8. Useful command snippets

```bash
# from local Mac
bash slurm/deploy.sh                                           # sync only
bash slurm/deploy.sh --run                                     # sync + submit
bash slurm/fetch_results.sh                                    # pull all results
bash slurm/fetch_results.sh <tag> --logs                       # one tag + logs

# on Trillium (after ssh)
cd $SCRATCH/<project>
bash slurm/<experiment>_v<N>.sh                                # submit a chain
squeue -u aniketrd                                             # check queue
scontrol show job <jobid>                                      # detailed state
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed # post-mortem
tail -f logs/<jobname>_<jobid>.out                             # live monitor
scancel <jobid>                                                # cancel one
scancel -u aniketrd                                            # cancel all (use carefully)
```

---

## 9. Suggested project layout

A new project that follows this workflow should have:

```
<project>/
├── slurm/
│   ├── deploy.sh              # sync local → Trillium (§3.1)
│   ├── fetch_results.sh       # pull results + logs back (§3.3)
│   ├── setup_env.sh           # one-time venv bootstrap (§4.4)
│   ├── <experiment>_v1.sh     # generator: emits + submits a chain (§3.2)
│   ├── <experiment>_v2.sh
│   └── _<stage>_<tag>.sh      # GENERATED, gitignored (slurm/_*.sh)
├── results/                   # gitignored; populated by fetch_results.sh
├── logs/                      # gitignored; populated by fetch_results.sh
└── <source-code>/
```

Add to `.gitignore`:
```
slurm/_*.sh
results/
logs/
trillium.md
```
