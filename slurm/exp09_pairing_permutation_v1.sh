#!/bin/bash
# =======================================================================
# exp09 v1: same-index pairing permutation
#
# motivation. swiglu.pdf §III claims each ffn layer is an exact input-routed
# cp decomposition with same-index coupling: hidden unit j only mixes the
# j-th projected w-feature with the j-th projected g-feature. this experiment
# tests whether that same-index restriction is *binding* — i.e., whether the
# model has learned a specific pairing the rest of the layer depends on, or
# whether the choice of pairing is just a parameterization the model is
# indifferent to.
#
# per layer (one at a time, all others intact), apply a random permutation π
# of size m and measure perplexity under two conditions:
#   joint  : permute gate_proj rows AND down_proj cols by the same π
#            (leaves G,U paired, breaks W-G coupling)
#   u_only : permute down_proj cols only by an independent π_u
#            (control: breaks U pairing without disturbing W-G)
# n_seeds = 3 random permutations per layer, mean ± std reported. both
# conditions perturb the same number of weights in the same way.
#
# decision rule. if joint/u_only ratio is large (≥5x) for most layers, the
# same-index W-G coupling is binding — section-3 finding stands. if
# comparable, the restriction is empirically toothless and section-3 framing
# needs to shift.
#
# resources. qwen2.5-0.5b on a single gpu; 24 layers × 3 seeds × 2 conds =
# 144 perplexity passes on 4096 tokens each. ~5–15 min wall on an h100, so
# 4h walltime is conservative. requests a full gpu node per the standard
# trillium pattern but uses only 1 gpu.
#
# usage (on trillium, after git pull):
#   cd $SCRATCH/swiglu && bash slurm/exp09_pairing_permutation_v1.sh
# =======================================================================
set -euo pipefail

TAG="${1:-pairing_perm_v1}"
RESULTS_DIR="results/${TAG}"

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

echo "=== exp09 pairing permutation v1 ==="
echo "  tag:         ${TAG}"
echo "  results_dir: ${RESULTS_DIR}"
echo ""

mkdir -p "logs"
mkdir -p "${RESULTS_DIR}"

# ── inner job script ─────────────────────────────────────────────────
cat > "slurm/_exp09_${TAG}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/swiglu/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/swiglu"
export PYTHONUNBUFFERED=1
# compute nodes may not have outbound internet; force hf to use cache only
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
EOF
cat >> "slurm/_exp09_${TAG}.sh" << EOF
python3 experiments/exp09_pairing_permutation.py \\
  --model Qwen/Qwen2.5-0.5B \\
  --max_tokens 4096 \\
  --n_seeds 3 \\
  --seed 42 \\
  --device cuda \\
  --results_dir ${RESULTS_DIR}
EOF
chmod +x "slurm/_exp09_${TAG}.sh"

# ── submit ──────────────────────────────────────────────────────────
JOB=$(sbatch --parsable \
  --partition=${PARTITION} \
  --job-name="exp09-${TAG}" \
  --output="logs/exp09_${TAG}_%j.out" \
  --error="logs/exp09_${TAG}_%j.err" \
  --time=04:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=4 \
  --account=${ACCOUNT} \
  "slurm/_exp09_${TAG}.sh")
echo "[submitted] exp09: job ${JOB}"

echo ""
echo "=== Submitted ==="
echo "  exp09: ${JOB}"
echo ""
echo "monitor:  tail -f logs/exp09_${TAG}_${JOB}.out"
echo "queue:    squeue -u aniketrd"
echo "fetch:    bash slurm/fetch_results.sh ${TAG} --logs    (run from local mac)"
