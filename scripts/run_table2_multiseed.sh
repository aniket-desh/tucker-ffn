#!/usr/bin/env bash
# Multi-seed reruns for Table 2: SwiGLU and var-preserving Tucker (hc_v3),
# seeds 1 and 2 (seed 0 is already in results/exp11/swiglu_seed0/ and
# results/exp11_hc_v3/tucker_seed0/). Sequential to avoid GPU contention.
set -euo pipefail
cd "$(dirname "$0")/.."

for seed in 1 2; do
  echo "=== SwiGLU seed=$seed ==="
  .venv/bin/python experiments/exp11_train_lm.py \
    --archs swiglu --seeds "$seed" \
    --d 512 --n_heads 8 --n_layers 8 --seq_len 1024 \
    --batch_size 24 --max_tokens 100000000 --peak_lr 3e-4 --warmup_steps 200 \
    --tucker_r 128 --tucker_s 128 \
    --results_dir "results/exp11_seed${seed}"

  echo "=== Tucker hc_v3 seed=$seed ==="
  .venv/bin/python experiments/exp11_train_lm.py \
    --archs tucker --seeds "$seed" \
    --d 512 --n_heads 8 --n_layers 8 --seq_len 1024 \
    --batch_size 24 --max_tokens 100000000 --peak_lr 3e-4 --warmup_steps 200 \
    --tucker_r 128 --tucker_s 128 \
    --tucker_diagonal_bias_init --tucker_diag_bias_eps 1e-2 \
    --results_dir "results/exp11_hc_v3_seed${seed}"
done

echo "=== all seeds done ==="
