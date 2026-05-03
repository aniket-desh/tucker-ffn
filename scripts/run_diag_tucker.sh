#!/usr/bin/env bash
# Diagonal-core Tucker from scratch: same arch as hc_v3 but core constrained
# to its superdiagonal (parameterized as a length-r vector via diagonal_only).
# One seed, same params, same 100M tokens.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python experiments/exp11_train_lm.py \
  --archs tucker_diag --seeds 0 \
  --d 512 --n_heads 8 --n_layers 8 --seq_len 1024 \
  --batch_size 24 --max_tokens 100000000 --peak_lr 3e-4 --warmup_steps 200 \
  --tucker_r 128 --tucker_s 128 \
  --results_dir results/exp11_diag

echo "=== diag tucker run done ==="
