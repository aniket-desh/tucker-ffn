#!/usr/bin/env bash
# tucker hill-climb v3: same architecture as v1/v2 but with the corrected
# init scales the reviewer flagged.
#
# v1/v2 used legacy init (std(C) = 1/sqrt(r) for the full core, and a
# "diagonal_bias_init" that put C_{aaa} = 1/sqrt(r) ~ 0.088 with off-diag
# noise of std 0.1/sqrt(r) ~ 0.0088 -- aggregate off-diagonal magnitude
# 0.1 sqrt(r) ~ 1.13, so the "warm-start" was actually 13x more random
# than diagonal at r=128). This produced pre-R activations of magnitude
# sqrt(r) ~ 11 instead of O(1).
#
# v3 fixes both:
#   - full core std = 1/r so that Var(z_a) ~ r^2 * 1/r^2 ~ O(1)
#   - diagonal_bias_init: C[a,a,a] = 1, off-diag std = eps/r with eps=1e-2,
#     so the layer evaluates exactly the swiglu recovery form z_a =
#     p_a * silu(q_a) at init plus tiny noise of aggregate magnitude
#     ~ eps = 1e-2 << 1.
# this is the right swiglu warm-start. parameterization is unchanged.

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source .env

python experiments/exp11_train_lm.py \
    --archs tucker \
    --seeds 0 \
    --d 512 --n_heads 8 --n_layers 8 \
    --vocab_size 50257 --seq_len 1024 \
    --tucker_r 128 --tucker_s 128 \
    --tucker_diagonal_bias_init \
    --tucker_diag_bias_eps 1e-2 \
    --tucker_core_lr_scale 1.0 \
    --batch_size 24 \
    --max_tokens 100000000 \
    --eval_every_tokens 2000000 \
    --n_val_seqs 96 \
    --warmup_steps 200 \
    --peak_lr 3e-4 \
    --results_dir results/exp11_hc_v3 2>&1
