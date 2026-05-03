#!/usr/bin/env bash
# tucker hill-climb v2: same as v1 (diagonal_bias_init) but pushes the core
# learning rate 2x harder than the rest of the model. motivation: P, Q, R
# only need to learn d-scale latent dictionaries while C carries r^3 entries
# of cross-channel structure -- if the off-diagonals get washed out by the
# diagonal entries' early gradient signal, scaling C's lr up should help
# the off-diagonals find their useful values faster. parameterization
# unchanged, init unchanged from v1.

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
    --tucker_core_lr_scale 2.0 \
    --batch_size 24 \
    --max_tokens 100000000 \
    --eval_every_tokens 2000000 \
    --n_val_seqs 96 \
    --warmup_steps 200 \
    --peak_lr 3e-4 \
    --results_dir results/exp11_hc_v2 2>&1
