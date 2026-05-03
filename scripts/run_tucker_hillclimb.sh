#!/usr/bin/env bash
# tucker hill-climb: same lm config as exp11 but with diagonal-bias init
# (tucker C is initialized so the layer is swiglu-equivalent at init, then
# is free to deviate into off-diagonal interactions during training).
# this is a pure init/optimization change; the parameterization (note
# section IV) is unchanged. resulting checkpoint is compared head-to-head
# against the matched-budget swiglu_seed0 from exp11.

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
    --tucker_core_lr_scale 1.0 \
    --batch_size 24 \
    --max_tokens 100000000 \
    --eval_every_tokens 2000000 \
    --n_val_seqs 96 \
    --warmup_steps 200 \
    --peak_lr 3e-4 \
    --results_dir results/exp11_hc 2>&1
