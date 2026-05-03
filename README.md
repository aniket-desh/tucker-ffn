# Tucker-core FFN

Code accompanying the paper *SwiGLU as a Routed CP Tensor Model*.

The library provides a Tucker-core generalization of SwiGLU and the empirical
machinery used to compare the two: the routed-CP routing-statistics probes on
pretrained models, the synthetic teacher-student setup that verifies the
separation bound directly, the from-scratch matched-budget LM training loop,
and the post-hoc analyses (stable rank, diagonal projection, distillation,
pairing-permutation, layerwise routing ablation, cross-model robustness).

## Setup

The project uses [uv](https://github.com/astral-sh/uv) for dependency
management. Torch is pinned to `2.6.0+cu124`; the pinning is load-bearing on
hosts whose CUDA driver predates the NCCL 2.27 ABI required by torch
`2.11+cu130`.

```bash
uv sync
```

## Layout

```
lib/                     core library (importable as `lib`)
  tucker_ffn.py          TuckerFFN, SwiGLUFFN, SwiGLUFFNAligned
  lm.py                  minimal LLaMA-style LM with swappable FFN
  routing.py             constant-/interpolated-/subset-alpha context managers
  permutation.py         layer-permutation context managers (exp09)
  activations.py         MLP I/O capture + per-channel CP quantities
  model_utils.py         HF model loading, swiglu-layer enumeration
  eval_utils.py          teacher-forced perplexity
  runner.py              shared CLI args + analysis-data loader
  log_utils.py           bracketed-tag logging
  plot_style.py          SIAM-style matplotlib theme + palette

experiments/             one file per paper experiment, runnable standalone
  exp02_routing_stats.py            §3 routing-coefficient distributions
  exp04_routing_ablation.py         §3 constant-α perplexity ablation
  exp09_pairing_permutation.py      Appendix A1 same-index pairing
  exp09b_noop_control.py            Appendix A1 no-op control
  exp10_synthetic_fitting.py        §5.1 synthetic teacher-student knee at m=k²
  exp10_svd_construction.py         §5.1 analytic SVD upper-bound construction
  exp11_train_lm.py                 §5.4 from-scratch LM training
  exp12_trained_tucker_analysis.py  §5.2 stable rank of V_j = R C^(j)
  exp13_diagonal_projection.py      §5.2 λ-sweep + ρ-truncation
  exp14b_tucker_teacher_distillation.py  §5.3 layer-level expressivity gap
  exp17_robustness.py               Appendix robustness to model family
  exp_layerwise_alpha.py            Appendix layerwise constant-α ablation
  assemble_figures.py               compose paper-ready figures + caption stubs

scripts/                 paper-snippet generators + multi-seed orchestration
  make_compute_accounting.py        FLOPs/token + measured throughput
  make_table2_seeds.py              aggregate multi-seed Table 2 (n=3 each)
  make_repro_appendix.py            full hyperparameter appendix
  make_diag_row.py                  diagonal-only Tucker row for Table 2
  rerun_eval_32k.py                 robustness on a 32K-token WikiText-2 chunk
  run_table2_multiseed.sh           drive seeds 1–2 of swiglu/tucker
  run_diag_tucker.sh                drive a diagonal-only Tucker run
```

## Reproducing the paper figures

Each experiment writes its outputs under `results/` and is independent;
the order below mirrors the paper.

```bash
# §3 — routing-coefficient framework on Qwen2.5-0.5B
.venv/bin/python experiments/exp02_routing_stats.py
.venv/bin/python experiments/exp04_routing_ablation.py
.venv/bin/python experiments/exp09_pairing_permutation.py
.venv/bin/python experiments/exp09b_noop_control.py

# §5.1 — synthetic teacher-student, separation bound + SVD construction
.venv/bin/python experiments/exp10_synthetic_fitting.py
.venv/bin/python experiments/exp10_svd_construction.py

# §5.4 — from-scratch matched-budget LM training (~22 min/run on A100)
.venv/bin/python experiments/exp11_train_lm.py \
  --archs swiglu,tucker --seeds 0,1,2 \
  --tucker_diagonal_bias_init --tucker_diag_bias_eps 1e-2

# §5.2 — post-hoc analyses on the trained Tucker LM
.venv/bin/python experiments/exp12_trained_tucker_analysis.py \
  --ckpt_glob "results/exp11_hc_v3/tucker_seed*/checkpoint_final.pt" \
  --results_dir results/exp12_hc_v3
.venv/bin/python experiments/exp13_diagonal_projection.py \
  --ckpt_glob "results/exp11_hc_v3/tucker_seed*/checkpoint_final.pt" \
  --results_dir results/exp13_hc_v3

# §5.3 — distillation
.venv/bin/python experiments/exp14b_tucker_teacher_distillation.py \
  --ckpt results/exp11_hc_v3/tucker_seed0/checkpoint_final.pt

# Appendix — cross-model robustness, layerwise ablation
.venv/bin/python experiments/exp17_robustness.py
.venv/bin/python experiments/exp_layerwise_alpha.py

# Compose paper figures + caption stubs
.venv/bin/python experiments/assemble_figures.py --tucker_variant hc_v3

# Generate paper snippets (Table 2 caption, repro appendix, etc.)
.venv/bin/python scripts/make_table2_seeds.py
.venv/bin/python scripts/make_compute_accounting.py
.venv/bin/python scripts/make_repro_appendix.py
.venv/bin/python scripts/make_diag_row.py
```

`assemble_figures.py` produces `fig_*.png`, `fig_*.pdf`, and matching
`fig_*.tex` `\includegraphics` stubs under `results/figures/`. The
`scripts/make_*.py` helpers write paper snippets to `snippets/`.

## Hardware

Experiments were run on a single NVIDIA A100 (UIUC NCSA Delta).
LM training is roughly 22 minutes per 100M-token seed; the synthetic
fitting and post-hoc analyses are at most a few minutes each.
