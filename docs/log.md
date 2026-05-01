---
author: aniket
tags:
  - log
---

# implementation log

research-engineer log of experiments and code changes for the swiglu
tensor-decomposition project. newest entries on top. each entry is dated
and tagged with a version label that maps to the git commit short hash.
keep entries factual: what was implemented, what was run, what the result
was, what i'd do next. discussion / motivation belongs in `swiglu.pdf`.

reading conventions:
  - dates are absolute (YYYY-MM-DD)
  - `vN.M (hash)` ties the entry to the commit it landed in
  - `[run]` lines record a concrete command + headline numbers
  - results paths are relative to repo root unless otherwise noted

## v0.3 — 2026-05-01 (TBD): exp09 same-index pairing permutation

motivation. swiglu.pdf §III shows each ffn layer is an exact input-routed
cp decomposition with same-index coupling: hidden unit j only mixes the
j-th projected w-feature with the j-th projected g-feature. the question
for the lesswrong post is whether that same-index restriction is *binding*
— i.e., whether the model has actually learned a specific pairing the
rest of the layer depends on, or whether the choice of pairing is just a
parameterization the model is indifferent to.

experimental design.
  - per layer (one at a time, all others intact), apply a random permutation
    π of size m to the channels and measure perplexity
  - condition `joint`: permute gate_proj rows AND down_proj cols by the
    same π. this leaves G,U paired (both move to π(j)) but breaks the W-G
    and W-U same-index couplings
  - condition `u_only`: permute down_proj cols only by π_u (control —
    breaks U pairing without disturbing W-G)
  - n_seeds = 3 random permutations per layer, mean ± std reported
  - both conditions perturb the same number of weights in the same way,
    so the comparison is parameter-fair

note: permuting gate_proj AND up_proj rows by the same π is a true no-op
(channel relabeling), so to break the W-G pairing we permute only G.
permuting U separately is the control. stated this in the experiment
docstring for readers who will check.

implementation.
  - new `lib/permutation.py`: `permuted_layer()` context manager that
    swaps gate_proj rows and/or down_proj cols on a single layer, with
    full restore on exit. handles gate bias (same row permutation) and
    leaves down_proj bias untouched (it lives on the d-dim output).
  - new `experiments/exp09_pairing_permutation.py`: per-layer × per-seed
    sweep, reuses `compute_perplexity`, `prepare_run`, plot style from
    `lib/`. plot is mean ± std curves vs layer, log-y, baseline line.
  - wired into `run_experiments.py` orchestrator + plot-only mode.
  - reuse: 100% of model loading, data loading, perplexity, plot style,
    cli args. only new code is the weight-permutation manager and the
    experiment loop / plot.

[run] python3 experiments/exp09_pairing_permutation.py --n_seeds 3
  model: Qwen/Qwen2.5-0.5B
  data:  wikitext-2 test, 4096 eval tokens
  device: mps (auto-detected)
  outputs: results/pairing_permutation.json, results/pairing_permutation.png
  → results to be filled in once run completes; key numbers:
    baseline ppl, mean joint/u_only ratio across layers, layer with
    largest joint-vs-u_only gap.

next.
  - if joint / u_only ratio is large (≥ 5x) for most layers, that's the
    section-3 finding: the same-index W-G coupling is binding.
  - if comparable, the restriction is empirically toothless and the
    section-3 framing needs to shift toward "the parameterization is
    one of many equivalent ones, so the routed-cp story is descriptive
    rather than prescriptive."
  - separately, option 2 from the design discussion (learnable Π in
    front of G, train and watch whether it stays near identity) is
    queued for the tucker-ffn architectural-modification section, not
    here.

## v0.2 — 2026-04-30 (7081bd3): repo refactor

split the single-file `run_experiments.py` (~1200 lines) into a `lib/`
package + `experiments/` directory of one script per experiment. each
experiment script is now independently runnable
(`python3 experiments/exp02_routing_stats.py --model ...`) while
`run_experiments.py` stays as the orchestrator that loads the model
once and dispatches to all selected experiments.

structure.
  ```
  lib/
    __init__.py
    log_utils.py     # log()
    plot_style.py    # PALETTE, LAYER_CMAP, setup_plot_style, color cycles
    model_utils.py   # detect_device, load_model, get_swiglu_layers, load_text_data
    activations.py   # capture_mlp_io, compute_channel_quantities, get_weight_and_bias
    eval_utils.py    # compute_perplexity
    routing.py       # ConstantRouting / Interpolated / ChannelSubset + ctx managers
    runner.py        # add_common_args, prepare_run (shared cli setup)
  experiments/
    exp01_sanity_check.py
    exp02_routing_stats.py
    exp03_channel_sparsity.py
    exp04_routing_ablation.py
    exp05_top_activating_tokens.py
    exp06_layerwise_ablation.py
    exp07_interpolation_sweep.py
    exp08_channel_subset_ablation.py
  run_experiments.py    # orchestrator + --plot_only mode
  ```

cleanup beyond the split.
  - removed unused `import seaborn as sns` and `import torch.nn.functional as F`
  - made cross-module constants public: `_PALETTE → PALETTE`, etc.
  - factored `_resolve_const_alpha` to dedupe mode→α mapping that was
    duplicated in `ablated_routing` and `ablated_single_layer`
  - added `.gitignore` for `__pycache__/` and `.DS_Store`

verification. all 8 plots regenerate from existing `.npz`/`.json`
artifacts via `--plot_only`, output filenames match exactly. import
+ syntax check passes on every module.

## v0.1 — 2026-04-08 (dcafced): initial 8 experiments

single file `run_experiments.py` containing the full experimental
suite from swiglu.pdf §VI plus four extensions. all on Qwen2.5-0.5B
(24 swiglu layers, d=896, m=4864), evaluated on a 4096-token chunk
of wikitext-2 test split (analysis chunk + held-out eval chunk, no
overlap).

experiments shipped.
  1. **sanity check** — verify the cp decomposition reproduces the
     forward pass exactly. for every layer, max abs error of
     reconstructed `y = U^T (α ⊙ gate_pre ⊙ up_pre)` vs the actual mlp
     output is ~1e-7 (float32 numerical noise). decomposition is real.
  2. **routing statistics** — per-channel α variance, marginal α
     histograms by layer, variance heatmap. variance follows a u-shape
     across depth (min around layers 10–12, steep rise toward end).
     layer 23 develops bimodal α with a spike at α≈1 (always-on
     channels). high variance is concentrated in a sparse subset of
     channels per layer.
  3. **channel sparsity** — 90% mass fraction and `exp(H(p))` effective
     channel count per token. ~50–60% of channels needed for 90% of
     |c_j| total mass; effective channel count ~2500–3000 of m=4864
     across most layers, sharply lower in layers 22–23. routed-cp
     atoms are not naturally sparse — far from sae l_0 of 50–100.
  4. **routing ablation** — perplexity when α(x) is replaced
     globally by uniform 0.5, per-channel mean, or 1.0 (bilinear).
     baseline 16.8 → uniform 2.7e5, ones 7.4e5, per-channel mean
     6.4e6. all catastrophic. per-channel-mean is *worst* because
     it simultaneously throttles channels that should fire strongly
     and keeps leaking signal through channels that should be silent.
  5. **top-activating tokens** — for highest-variance channels per
     layer, the top-k tokens by α·|c|. dumped to json for manual
     inspection / clustering. preliminary look suggests semantic
     coherence in deeper layers but no rigorous clustering yet.
  6. **layerwise ablation** — single-layer constant-α replacement,
     sweep over layers × {mean, uniform, ones}. used to identify which
     layers depend most on input-dependent routing.
  7. **interpolation sweep** — α^(λ) = (1-λ)·sigmoid + λ·E[α], for
     λ ∈ {0.0, 0.1, …, 1.0}. dose-response curve from baseline
     to fully ablated.
  8. **channel-subset ablation** — ablate routing only on
     {top 10% by var, bottom 10% by var, top 10% by |c|, bottom 90%
     by var}. tests whether the small high-variance minority carries
     most of the routing signal.

infrastructure choices baked in.
  - logging follows `docs/format.md` bracketed-tag convention
  - all variable / file / metric names snake_case
  - matplotlib siam-preprint theme: serif, all spines, light dotted
    grid, 300 dpi, `coolwarm` for layer-position colormap
  - constant-α replacement implemented by swapping `mlp.act_fn` with
    a small `nn.Module` that overrides silu — avoids any forward-pass
    surgery on huggingface internals.
  - intermediate `.npz` / `.json` saves separate from plots so plot
    regeneration is decoupled from model loading (`--plot_only`).
