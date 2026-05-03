#!/usr/bin/env python3
"""swiglu tensor decomposition analysis — orchestrator.

empirical validation that every swiglu ffn computes an exact input-dependent
cp decomposition of a third-order interaction tensor (note eq 10):

  A(x) = sum_j alpha_j(x) * u_j (x) w_j (x) g_j,   alpha_j(x) = sigmoid(g_j^T x)

the output of each swiglu layer decomposes as:

  y = sum_j alpha_j(x) * u_j * (w_j^T x) * (g_j^T x)

where u_j, w_j, g_j are columns of down_proj, up_proj, gate_proj respectively.
each channel contribution is c_j(x) = alpha_j(x) * (w_j^T x) * (g_j^T x).
see docs/note.pdf for full theory.

experiments (each lives in experiments/expNN_*.py and can be run standalone):
  1. numerical sanity check — verify decomposition matches forward pass
  2. routing coefficient statistics — alpha_j(x) distributions across layers
  3. channel sparsity — concentration of channel contributions |c_j(x)|
  4. routing ablation — perplexity with alpha replaced by constants
  5. top-activating tokens — semantic clustering of high-variance channels
  6. layerwise ablation — ablate routing one layer at a time
  7. interpolation sweep — dose-response from normal to fully ablated routing
  8. channel-subset ablation — ablate high/low variance or contribution channels
  9. pairing permutation — break same-index W-G coupling (joint π) vs U-only control
 10. synthetic fitting limit — verify theorem 1 separation bound on synthetic teachers
 11. lm training — train swiglu vs tucker lm at matched param count
 12. trained tucker analysis — stable rank of V_j across layers
 13. diagonal projection — c-diagonal projection + per-j svd truncation perplexity
 14. distillation — fit student swiglu/tucker to a pretrained teacher ffn

note: pythia models use standard gelu mlp, NOT swiglu. default model is
Qwen/Qwen2.5-0.5B which has the required gate_proj / up_proj / down_proj
structure. any llama/mistral/gemma-family model also works.

usage:
  python run_experiments.py
  python run_experiments.py --model Qwen/Qwen2.5-0.5B --max_tokens 4096
  python run_experiments.py --experiments 1,2,3
  python run_experiments.py --plot_only --experiments 2,3,4

  # individual scripts:
  python experiments/exp02_routing_stats.py --model Qwen/Qwen2.5-0.5B

dependencies: torch, transformers, datasets, numpy, matplotlib
"""

import argparse
import json
import os
import time

import numpy as np

from experiments.exp01_sanity_check import run_sanity_check
from experiments.exp02_routing_stats import plot_routing_stats, run_routing_stats
from experiments.exp03_channel_sparsity import (
    plot_channel_sparsity,
    run_channel_sparsity,
)
from experiments.exp04_routing_ablation import plot_ablation, run_ablation
from experiments.exp05_top_activating_tokens import run_top_activating
from experiments.exp06_layerwise_ablation import (
    plot_layerwise_ablation,
    run_layerwise_ablation,
)
from experiments.exp07_interpolation_sweep import (
    plot_interpolation_sweep,
    run_interpolation_sweep,
)
from experiments.exp08_channel_subset_ablation import (
    plot_channel_subset_ablation,
    run_channel_subset_ablation,
)
from experiments.exp09_pairing_permutation import (
    plot_pairing_permutation,
    run_pairing_permutation,
)
from lib import add_common_args, log, prepare_run, setup_plot_style


# ── plot-only mode ───────────────────────────────────────────────────────────

def replot_from_saved(exps, results_dir):
    """regenerate plots from saved .npz / .json files without loading a model."""
    t0 = time.time()

    if 2 in exps:
        path = os.path.join(results_dir, "routing_stats.npz")
        if os.path.exists(path):
            data = np.load(path)
            variances = data["variances"]
            means = data["means"]
            # reconstruct histogram tuples
            if "hist_counts" in data and "hist_edges" in data:
                hist_counts = data["hist_counts"]
                hist_edges = data["hist_edges"]
                hists = [(hist_counts[i], hist_edges) for i in range(len(hist_counts))]
            else:
                log("error", "routing_stats.npz missing histogram data — rerun experiment 2 to regenerate")
                hists = None
            if hists is not None:
                plot_routing_stats(variances, means, hists, results_dir)
        else:
            log("error", f"routing_stats.npz not found in {results_dir}/")

    if 3 in exps:
        path = os.path.join(results_dir, "channel_sparsity_stats.npz")
        if os.path.exists(path):
            data = np.load(path)
            total_ch = int(data["total_channels"]) if "total_channels" in data else 4864
            plot_channel_sparsity(
                data["frac_90_mean"], data["frac_90_std"],
                data["eff_channels"], total_ch, results_dir,
            )
        else:
            log("error", f"channel_sparsity_stats.npz not found in {results_dir}/")

    if 4 in exps:
        path = os.path.join(results_dir, "ablation_results.json")
        if os.path.exists(path):
            with open(path) as f:
                results = json.load(f)
            plot_ablation(results, results_dir)
        else:
            log("error", f"ablation_results.json not found in {results_dir}/")

    if 6 in exps:
        path = os.path.join(results_dir, "layerwise_ablation.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            plot_layerwise_ablation(data, results_dir)
        else:
            log("error", f"layerwise_ablation.json not found in {results_dir}/")

    if 7 in exps:
        path = os.path.join(results_dir, "interpolation_sweep.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            plot_interpolation_sweep(data["lambdas"], data["ppls"], results_dir)
        else:
            log("error", f"interpolation_sweep.json not found in {results_dir}/")

    if 8 in exps:
        path = os.path.join(results_dir, "channel_subset_ablation.json")
        if os.path.exists(path):
            with open(path) as f:
                results = json.load(f)
            plot_channel_subset_ablation(results, results_dir)
        else:
            log("error", f"channel_subset_ablation.json not found in {results_dir}/")

    if 9 in exps:
        path = os.path.join(results_dir, "pairing_permutation.json")
        if os.path.exists(path):
            with open(path) as f:
                results = json.load(f)
            plot_pairing_permutation(results, results_dir)
        else:
            log("error", f"pairing_permutation.json not found in {results_dir}/")

    log("done", f"plots regenerated | time={time.time() - t0:.1f}s")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="swiglu tensor decomposition analysis")
    add_common_args(parser)
    parser.add_argument(
        "--experiments", type=str, default="1,2,3,4,5",
        help="comma-separated experiment numbers to run (default: 1-5)",
    )
    parser.add_argument(
        "--plot_only", action="store_true",
        help="regenerate plots from saved results (no model loading)",
    )
    args = parser.parse_args()

    exps = set(int(e) for e in args.experiments.split(","))

    # ── plot-only mode: regenerate from saved data, no model needed ──
    if args.plot_only:
        os.makedirs(args.results_dir, exist_ok=True)
        setup_plot_style()
        log("info", f"plot-only mode | results_dir={args.results_dir}")
        replot_from_saved(exps, args.results_dir)
        return

    log("info", "swiglu tensor decomposition analysis")
    log("info", f"experiments={sorted(exps)}")
    print()

    t0_total = time.time()
    ctx = prepare_run(args, capture_activations=True)
    layers_info = ctx["layers_info"]
    mlp_inputs = ctx["mlp_inputs"]
    mlp_outputs = ctx["mlp_outputs"]
    model = ctx["model"]
    tokenizer = ctx["tokenizer"]
    analysis_ids = ctx["analysis_ids"]
    eval_ids = ctx["eval_ids"]
    device = ctx["device"]
    results_dir = args.results_dir

    if 1 in exps:
        run_sanity_check(layers_info, mlp_inputs, mlp_outputs)
        print()

    # experiment 2 outputs are also reused by experiment 5
    all_vars = None
    if 2 in exps or 5 in exps:
        all_vars, _ = run_routing_stats(layers_info, mlp_inputs, results_dir)
        print()

    if 3 in exps:
        run_channel_sparsity(layers_info, mlp_inputs, results_dir)
        print()

    if 4 in exps:
        run_ablation(model, layers_info, mlp_inputs, eval_ids, device, results_dir)
        print()

    if 5 in exps:
        run_top_activating(
            layers_info, mlp_inputs, all_vars, analysis_ids,
            tokenizer, results_dir,
        )
        print()

    if 6 in exps:
        run_layerwise_ablation(
            model, layers_info, mlp_inputs, eval_ids, device, results_dir,
        )
        print()

    if 7 in exps:
        run_interpolation_sweep(
            model, layers_info, mlp_inputs, eval_ids, device, results_dir,
        )
        print()

    if 8 in exps:
        run_channel_subset_ablation(
            model, layers_info, mlp_inputs, eval_ids, device, results_dir,
        )
        print()

    if 9 in exps:
        run_pairing_permutation(
            model, layers_info, eval_ids, device, results_dir,
        )
        print()

    log("done", f"all experiments complete | total_time={time.time() - t0_total:.1f}s")


if __name__ == "__main__":
    main()
