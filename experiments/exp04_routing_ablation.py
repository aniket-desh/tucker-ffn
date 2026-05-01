#!/usr/bin/env python3
"""experiment 4: routing ablation (perplexity).

measure perplexity when routing coefficients are replaced by constants.
this directly tests whether the input-dependent routing alpha_j(x) =
sigmoid(g_j^T x) carries meaningful information beyond what a static
bilinear interaction provides. large perplexity increase = routing
matters; small increase = the gate mostly acts as a fixed scale.

outputs (under --results_dir):
  ablation_results.json
  ablation_perplexity.png
"""

import argparse
import json
import os
import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import (
    PALETTE,
    ablated_routing,
    add_common_args,
    compute_channel_quantities,
    compute_perplexity,
    log,
    prepare_run,
)


def plot_ablation(results, results_dir):
    """generate ablation bar chart from results dict."""
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["baseline", r"$\alpha=0.5$" + "\n(uniform)",
             r"$\alpha=\mathbb{E}[\alpha_j]$" + "\n(per-ch mean)",
             r"$\alpha=1$" + "\n(bilinear)"]
    ppls = [results["baseline"], results["uniform"], results["mean"], results["ones"]]
    colors = [PALETTE["primary"]] + [PALETTE["ablation"]] * 3

    bars = ax.bar(names, ppls, color=colors, alpha=0.85, width=0.55,
                  edgecolor="0.3", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_ylabel("perplexity")

    for bar, val in zip(bars, ppls):
        label = f"{val:.1f}" if val < 1000 else f"{val:.1e}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.3, label,
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim(top=max(ppls) * 8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ablation_perplexity.png"))
    plt.close()
    log("done", f"saved ablation_perplexity.png -> {results_dir}/")


def run_ablation(model, layers_info, mlp_inputs, eval_ids, device, results_dir):
    log("info", "experiment 4: routing ablation")

    # calibrate: compute per-channel mean alpha from analysis data
    mean_alphas = {}
    for info in layers_info:
        idx = info["layer_idx"]
        _, _, alpha, _ = compute_channel_quantities(mlp_inputs[idx], info)
        mean_alphas[idx] = alpha.mean(dim=0)  # (m,)

    # baseline
    ppl_baseline = compute_perplexity(model, eval_ids, device)
    log("result", f"baseline | perplexity={ppl_baseline:.2f}")

    results = {"baseline": ppl_baseline}

    modes = [
        ("uniform", "alpha=0.5"),
        ("mean", "alpha=E[alpha_j]"),
        ("ones", "alpha=1.0 (pure bilinear)"),
    ]

    for mode, desc in modes:
        with ablated_routing(model, layers_info, mode, mean_alphas):
            ppl = compute_perplexity(model, eval_ids, device)
        delta = ppl - ppl_baseline
        log("result", f"{desc} | perplexity={ppl:.2f} | delta={delta:+.2f}")
        results[mode] = ppl

    with open(os.path.join(results_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log("done", f"saved ablation_results.json -> {results_dir}/")

    plot_ablation(results, results_dir)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    ctx = prepare_run(args, capture_activations=True)
    run_ablation(
        ctx["model"], ctx["layers_info"], ctx["mlp_inputs"],
        ctx["eval_ids"], ctx["device"], args.results_dir,
    )


if __name__ == "__main__":
    main()
