#!/usr/bin/env python3
"""experiment 6: layerwise routing ablation (3 modes).

for each layer independently, replaces alpha_j(x) with a constant while
keeping all other layers intact. runs three modes:
  - mean:    alpha = E_x[alpha_j(x)] per channel (calibrated)
  - uniform: alpha = 0.5
  - ones:    alpha = 1.0 (pure bilinear)

the perplexity delta per layer reveals which layers depend most on
input-dependent routing vs static bilinear interaction, and how the
sensitivity profile differs across ablation types.

outputs (under --results_dir):
  layerwise_ablation.json
  layerwise_ablation.png
"""

import argparse
import json
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import (
    PALETTE,
    ablated_single_layer,
    add_common_args,
    compute_channel_quantities,
    compute_perplexity,
    log,
    prepare_run,
)


def plot_layerwise_ablation(results, results_dir):
    """plot per-layer perplexity delta for each ablation mode side by side."""
    ppl_baseline = results["baseline"]
    modes = [k for k in results if k not in ("baseline",)]
    n_layers = len(results[modes[0]])

    layers_x = np.arange(n_layers)
    n_modes = len(modes)
    bar_width = 0.8 / n_modes

    mode_style = {
        "mean":    {"color": PALETTE["ablation"], "label": r"$\alpha = \mathbb{E}[\alpha_j]$"},
        "uniform": {"color": PALETTE["secondary"], "label": r"$\alpha = 0.5$"},
        "ones":    {"color": PALETTE["primary"],   "label": r"$\alpha = 1$ (bilinear)"},
    }

    fig, ax = plt.subplots(figsize=(11, 4))
    for i, mode in enumerate(modes):
        layer_ppls = results[mode]
        deltas = np.array([layer_ppls.get(j, layer_ppls.get(str(j))) - ppl_baseline
                           for j in range(n_layers)])
        offset = (i - (n_modes - 1) / 2) * bar_width
        style = mode_style.get(mode, {"color": PALETTE["neutral"], "label": mode})
        ax.bar(layers_x + offset, deltas, width=bar_width, alpha=0.85,
               color=style["color"], edgecolor="0.3", linewidth=0.3,
               label=style["label"])

    ax.set_xlabel("layer")
    ax.set_ylabel(r"$\Delta$ perplexity")
    ax.set_yscale("symlog", linthresh=1)
    ax.axhline(0, color=PALETTE["neutral"], lw=0.6)
    ax.legend(framealpha=0.9, edgecolor="0.8")
    ax.set_xticks(layers_x)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "layerwise_ablation.png"))
    plt.close()
    log("done", f"saved layerwise_ablation.png -> {results_dir}/")


def run_layerwise_ablation(model, layers_info, mlp_inputs, eval_ids, device,
                           results_dir):
    log("info", "experiment 6: layerwise routing ablation (3 modes)")

    # calibrate mean alphas
    mean_alphas = {}
    for info in layers_info:
        idx = info["layer_idx"]
        _, _, alpha, _ = compute_channel_quantities(mlp_inputs[idx], info)
        mean_alphas[idx] = alpha.mean(dim=0)

    ppl_baseline = compute_perplexity(model, eval_ids, device)
    log("result", f"baseline | perplexity={ppl_baseline:.2f}")

    n_layers = len(layers_info)
    ablation_modes = [
        ("mean",    "E[alpha_j]"),
        ("uniform", "alpha=0.5"),
        ("ones",    "alpha=1.0"),
    ]

    results = {"baseline": ppl_baseline}

    for mode, desc in ablation_modes:
        layer_ppls = {}
        log("info", f"  mode: {desc}")
        for li in range(n_layers):
            with ablated_single_layer(model, layers_info, li, mode, mean_alphas):
                ppl = compute_perplexity(model, eval_ids, device)
            delta = ppl - ppl_baseline
            layer_ppls[li] = ppl
            log("result", f"  layer {li:02d}/{n_layers-1:02d} | {desc} | "
                f"perplexity={ppl:.2f} | delta={delta:+.2f}")
        results[mode] = layer_ppls

    with open(os.path.join(results_dir, "layerwise_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)
    log("done", f"saved layerwise_ablation.json -> {results_dir}/")

    plot_layerwise_ablation(results, results_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    ctx = prepare_run(args, capture_activations=True)
    run_layerwise_ablation(
        ctx["model"], ctx["layers_info"], ctx["mlp_inputs"],
        ctx["eval_ids"], ctx["device"], args.results_dir,
    )


if __name__ == "__main__":
    main()
