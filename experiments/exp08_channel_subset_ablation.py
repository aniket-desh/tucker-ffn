#!/usr/bin/env python3
"""experiment 8: channel-subset ablation.

ablate routing for specific channel subsets to test which channels matter.

conditions:
  - top 10% by variance (high-variance = most input-dependent routing)
  - bottom 10% by variance (low-variance = near-static atoms)
  - top 10% by mean |c_j| (highest-contribution channels)
  - bottom 90% by variance (freeze the boring majority)

this directly tests whether the small minority of high-variance channels
carries most of the routing signal, as the cp decomposition suggests.

outputs (under --results_dir):
  channel_subset_ablation.json
  channel_subset_ablation.png
"""

import argparse
import json
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import (
    COLOR_CYCLE,
    PALETTE,
    add_common_args,
    channel_subset_ablation,
    compute_channel_quantities,
    compute_perplexity,
    log,
    prepare_run,
)


def plot_channel_subset_ablation(results, results_dir):
    """bar chart comparing perplexity under different channel-subset ablations."""
    fig, ax = plt.subplots(figsize=(8, 4))

    conditions = list(results.keys())
    ppls = [results[c] for c in conditions]

    # color: baseline=primary, ablation conditions=cycle
    n = len(conditions)
    colors = [PALETTE["primary"]] + [COLOR_CYCLE[i % len(COLOR_CYCLE)]
                                      for i in range(1, n)]

    bars = ax.bar(range(n), ppls, color=colors, alpha=0.85,
                  edgecolor="0.3", linewidth=0.4)
    ax.set_xticks(range(n))
    ax.set_xticklabels(conditions, fontsize=9, rotation=15, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("perplexity")

    for bar, val in zip(bars, ppls):
        label = f"{val:.1f}" if val < 1000 else f"{val:.1e}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.4, label,
                ha="center", va="bottom", fontsize=8)

    ax.set_ylim(top=max(ppls) * 8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "channel_subset_ablation.png"))
    plt.close()
    log("done", f"saved channel_subset_ablation.png -> {results_dir}/")


def run_channel_subset_ablation(model, layers_info, mlp_inputs, eval_ids,
                                device, results_dir):
    log("info", "experiment 8: channel-subset ablation")

    # compute per-channel statistics from analysis data
    mean_alphas = {}
    channel_vars = {}
    channel_mean_abs_c = {}
    m = layers_info[0]["gate_proj"].weight.shape[0]

    for info in layers_info:
        idx = info["layer_idx"]
        _, _, alpha, c = compute_channel_quantities(mlp_inputs[idx], info)
        mean_alphas[idx] = alpha.mean(dim=0)
        channel_vars[idx] = alpha.var(dim=0).numpy()
        channel_mean_abs_c[idx] = c.abs().mean(dim=0).numpy()

    ppl_baseline = compute_perplexity(model, eval_ids, device)
    log("result", f"baseline | perplexity={ppl_baseline:.2f}")

    k_top = max(1, m // 10)       # top 10%
    k_bot = max(1, m // 10)       # bottom 10%
    k_majority = m - k_top        # bottom 90%

    conditions = {}

    def make_masks(selector_fn, description):
        masks = {}
        for info in layers_info:
            idx = info["layer_idx"]
            mask = torch.zeros(m, dtype=torch.bool)
            indices = selector_fn(idx)
            mask[indices] = True
            masks[idx] = mask
        n_ablated = int(mask.sum())
        log("info", f"  {description} | channels_ablated={n_ablated}/{m}")
        return masks

    # condition 1: ablate top 10% by variance
    conditions["top 10%\nby variance"] = make_masks(
        lambda idx: np.argsort(channel_vars[idx])[-k_top:],
        "top 10% by variance",
    )

    # condition 2: ablate bottom 10% by variance
    conditions["bottom 10%\nby variance"] = make_masks(
        lambda idx: np.argsort(channel_vars[idx])[:k_bot],
        "bottom 10% by variance",
    )

    # condition 3: ablate top 10% by mean |c_j|
    conditions["top 10%\nby |c_j|"] = make_masks(
        lambda idx: np.argsort(channel_mean_abs_c[idx])[-k_top:],
        "top 10% by mean |c_j|",
    )

    # condition 4: ablate bottom 90% by variance (keep only the routers)
    conditions["bottom 90%\nby variance"] = make_masks(
        lambda idx: np.argsort(channel_vars[idx])[:k_majority],
        "bottom 90% by variance",
    )

    results = {"baseline": ppl_baseline}

    for name, masks in conditions.items():
        with channel_subset_ablation(model, layers_info, masks, mean_alphas):
            ppl = compute_perplexity(model, eval_ids, device)
        delta = ppl - ppl_baseline
        log("result", f"{name.replace(chr(10), ' ')} | perplexity={ppl:.2f} | "
            f"delta={delta:+.2f}")
        results[name] = ppl

    with open(os.path.join(results_dir, "channel_subset_ablation.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("done", f"saved channel_subset_ablation.json -> {results_dir}/")

    plot_channel_subset_ablation(results, results_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    ctx = prepare_run(args, capture_activations=True)
    run_channel_subset_ablation(
        ctx["model"], ctx["layers_info"], ctx["mlp_inputs"],
        ctx["eval_ids"], ctx["device"], args.results_dir,
    )


if __name__ == "__main__":
    main()
