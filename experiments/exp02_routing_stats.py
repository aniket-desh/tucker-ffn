#!/usr/bin/env python3
"""experiment 2: routing coefficient statistics.

alpha_j(x) = sigmoid(g_j^T x) is the per-channel routing coefficient.
per-channel variance s_j = Var_x[alpha_j(x)] measures whether channel j
is a static bilinear atom (low s_j) or a genuine input-dependent router
(high s_j). this is the core empirical observable from the routed-cp
picture (note section on interpretability).

outputs (under --results_dir):
  routing_stats.npz                 — variances, means, histograms
  alpha_distribution_by_layer.png
  routing_variance_heatmap.png
  routing_variance_by_layer.png
"""

import argparse
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import (
    LAYER_CMAP,
    PALETTE,
    add_common_args,
    compute_channel_quantities,
    log,
    prepare_run,
)


def plot_routing_stats(variances, means, hists, results_dir):
    """generate all routing statistics plots from precomputed arrays.

    args:
        variances: (n_layers, m) array of per-channel variances
        means:     (n_layers, m) array of per-channel means
        hists:     list of (hist_counts, bin_edges) tuples per layer
        results_dir: output directory
    """
    n_layers = variances.shape[0]

    # ── plot 1: alpha distribution for representative layers ──
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    picks = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
    cmap = plt.get_cmap(LAYER_CMAP)
    layer_colors = [cmap(i / max(n_layers - 1, 1)) for i in range(n_layers)]

    for ax, li in zip(axes.flat, picks):
        h, e = hists[li]
        centers = (e[:-1] + e[1:]) / 2
        ax.fill_between(centers, h, alpha=0.25, color=layer_colors[li])
        ax.plot(centers, h, color=layer_colors[li], lw=1.5)
        ax.set_xlabel(r"$\alpha_j(x)$")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        mean_a = means[li].mean()
        ax.axvline(mean_a, color=PALETTE["neutral"], ls="--", lw=0.8,
                   label=f"Layer {li}, mean = {mean_a:.2f}")
        ax.legend(framealpha=0.9, edgecolor="0.8")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "alpha_distribution_by_layer.png"))
    plt.close()
    log("done", f"saved alpha_distribution_by_layer.png -> {results_dir}/")

    # ── plot 2: per-channel variance heatmap across layers ──
    sort_idx = variances.mean(axis=0).argsort()[::-1]
    n_show = min(200, variances.shape[1])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(variances[:, sort_idx[:n_show]], aspect="auto",
                   cmap="viridis", interpolation="nearest")
    ax.set_xlabel(f"Channel (sorted by mean variance, top {n_show})")
    ax.set_ylabel("Layer")
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Variance")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "routing_variance_heatmap.png"))
    plt.close()
    log("done", f"saved routing_variance_heatmap.png -> {results_dir}/")

    # ── plot 3: mean variance by layer (with iqr band) ──
    mean_var_per_layer = variances.mean(axis=1)
    q25 = np.percentile(variances, 25, axis=1)
    q75 = np.percentile(variances, 75, axis=1)
    layers_x = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.fill_between(layers_x, q25, q75, alpha=0.15, color=PALETTE["primary"],
                    label="25th–75th percentile")
    ax.plot(layers_x, mean_var_per_layer, "o-", color=PALETTE["primary"],
            ms=4, lw=1.5, label="Mean")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"Mean $\mathrm{Var}_x[\alpha_j(x)]$")
    ax.legend(framealpha=0.9, edgecolor="0.8")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "routing_variance_by_layer.png"))
    plt.close()
    log("done", f"saved routing_variance_by_layer.png -> {results_dir}/")


def run_routing_stats(layers_info, mlp_inputs, results_dir):
    log("info", "experiment 2: routing coefficient statistics")

    all_vars = []
    all_means = []
    layer_hists = []

    for info in layers_info:
        idx = info["layer_idx"]
        _, _, alpha, _ = compute_channel_quantities(mlp_inputs[idx], info)

        var_j = alpha.var(dim=0).numpy()
        mean_j = alpha.mean(dim=0).numpy()
        all_vars.append(var_j)
        all_means.append(mean_j)

        hist, edges = np.histogram(alpha.numpy().ravel(), bins=100, range=(0, 1))
        layer_hists.append((hist, edges))

        log("eval", f"layer {idx:02d} | mean_alpha={mean_j.mean():.3f} | "
            f"mean_var={var_j.mean():.2e} | max_var={var_j.max():.2e}")

    # save numerical results (including histograms for plot-only regeneration)
    variances = np.stack(all_vars)
    means = np.stack(all_means)
    hist_counts = np.stack([h for h, _ in layer_hists])
    hist_edges = layer_hists[0][1]  # all layers share the same bin edges
    np.savez(
        os.path.join(results_dir, "routing_stats.npz"),
        variances=variances,
        means=means,
        hist_counts=hist_counts,
        hist_edges=hist_edges,
    )
    log("done", f"saved routing_stats.npz -> {results_dir}/")

    plot_routing_stats(variances, means, layer_hists, results_dir)

    return all_vars, all_means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    ctx = prepare_run(args, capture_activations=True)
    run_routing_stats(ctx["layers_info"], ctx["mlp_inputs"], args.results_dir)


if __name__ == "__main__":
    main()
