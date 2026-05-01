#!/usr/bin/env python3
"""experiment 3: channel sparsity and concentration.

c_j(x) = alpha_j(x) * (w_j^T x) * (g_j^T x) is the scalar channel
contribution (note eq 4). for each token we ask: how many channels
carry most of the signal?

two metrics:
  - frac_90: fraction of channels needed for 90% of sum_j |c_j(x)|
  - eff_channels: exp(entropy) of normalized |c_j(x)|, the effective
    number of active channels (= m if uniform, << m if sparse)

outputs (under --results_dir):
  channel_sparsity_stats.npz
  channel_sparsity.png
"""

import argparse
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import (
    PALETTE,
    add_common_args,
    compute_channel_quantities,
    log,
    prepare_run,
)


def plot_channel_sparsity(frac_90_mean, frac_90_std, eff_channels, total_channels,
                          results_dir):
    """generate channel sparsity plots from precomputed arrays."""
    n_layers = len(frac_90_mean)
    layers_x = np.arange(n_layers)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax1.fill_between(layers_x,
                     frac_90_mean - frac_90_std,
                     frac_90_mean + frac_90_std,
                     alpha=0.15, color=PALETTE["primary"])
    ax1.plot(layers_x, frac_90_mean, "o-", color=PALETTE["primary"], ms=4, lw=1.5)
    ax1.set_xlabel("layer")
    ax1.set_ylabel("fraction of channels")
    ax1.set_ylim(0, 1)
    ax1.axhline(0.5, color=PALETTE["neutral"], ls=":", lw=0.8, alpha=0.5)
    ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes, va="top")

    ax2.plot(layers_x, eff_channels, "s-", color=PALETTE["accent"], ms=4, lw=1.5)
    ax2.axhline(total_channels, color=PALETTE["neutral"], ls="--", lw=0.8, alpha=0.5,
                label=f"total = {total_channels}")
    ax2.set_xlabel("layer")
    ax2.set_ylabel("effective channels")
    ax2.legend(framealpha=0.9, edgecolor="0.8", loc="lower left")
    ax2.text(0.02, 0.95, "(b)", transform=ax2.transAxes, va="top")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "channel_sparsity.png"))
    plt.close()
    log("done", f"saved channel_sparsity.png -> {results_dir}/")


def run_channel_sparsity(layers_info, mlp_inputs, results_dir):
    log("info", "experiment 3: channel sparsity and concentration")

    m = layers_info[0]["gate_proj"].weight.shape[0]
    frac_90_means = []
    frac_90_stds = []
    eff_ch_means = []

    for info in layers_info:
        idx = info["layer_idx"]
        _, _, _, c = compute_channel_quantities(mlp_inputs[idx], info)
        c_abs = c.abs()

        # fraction of channels for 90% of total |c_j|
        sorted_c, _ = c_abs.sort(dim=1, descending=True)
        cum = sorted_c.cumsum(dim=1)
        total = c_abs.sum(dim=1, keepdim=True)
        above_90 = cum >= 0.9 * total
        # argmax on bool gives first True; +1 because we want count not index
        first_above = above_90.float().argmax(dim=1) + 1
        frac_90 = first_above.float() / m

        # effective channel count: exp(entropy) of normalized |c_j|
        p = c_abs / (total + 1e-10)
        entropy = -(p * (p + 1e-10).log()).sum(dim=1)
        eff_channels = entropy.exp()

        f90_mean = frac_90.mean().item()
        f90_std = frac_90.std().item()
        eff_mean = eff_channels.mean().item()
        frac_90_means.append(f90_mean)
        frac_90_stds.append(f90_std)
        eff_ch_means.append(eff_mean)

        log("eval", f"layer {idx:02d} | frac_for_90pct={f90_mean:.3f} | eff_channels={eff_mean:.1f}/{m}")

    frac_arr = np.array(frac_90_means)
    std_arr = np.array(frac_90_stds)
    eff_arr = np.array(eff_ch_means)
    np.savez(
        os.path.join(results_dir, "channel_sparsity_stats.npz"),
        frac_90_mean=frac_arr,
        frac_90_std=std_arr,
        eff_channels=eff_arr,
        total_channels=m,
    )
    log("done", f"saved channel_sparsity_stats.npz -> {results_dir}/")

    plot_channel_sparsity(frac_arr, std_arr, eff_arr, m, results_dir)

    return frac_90_means, eff_ch_means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    ctx = prepare_run(args, capture_activations=True)
    run_channel_sparsity(ctx["layers_info"], ctx["mlp_inputs"], args.results_dir)


if __name__ == "__main__":
    main()
