#!/usr/bin/env python3
"""experiment 15: tucker core visualization.

shows that the trained core C in R^{s x r x r} has substantial off-diagonal
mass, qualitatively confirming what the stable-rank histogram of exp12
shows quantitatively. for selected output channels alpha, plot the
(r x r) frontal slice C[alpha, :, :] as a heatmap. if the tucker layer
were equivalent to swiglu, all slices would be effectively zero off the
superdiagonal C[alpha, alpha, alpha].

we also report the off-diagonal mass fraction: ||C - diag(C)||_F^2 /
||C||_F^2, which equals 0 for an aligned-swiglu and 1 for a uniformly
random (no-diagonal-bias) core.

outputs (under --results_dir):
  core_visualization.png
  core_offdiagonal_mass.json
"""

import argparse
import glob
import json
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import LAYER_CMAP, PALETTE, log, setup_plot_style  # noqa: E402
from lib.lm import make_lm  # noqa: E402


def load_tucker_lm(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = sd["cfg"]
    model = make_lm("tucker", d=cfg["d"], n_heads=cfg["n_heads"],
                     n_layers=cfg["n_layers"], vocab_size=cfg["vocab_size"],
                     max_seq_len=cfg["seq_len"], r=cfg["tucker_r"],
                     s=cfg["tucker_s"])
    model.load_state_dict(sd["model_state_dict"])
    return model, cfg


def offdiag_mass_fraction(C):
    """fraction of frobenius mass that is NOT on the superdiagonal."""
    s, r1, r2 = C.shape
    kdim = min(s, r1, r2)
    diag_mass2 = 0.0
    for i in range(kdim):
        diag_mass2 += float(C[i, i, i] ** 2)
    total_mass2 = float((C.float() ** 2).sum())
    return 1.0 - diag_mass2 / max(total_mass2, 1e-30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_glob", type=str,
                        default="results/exp11/tucker_seed*/checkpoint_final.pt")
    parser.add_argument("--results_dir", type=str, default="results/exp15")
    parser.add_argument("--n_alpha", type=int, default=4,
                        help="number of frontal slices C[alpha, :, :] to plot")
    parser.add_argument("--n_layers_show", type=int, default=4,
                        help="layers to facet")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    setup_plot_style()

    paths = sorted(glob.glob(args.ckpt_glob))
    if not paths:
        log("error", f"no checkpoints at {args.ckpt_glob}")
        return

    summary = {}
    for p in paths:
        tag = pathlib.Path(p).parent.name
        log("info", f"=== visualizing {tag} ({p}) ===")
        model, cfg = load_tucker_lm(p)
        L = len(model.blocks)
        layer_picks = np.linspace(0, L - 1, args.n_layers_show, dtype=int)
        alpha_picks = list(range(args.n_alpha))

        offdiag_per_layer = []
        for li, blk in enumerate(model.blocks):
            C = blk.ffn.core().detach()
            offdiag_per_layer.append(offdiag_mass_fraction(C))
        log("eval", f"{tag} | mean offdiag mass fraction = "
            f"{np.mean(offdiag_per_layer):.4f} | "
            f"min = {min(offdiag_per_layer):.4f} | "
            f"max = {max(offdiag_per_layer):.4f}")
        summary[tag] = {
            "offdiag_mass_per_layer": offdiag_per_layer,
            "mean_offdiag": float(np.mean(offdiag_per_layer)),
        }

        # heatmaps: rows = layer_picks, cols = alpha_picks
        fig, axes = plt.subplots(len(layer_picks), len(alpha_picks),
                                  figsize=(2.4 * len(alpha_picks),
                                           2.4 * len(layer_picks)),
                                  sharex=True, sharey=True)
        if len(layer_picks) == 1:
            axes = axes.reshape(1, -1)
        if len(alpha_picks) == 1:
            axes = axes.reshape(-1, 1)
        for i, li in enumerate(layer_picks):
            C = model.blocks[li].ffn.core().detach().float().cpu().numpy()
            vmax = max(abs(C.max()), abs(C.min()))
            for j, al in enumerate(alpha_picks):
                im = axes[i, j].imshow(C[al], cmap="RdBu_r", vmin=-vmax,
                                        vmax=vmax, aspect="equal")
                if i == 0:
                    axes[i, j].set_title(fr"$\alpha={al}$")
                if j == 0:
                    axes[i, j].set_ylabel(f"layer {li}\n$i$")
                if i == len(layer_picks) - 1:
                    axes[i, j].set_xlabel(r"$j$")
        plt.tight_layout()
        out = os.path.join(args.results_dir, f"core_visualization_{tag}.png")
        plt.savefig(out)
        plt.close()
        log("done", f"saved core_visualization_{tag}.png -> {args.results_dir}/")

    with open(os.path.join(args.results_dir,
                            "core_offdiagonal_mass.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("done", f"saved core_offdiagonal_mass.json -> {args.results_dir}/")


if __name__ == "__main__":
    main()
