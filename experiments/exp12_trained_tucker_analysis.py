#!/usr/bin/env python3
"""experiment 12: stable rank of V_j on trained tucker ffns.

under the matched-coordinates statement of theorem 1, the relevant
per-gate matrix is V_j = R C^(j) in R^{d x r}, where C^(j) = C[:, :, j]
selects the gate-j frontal slice of the tucker core. an aligned-swiglu
trying to match a tucker layer needs at least sum_j rank(V_j) hidden
units. so the natural empirical question after training tucker from
scratch (exp11) is: do learned V_j actually have rank > 1, or does the
optimizer collapse to rank-1 per gate (which would make tucker no more
expressive than swiglu)?

we report two rank proxies on each layer's V_j:
  stable rank        = ||V_j||_F^2 / ||V_j||_2^2   (frobenius / spectral)
  effective rank     = exp(H(sigma_i^2 / sum sigma^2))  (entropy of squared
                      singular values)
both are continuous, well-defined for floats, and bounded above by min(d, r).
the prediction is that both should be > 1 (often >> 1) if the trained
model exploits cross-channel structure.

input: a directory of trained checkpoints from exp11 (each tucker run).

outputs (under --results_dir):
  stable_rank.npz             — arrays per checkpoint, {layer x j}
  stable_rank_histogram.png   — histogram of stable rank, per-layer facets
  stable_rank_heatmap.png     — heatmap layer x j of stable rank
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
from lib.lm import LM, LMConfig, FFNConfig, make_lm  # noqa: E402


def stable_rank(M):
    """||M||_F^2 / ||M||_op^2; continuous proxy for rank, bounded by min(M.shape)."""
    s = torch.linalg.svdvals(M.float())
    op2 = (s[0] ** 2).clamp(min=1e-30)
    fro2 = (s ** 2).sum()
    return float(fro2 / op2)


def effective_rank(M):
    """exp(entropy of normalized squared singular values)."""
    s = torch.linalg.svdvals(M.float())
    p = (s ** 2)
    p = p / p.sum().clamp(min=1e-30)
    p = p.clamp(min=1e-30)
    H = -(p * p.log()).sum()
    return float(torch.exp(H))


def load_tucker_lm(ckpt_path):
    """rebuild LM from saved cfg and load weights."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = sd["cfg"]
    if cfg["arch"] == "tucker":
        model = make_lm("tucker", d=cfg["d"], n_heads=cfg["n_heads"],
                         n_layers=cfg["n_layers"], vocab_size=cfg["vocab_size"],
                         max_seq_len=cfg["seq_len"], r=cfg["tucker_r"],
                         s=cfg["tucker_s"])
    elif cfg["arch"] == "tucker_diag":
        model = make_lm("tucker", d=cfg["d"], n_heads=cfg["n_heads"],
                         n_layers=cfg["n_layers"], vocab_size=cfg["vocab_size"],
                         max_seq_len=cfg["seq_len"], r=cfg["tucker_r"],
                         s=cfg["tucker_s"], diagonal_only=True)
    else:
        raise ValueError(f"not a tucker checkpoint: arch={cfg['arch']}")
    model.load_state_dict(sd["model_state_dict"])
    return model, cfg


def analyze_one(ckpt_path):
    """compute V_j stable & effective ranks per (layer, j) from one checkpoint.

    vectorized: stack the per-gate V_j tensors into a (r, d, r) batch and call
    torch.linalg.svdvals once per layer instead of r times.
    """
    model, cfg = load_tucker_lm(ckpt_path)
    L = len(model.blocks)
    r = cfg["tucker_r"]
    s_rank = np.zeros((L, r))
    e_rank = np.zeros((L, r))
    op_norm = np.zeros((L, r))
    for li, blk in enumerate(model.blocks):
        ffn = blk.ffn  # TuckerFFN
        R = ffn.R.detach().float()           # (d, s)
        C = ffn.core().detach().float()      # (s, r, r)
        # V[j] = R @ C[:, :, j] -> stack via einsum to (r2, d, r)
        V = torch.einsum("ds,srj->jdr", R, C)
        sv = torch.linalg.svdvals(V)         # (r2, k)  k = min(d, r)
        sv2 = sv ** 2
        # stable rank = ||V||_F^2 / ||V||_op^2 = sum sv^2 / max sv^2
        op2 = sv2[:, 0].clamp(min=1e-30)
        s_rank[li] = (sv2.sum(dim=-1) / op2).cpu().numpy()
        # effective rank = exp(entropy of normalized sv^2)
        p = sv2 / sv2.sum(dim=-1, keepdim=True).clamp(min=1e-30)
        p = p.clamp(min=1e-30)
        H = -(p * p.log()).sum(dim=-1)
        e_rank[li] = torch.exp(H).cpu().numpy()
        op_norm[li] = sv[:, 0].cpu().numpy()
    return s_rank, e_rank, op_norm, cfg


def plot_stable_rank(results, results_dir):
    """histogram of stable rank, faceted by layer; heatmap layer x j."""
    setup_plot_style()
    for tag, (s_rank, e_rank, op_norm, cfg) in results.items():
        L = s_rank.shape[0]
        r = s_rank.shape[1]

        # histogram, one panel per few layers
        n_panels = min(4, L)
        layer_picks = np.linspace(0, L - 1, n_panels, dtype=int)
        fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3),
                                  sharey=True)
        if n_panels == 1:
            axes = [axes]
        cmap = plt.get_cmap(LAYER_CMAP)
        for ax, li in zip(axes, layer_picks):
            color = cmap(li / max(L - 1, 1))
            ax.hist(s_rank[li], bins=20, color=color, alpha=0.75,
                    edgecolor="0.3", linewidth=0.4)
            ax.axvline(s_rank[li].mean(), color=PALETTE["neutral"], ls="--",
                        lw=0.8, label=f"mean = {s_rank[li].mean():.2f}")
            ax.set_xlabel("stable rank of $V_j$")
            ax.set_title(f"layer {li}")
            ax.legend(framealpha=0.9, edgecolor="0.8", fontsize=8)
        axes[0].set_ylabel("count (gates)")
        plt.tight_layout()
        out = os.path.join(results_dir, f"stable_rank_histogram_{tag}.png")
        plt.savefig(out)
        plt.close()
        log("done", f"saved stable_rank_histogram_{tag}.png -> {results_dir}/")

        # heatmap layer x j (sort gates by mean stable rank to make pattern clear)
        sort_idx = s_rank.mean(axis=0).argsort()[::-1]
        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(s_rank[:, sort_idx], aspect="auto", cmap="viridis",
                        interpolation="nearest", vmin=1.0)
        ax.set_xlabel("gate index $j$ (sorted by mean stable rank)")
        ax.set_ylabel("layer")
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("stable rank of $V_j$")
        plt.tight_layout()
        out = os.path.join(results_dir, f"stable_rank_heatmap_{tag}.png")
        plt.savefig(out)
        plt.close()
        log("done", f"saved stable_rank_heatmap_{tag}.png -> {results_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_glob", type=str,
                        default="results/exp11/tucker_seed*/checkpoint_final.pt",
                        help="glob pattern for tucker checkpoints to analyze")
    parser.add_argument("--results_dir", type=str, default="results/exp12")
    parser.add_argument("--plot_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.plot_only:
        path = os.path.join(args.results_dir, "stable_rank.npz")
        data = np.load(path, allow_pickle=True)
        results = {}
        for key in data.files:
            if key.endswith("__s_rank"):
                tag = key[:-len("__s_rank")]
                results[tag] = (data[f"{tag}__s_rank"],
                                 data[f"{tag}__e_rank"],
                                 data[f"{tag}__op_norm"],
                                 None)
        plot_stable_rank(results, args.results_dir)
        return

    paths = sorted(glob.glob(args.ckpt_glob))
    if not paths:
        log("error", f"no checkpoints found at {args.ckpt_glob}")
        return
    log("info", f"found {len(paths)} checkpoints")

    results = {}
    save_dict = {}
    for p in paths:
        tag = pathlib.Path(p).parent.name
        log("info", f"analyzing {tag} ({p})")
        s_rank, e_rank, op_norm, cfg = analyze_one(p)
        results[tag] = (s_rank, e_rank, op_norm, cfg)
        log("eval", f"{tag} | mean stable_rank = {s_rank.mean():.2f} | "
            f"mean effective_rank = {e_rank.mean():.2f} | "
            f"min = {s_rank.min():.2f} | max = {s_rank.max():.2f}")
        save_dict[f"{tag}__s_rank"] = s_rank
        save_dict[f"{tag}__e_rank"] = e_rank
        save_dict[f"{tag}__op_norm"] = op_norm

    np.savez(os.path.join(args.results_dir, "stable_rank.npz"), **save_dict)
    log("done", f"saved stable_rank.npz -> {args.results_dir}/")
    plot_stable_rank(results, args.results_dir)


if __name__ == "__main__":
    main()
