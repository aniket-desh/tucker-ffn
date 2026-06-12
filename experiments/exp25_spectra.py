#!/usr/bin/env python3
"""experiment 25 (sprint 2, confound B): is "per-route rank ≈ 4" real or a
stable-rank artifact?

For every trained checkpoint with per-route matrices (Tucker V_j = R C^(j),
LL1 V_b = U_b A_b^T), compute the FULL singular spectrum of every per-route
matrix and report, per layer:

  stable rank        ||V||_F^2 / ||V||_op^2   (sprint-1 metric)
  numerical rank     #{sigma_i > tau * sigma_1} at tau in {0.1, 0.01}
  spectral entropy   exp(H(sigma^2 / sum sigma^2))  ("effective rank", Roy &
                     Vetterli)
  top-k energy       fraction of ||V||_F^2 in top-{1,4,16} singular values

Init dependence: also run on a freshly initialized (untrained) Tucker model
with and without the diagonal warm start, and on the no-warm-start trained
probe if present (results/s2_lm/tucker_noWS).

outputs: results/exp25/exp25_spectra.json + exp25_spectra.png
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

from lib import log, setup_plot_style  # noqa: E402
from lib.ll1_ffn import LL1FFN  # noqa: E402
from lib.tucker_ffn import TuckerFFN  # noqa: E402
from exp19_interp_proxies import load_ckpt  # noqa: E402


def route_matrices(ffn):
    if isinstance(ffn, TuckerFFN):
        return torch.stack([ffn.R @ ffn.C[:, :, j] for j in range(ffn.r)])
    if isinstance(ffn, LL1FFN):
        return ffn.per_gate_matrices()
    return None


@torch.no_grad()
def spectrum_stats(V):
    """V: (n_routes, d, r). returns per-route spectra stats + raw mean
    spectrum (normalized by sigma_1)."""
    S = torch.linalg.svdvals(V)                      # (n_routes, min(d,r))
    s1 = S[:, :1] + 1e-12
    e = S ** 2
    tot = e.sum(1, keepdim=True) + 1e-12
    p = e / tot
    H = -(p * (p + 1e-12).log()).sum(1)
    return {
        "stable_rank": (e.sum(1) / (S[:, 0] ** 2 + 1e-12)).mean().item(),
        "numrank_0.1": (S > 0.1 * s1).sum(1).float().mean().item(),
        "numrank_0.01": (S > 0.01 * s1).sum(1).float().mean().item(),
        "spectral_entropy_rank": H.exp().mean().item(),
        "top1_energy": (e[:, :1].sum(1) / tot.squeeze(1)).mean().item(),
        "top4_energy": (e[:, :4].sum(1) / tot.squeeze(1)).mean().item(),
        "top16_energy": (e[:, :16].sum(1) / tot.squeeze(1)).mean().item(),
        "mean_spectrum": (S / s1).mean(0)[:32].tolist(),
    }


def analyze_model(model, tag):
    rows = []
    for li, blk in enumerate(model.blocks):
        V = route_matrices(blk.ffn)
        if V is None:
            continue
        st = spectrum_stats(V.float())
        st.update({"tag": tag, "layer": li})
        rows.append(st)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", type=str, default=None,
                    help="comma-separated checkpoint paths; default: sprint-1 "
                         "tucker+ll1 seeds + s2 tucker probes if present")
    ap.add_argument("--results_dir", type=str, default="results/exp25")
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    if args.ckpts:
        paths = args.ckpts.split(",")
    else:
        paths = sorted(glob.glob("results/sprint_lm/tucker_seed*/checkpoint_final.pt")) \
            + sorted(glob.glob("results/sprint_lm/ll1_l*_seed0/checkpoint_final.pt")) \
            + sorted(glob.glob("results/sprint_lm/ll1_l4_seed*/checkpoint_final.pt")) \
            + sorted(glob.glob("results/s2_lm/tucker_*/tucker_seed0/checkpoint_final.pt"))
        paths = list(dict.fromkeys(paths))

    all_rows = []
    for p in paths:
        try:
            model, cfg = load_ckpt(p, "cpu")
        except Exception as e:
            log("error", f"{p}: {e}")
            continue
        tag = "/".join(p.split("/")[1:-1])
        all_rows += analyze_model(model, tag)
        log("info", f"{tag}: done")
        del model

    # untrained baselines: tucker random init and diag warm start
    for init_tag, kw in [("INIT_tucker_diagWS", dict(diagonal_bias_init=True)),
                         ("INIT_tucker_random", dict())]:
        torch.manual_seed(0)
        ffn = TuckerFFN(512, r=128, s=128, **kw)
        V = route_matrices(ffn)
        st = spectrum_stats(V.float())
        st.update({"tag": init_tag, "layer": -1})
        all_rows.append(st)
        log("info", f"{init_tag}: stable_rank={st['stable_rank']:.2f} "
            f"specH={st['spectral_entropy_rank']:.2f}")

    with open(os.path.join(args.results_dir, "exp25_spectra.json"), "w") as f:
        json.dump(all_rows, f, indent=2)

    # summary print + figure
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    tags = sorted({r["tag"] for r in all_rows})
    cmap = plt.get_cmap("tab10")
    print(f"{'tag':42s} {'stable':>7s} {'nr0.1':>6s} {'nr0.01':>7s} "
          f"{'specH':>6s} {'top4E':>6s}")
    for ti, tag in enumerate(tags):
        rows = [r for r in all_rows if r["tag"] == tag]
        m = lambda k: np.mean([r[k] for r in rows])
        print(f"{tag:42s} {m('stable_rank'):7.2f} {m('numrank_0.1'):6.1f} "
              f"{m('numrank_0.01'):7.1f} {m('spectral_entropy_rank'):6.1f} "
              f"{m('top4_energy'):6.2f}")
        spec = np.mean([r["mean_spectrum"] for r in rows], axis=0)
        ax = axes[0] if "tucker" in tag.lower() else axes[1]
        ax.plot(range(1, len(spec) + 1), spec, label=tag[:28],
                color=cmap(ti % 10), lw=1.2)
    for ax, t in zip(axes, ["Tucker variants", "LL1 variants"]):
        ax.set_yscale("log")
        ax.set_xlabel("singular value index")
        ax.set_ylabel(r"$\sigma_i / \sigma_1$ (mean over routes/layers)")
        ax.set_title(t, fontsize=10)
        ax.legend(fontsize=5)
    plt.tight_layout()
    out = os.path.join(args.results_dir, "exp25_spectra.png")
    plt.savefig(out, dpi=180)
    log("done", f"saved {out}")


if __name__ == "__main__":
    main()
