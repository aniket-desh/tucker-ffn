#!/usr/bin/env python3
"""Figures from exp19 interp proxies: top-k decomposability, effective active
units by layer, ablation locality, stable rank. Averages across seeds per arch."""
import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from lib import PALETTE, setup_plot_style  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/exp19/exp19_results.json")
ap.add_argument("--out_dir", default="docs/structured_tensor_ffn_sprint/figures")
args = ap.parse_args()

with open(args.results) as f:
    rows = json.load(f)

os.makedirs(args.out_dir, exist_ok=True)
setup_plot_style()

ARCH_STYLE = {
    "swiglu": (PALETTE["ablation"], "SwiGLU (1493 atoms)"),
    "ll1_l4": (PALETTE["primary"], "LL1 L=4 (498 blocks)"),
    "tucker": (PALETTE["accent"], "Tucker (128 gates)"),
}
def arch_key(a):
    return a if a in ARCH_STYLE else None

by_arch = defaultdict(list)
for r in rows:
    k = arch_key(r["arch"])
    if k:
        by_arch[k].append(r)

# ── top-k decomposability ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
for k, rs in by_arch.items():
    color, label = ARCH_STYLE[k]
    ks = sorted({int(x) for r in rs for x in r["topk_loss"]})
    curves = []
    for r in rs:
        curves.append([r["topk_loss"][str(x)] if str(x) in r["topk_loss"]
                       else r["topk_loss"].get(x, np.nan) for x in ks])
    curves = np.array(curves, dtype=float)
    base = np.mean([r["base_loss"] for r in rs])
    m = np.nanmean(curves, 0)
    sd = np.nanstd(curves, 0)
    n_units = rs[0]["layers"][0]["n_units"]
    for ax, xs in ((axes[0], np.array(ks)),
                   (axes[1], np.array(ks) / n_units)):
        ax.plot(xs, m - base, "o-", color=color, label=label, ms=3.5)
        ax.fill_between(xs, m - sd - base, m + sd - base, color=color, alpha=0.15)
axes[0].set_xlabel("routed units kept per token (k)")
axes[1].set_xlabel("fraction of units kept per token")
for ax in axes:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("excess val loss (nats)")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "interp_topk.png"), dpi=180)
plt.savefig(os.path.join(args.out_dir, "interp_topk.pdf"))
plt.close()

# ── effective active fraction by layer ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
for k, rs in by_arch.items():
    color, label = ARCH_STYLE[k]
    eff_frac = np.array([[lm["eff_active_frac"] for lm in r["layers"]] for r in rs])
    eff_abs = np.array([[lm["eff_active"] for lm in r["layers"]] for r in rs])
    L = eff_frac.shape[1]
    axes[0].plot(range(L), eff_frac.mean(0), "o-", color=color, label=label, ms=3.5)
    axes[0].fill_between(range(L), eff_frac.mean(0) - eff_frac.std(0),
                         eff_frac.mean(0) + eff_frac.std(0), color=color, alpha=0.15)
    axes[1].plot(range(L), eff_abs.mean(0), "o-", color=color, label=label, ms=3.5)
axes[0].set_ylabel("effective active fraction"); axes[1].set_ylabel("effective active units")
axes[1].set_yscale("log")
for ax in axes:
    ax.set_xlabel("layer"); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "interp_effactive.png"), dpi=180)
plt.savefig(os.path.join(args.out_dir, "interp_effactive.pdf"))
plt.close()

# ── ablation locality ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.4, 3.6))
for k, rs in by_arch.items():
    color, label = ARCH_STYLE[k]
    deltas = []
    for r in rs:
        for li, ab in r["ablation"].items():
            deltas.extend(ab["delta_loss"])
    deltas = np.sort(np.array(deltas))[::-1]
    ax.plot(np.arange(1, len(deltas) + 1), np.maximum(deltas, 1e-6), "o-",
            color=color, label=label, ms=3)
ax.set_yscale("log")
ax.set_xlabel("unit (sorted by ablation effect)")
ax.set_ylabel("Δ val loss from single-unit ablation")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "interp_ablation.png"), dpi=180)
plt.savefig(os.path.join(args.out_dir, "interp_ablation.pdf"))
plt.close()

# ── stable rank + tucker core table ────────────────────────────────────────
lines = ["| arch | seed | stable rank mean (max) | superdiag energy | core eff-entry frac |",
         "|---|---|---|---|---|"]
for r in rows:
    lm0 = r["layers"][0]
    sr = [f"{lm['stable_rank_mean']:.2f}" if lm["stable_rank_mean"] else "—"
          for lm in r["layers"]]
    mean_sr = np.mean([lm["stable_rank_mean"] for lm in r["layers"]
                       if lm["stable_rank_mean"]]) if r["arch"] != "swiglu" else None
    max_sr = np.max([lm["stable_rank_max"] for lm in r["layers"]
                     if lm["stable_rank_max"]]) if r["arch"] != "swiglu" else None
    sd = np.mean([lm.get("superdiag_energy_frac", np.nan) for lm in r["layers"]])
    ce = np.mean([lm.get("core_eff_entry_frac", np.nan) for lm in r["layers"]])
    lines.append(f"| {r['arch']} | {r['seed']} | "
                 f"{f'{mean_sr:.2f} ({max_sr:.2f})' if mean_sr else '1 (def)'} | "
                 f"{'' if np.isnan(sd) else f'{sd:.3f}'} | "
                 f"{'' if np.isnan(ce) else f'{ce:.4f}'} |")
out_t = "docs/structured_tensor_ffn_sprint/tables/interp_summary.md"
os.makedirs(os.path.dirname(out_t), exist_ok=True)
with open(out_t, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print("saved figures + table")
