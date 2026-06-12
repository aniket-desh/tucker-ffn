#!/usr/bin/env python3
"""Summarize sprint LM runs: final val losses per arch/seed, L-sweep figure,
loss-curve figure, and a markdown/LaTeX table. Reads results/sprint_lm/*/loss_log.json."""
import argparse
import glob
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from lib import PALETTE, setup_plot_style  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results_dir", default="results/sprint_lm")
ap.add_argument("--out_dir", default="docs/structured_tensor_ffn_sprint")
args = ap.parse_args()

runs = {}
for path in sorted(glob.glob(os.path.join(args.results_dir, "*", "loss_log.json"))):
    tag = os.path.basename(os.path.dirname(path))
    arch, seed = tag.rsplit("_seed", 1)
    with open(path) as f:
        loss_log = json.load(f)
    if not loss_log:
        continue
    runs.setdefault(arch, {})[int(seed)] = loss_log

print(f"{'arch':12s} {'seeds':14s} {'final val loss':>22s} {'ppl':>8s}")
table = {}
for arch, seeds in sorted(runs.items()):
    finals = {s: ll[-1]["val_loss"] for s, ll in seeds.items()
              if ll[-1]["tokens"] >= 99_000_000}
    finals_done = {s: v for s, v in finals.items()}
    vals = np.array(list(finals_done.values()))
    if len(vals) == 0:
        continue
    table[arch] = (vals.mean(), vals.std(), len(vals), sorted(finals_done))
    print(f"{arch:12s} {str(sorted(finals_done)):14s} "
          f"{vals.mean():>10.4f} ± {vals.std():.4f} ({len(vals)})"
          f" {math.exp(vals.mean()):>8.2f}")

setup_plot_style()

# ── L-sweep figure ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.2, 3.8))
Ls, means, stds = [], [], []
for arch, (m, s, n, _) in table.items():
    if arch.startswith("ll1_l"):
        Ls.append(int(arch[5:])); means.append(m); stds.append(s)
order = np.argsort(Ls)
Ls = np.array(Ls)[order]; means = np.array(means)[order]; stds = np.array(stds)[order]
if len(Ls):
    ax.errorbar(Ls, means, yerr=stds, fmt="o-", color="#31a354",
                label="LL1 (L sweep)", capsize=3)
for arch, color, label in [("swiglu", "#2c7fb8", "SwiGLU"),
                           ("tucker", "#d7301f", "dense Tucker")]:
    if arch in table:
        m, s, n, _ = table[arch]
        ax.axhline(m, color=color, ls="--", label=f"{label} ({n} seeds)")
        ax.axhspan(m - s, m + s, color=color, alpha=0.12)
ax.set_xscale("log", base=2)
ax.set_xticks(Ls); ax.set_xticklabels([str(int(x)) for x in Ls])
ax.set_xlabel("LL1 block rank L (params matched)")
ax.set_ylabel("final validation loss (nats)")
ax.legend(fontsize=8)
plt.tight_layout()
os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
out = os.path.join(args.out_dir, "figures", "lm_lsweep.png")
plt.savefig(out, dpi=180)
plt.savefig(out.replace(".png", ".pdf"))
plt.close()
print("saved", out)

# ── curves figure ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.2, 4.0))
style = {"swiglu": ("#2c7fb8", "SwiGLU"),
         "ll1_l4": ("#31a354", "LL1 (L=4)"),
         "tucker": ("#d7301f", "dense Tucker")}
for arch, (color, label) in style.items():
    if arch not in runs:
        continue
    curves, tok = [], None
    nmin = min(len(ll) for ll in runs[arch].values())
    for s, ll in sorted(runs[arch].items()):
        tok = np.array([e["tokens"] for e in ll[:nmin]]) / 1e6
        curves.append([e["val_loss"] for e in ll[:nmin]])
    curves = np.array(curves)
    m, sd = curves.mean(0), curves.std(0)
    ax.plot(tok, m, color=color, label=f"{label} (n={curves.shape[0]})")
    ax.fill_between(tok, m - sd, m + sd, color=color, alpha=0.18)
ax.set_xlabel("training tokens (M)")
ax.set_ylabel("validation loss (nats)")
ax.set_ylim(top=6.5)
ax.legend(fontsize=8)
plt.tight_layout()
out = os.path.join(args.out_dir, "figures", "lm_curves.png")
plt.savefig(out, dpi=180)
plt.savefig(out.replace(".png", ".pdf"))
plt.close()
print("saved", out)

# ── tables ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(args.out_dir, "tables"), exist_ok=True)
with open(os.path.join(args.out_dir, "tables", "lm_final.md"), "w") as f:
    f.write("| arch | n seeds | final val loss | ppl |\n|---|---|---|---|\n")
    for arch, (m, s, n, seeds) in sorted(table.items()):
        f.write(f"| {arch} | {n} | {m:.4f} ± {s:.4f} | {math.exp(m):.2f} |\n")
print("saved tables/lm_final.md")
