#!/usr/bin/env python3
"""experiment 23 (sprint 2, Exp B): structured factor matrices on real FFN maps.

Question: do Monarch / butterfly / block-diagonal / low-rank factor matrices
buy approximation quality per parameter when fitting real pretrained FFN
input-output maps? At matched parameters, structured projections afford a much
WIDER ffn (e.g. monarch nb=4 at d=896: ~4x the hidden width of dense swiglu);
the trade under test is width-via-structure vs dense expressivity.

Students at each budget (params == FLOPs for all kinds):
  swiglu               dense baseline
  swiglu_lowrank       rank chosen to fit budget at m = dense m x2
  swiglu_blockdiag4    block-diagonal, nb=4 (no cross-block mixing — control)
  swiglu_monarch4      monarch, nb=4 (cross-block mixing via permutation)
  swiglu_butterfly4    butterfly stages + blockdiag resize (global mixing)
  ll1_l4               dense LL1 (sprint-1 best)
  ll1_l4_monarch4      LL1 with monarch A/U
  ll1_l4_blockdiag4    LL1 with block-diagonal A/U

outputs: results/exp23/exp23_results.json + exp23_structured.png
"""

import argparse
import json
import math
import os
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import log, setup_plot_style  # noqa: E402
from lib.tucker_ffn import SwiGLUFFN  # noqa: E402
from lib.ll1_ffn import LL1FFN, ll1_blocks_for_params  # noqa: E402
from lib.structured_ffn import (  # noqa: E402
    StructuredLL1,
    StructuredSwiGLU,
    ll1_struct_blocks_for_params,
    swiglu_struct_width_for_params,
)
from exp18_ll1_synthetic import fit_student  # noqa: E402
from exp21_qwen_distill import capture_layer_io  # noqa: E402


def build_student(arch, d, budget, seed):
    torch.manual_seed(seed)
    if arch == "swiglu":
        m = max(1, round(budget / (3 * d)))
        s = SwiGLUFFN(d, m)
        meta = {"m": m}
    elif arch == "ll1_l4":
        B = ll1_blocks_for_params(d, 4, budget)
        s = LL1FFN(d, n_blocks=B, block_rank=4)
        meta = {"B": B, "L": 4}
    elif arch.startswith("swiglu_"):
        kind = arch.split("_", 1)[1].rstrip("0123456789")
        nb = int(arch[len("swiglu_") + len(kind):] or 4)
        kw = {"n_blocks": nb} if kind != "lowrank" else {"rank": None}
        if kind == "lowrank":
            # width = 2x dense width at this budget; rank to fit
            m = 2 * max(1, round(budget / (3 * d)))
            rank = max(1, budget // (3 * (d + m)))
            s = StructuredSwiGLU(d, m, "lowrank", rank=rank)
            meta = {"m": m, "rank": rank}
        else:
            m = swiglu_struct_width_for_params(d, budget, kind, n_blocks=nb)
            s = StructuredSwiGLU(d, m, kind, n_blocks=nb)
            meta = {"m": m, "nb": nb}
    elif arch.startswith("ll1_l4_"):
        kind = arch[len("ll1_l4_"):].rstrip("0123456789")
        nb = int(arch[len("ll1_l4_") + len(kind):] or 4)
        B = ll1_struct_blocks_for_params(d, 4, budget, kind, n_blocks=nb)
        s = StructuredLL1(d, routes=B, block_rank=4, kind=kind, n_blocks=nb)
        meta = {"B": B, "L": 4, "nb": nb}
    else:
        raise ValueError(arch)
    return s, s.num_params() if hasattr(s, "num_params") else \
        sum(p.numel() for p in s.parameters()), meta


ARCHS = ["swiglu", "swiglu_lowrank", "swiglu_blockdiag4", "swiglu_monarch4",
         "swiglu_butterfly4", "ll1_l4", "ll1_l4_monarch4", "ll1_l4_blockdiag4"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", type=str, default="4,12")
    ap.add_argument("--n_tokens", type=int, default=120_000)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--budgets", type=str, default="600000,1200000")
    ap.add_argument("--archs", type=str, default=",".join(ARCHS))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n_seeds", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--results_dir", type=str, default="results/exp23")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    res_path = os.path.join(args.results_dir, "exp23_results.json")
    layers = [int(s) for s in args.layers.split(",")]

    if not args.plot_only:
        io = capture_layer_io(args.model, layers, args.n_tokens, args.seq_len,
                              args.device)
        results = []
        for li in layers:
            X, Y = io[li]
            d = X.shape[1]
            n_val = int(len(X) * args.val_frac)
            perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
            X, Y = X[perm].float(), Y[perm].float()
            Y = Y / Y[n_val:].std()
            x_tr, y_tr = X[n_val:].to(args.device), Y[n_val:].to(args.device)
            x_va, y_va = X[:n_val].to(args.device), Y[:n_val].to(args.device)
            var_y = y_va.var().item()
            log("info", f"layer {li}: d={d} train={len(x_tr)} val={len(x_va)}")
            for budget in [int(b) for b in args.budgets.split(",")]:
                for arch in args.archs.split(","):
                    mses = []
                    meta = None
                    for seed in range(args.n_seeds):
                        student, n_params, meta = build_student(
                            arch, d, budget, seed=seed * 31 + 5)
                        student = student.to(args.device)
                        t0 = time.time()
                        mse = fit_student(student, x_tr, y_tr, x_va, y_va,
                                          args.steps, args.lr, args.batch,
                                          args.device)
                        mses.append(mse / var_y)
                        log("train", f"L{li} <- {arch:20s} budget={budget} "
                            f"params={n_params} {meta} seed={seed} "
                            f"relMSE={mse/var_y:.4f} ({time.time()-t0:.0f}s)")
                        del student
                        torch.cuda.empty_cache()
                    results.append({"layer": li, "arch": arch, "budget": budget,
                                    "params": n_params, "meta": meta,
                                    "rel_mse_seeds": mses})
                    with open(res_path, "w") as f:
                        json.dump(results, f, indent=2)
            del x_tr, y_tr, x_va, y_va
            torch.cuda.empty_cache()
        log("done", f"exp23 complete -> {res_path}")

    with open(res_path) as f:
        results = json.load(f)

    setup_plot_style()
    budgets = sorted({r["budget"] for r in results})
    fig, axes = plt.subplots(1, len(layers), figsize=(5.0 * len(layers), 3.8),
                             sharey=True)
    if len(layers) == 1:
        axes = [axes]
    colors = {"swiglu": "#2c7fb8", "swiglu_lowrank": "#9ecae1",
              "swiglu_blockdiag4": "#fd8d3c", "swiglu_monarch4": "#6a51a3",
              "swiglu_butterfly4": "#c994c7", "ll1_l4": "#31a354",
              "ll1_l4_monarch4": "#74c476", "ll1_l4_blockdiag4": "#a1d99b"}
    for ax, li in zip(axes, layers):
        for arch in [a for a in ARCHS if any(r["arch"] == a for r in results)]:
            xs, ms = [], []
            for b in budgets:
                rows = [r for r in results
                        if r["layer"] == li and r["arch"] == arch and r["budget"] == b]
                if rows:
                    xs.append(rows[0]["params"])
                    ms.append(np.mean(rows[0]["rel_mse_seeds"]))
            ax.plot(xs, ms, "o-", color=colors.get(arch, "k"), label=arch, ms=4)
        ax.set_xscale("log")
        ax.set_xlabel("student parameters")
        ax.set_title(f"Qwen2.5-0.5B layer {li}", fontsize=10)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("val MSE / Var(y)")
    plt.tight_layout()
    out = os.path.join(args.results_dir, "exp23_structured.png")
    plt.savefig(out, dpi=180)
    plt.close()
    log("done", f"saved {out}")


if __name__ == "__main__":
    main()
