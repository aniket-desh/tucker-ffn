#!/usr/bin/env python3
"""experiment 18: synthetic teacher-student recovery across the CP-LL1-Tucker ladder.

Hypothesis H1 (theory_notes §8): the relevant control variable for a routed
tensor FFN is the per-gate interaction rank L. A student whose block rank
meets the teacher's true rank should fit it; a student below should not.

Teachers (d=64, output normalized to unit std):
  cp       — LL1(B=32, L=1): 32 rank-one routed atoms (SwiGLU-structured)
  ll1_l4   — LL1(B=16, L=4): 16 routes, each gating a rank-4 block
  tucker   — TuckerFFN(r=s=16), generic dense core (full-rank slices)

Students, all unconstrained (they learn their own dictionaries — this tests
trainable recovery, not the aligned theorem, which exp10 already verified):
  swiglu(m)     — CP, one gate per atom
  ll1_l{1,2,4,8,16}(B) — B matched to the parameter budget
  tucker(r=s)   — r chosen to meet the budget

Protocol: fixed student parameter budget N (default 9216 = LL1(16,4) teacher
size at d=64), L-sweep at 1x budget, plus a budget sweep {0.25, 0.5, 1, 2}x
for archs {swiglu, ll1_l4, tucker}. n_seeds fits per cell; report all seeds
(mean/min in the plots).

Metrics: relative val MSE = MSE / Var(y). Exact-recovery threshold: <1e-6.

outputs (under --results_dir):
  exp18_results.json   — every cell: arch, L, B, params, seeds, val rel-MSE
  exp18_lsweep.png     — rel-MSE vs student block rank, panel per teacher
  exp18_budget.png     — rel-MSE vs params, panel per teacher
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

from lib import COLOR_CYCLE, PALETTE, log, setup_plot_style  # noqa: E402
from lib.ll1_ffn import LL1FFN, ll1_blocks_for_params, ll1_params  # noqa: E402
from lib.tucker_ffn import SwiGLUFFN, TuckerFFN, swiglu_params, tucker_params  # noqa: E402


def make_teacher(kind, d, gen_seed, device):
    g = torch.Generator().manual_seed(gen_seed)
    if kind == "cp":
        t = LL1FFN(d, n_blocks=32, block_rank=1)
    elif kind == "ll1_l4":
        t = LL1FFN(d, n_blocks=16, block_rank=4)
    elif kind == "tucker":
        t = TuckerFFN(d, r=16, s=16, diagonal_only=False)
        with torch.no_grad():
            # generic full-rank slices (resample like exp10)
            for _ in range(20):
                C = torch.randn(16, 16, 16, generator=g)
                if all(torch.linalg.matrix_rank(C[:, :, j]) == 16
                       for j in range(16)):
                    break
            t.C.copy_(C / 16.0)
            t.P.copy_(torch.randn(d, 16, generator=g) / d ** 0.5)
            t.Q.copy_(torch.randn(d, 16, generator=g) / d ** 0.5)
            t.R.copy_(torch.randn(d, 16, generator=g) / 16 ** 0.5)
    else:
        raise ValueError(kind)
    if kind != "tucker":
        with torch.no_grad():
            for p in t.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * (1.0 / math.sqrt(d)))
    return t.to(device).eval()


def tucker_r_for_params(d, target):
    """largest r=s with 3dr + r^3 <= target."""
    r = 1
    while tucker_params(d, r + 1, r + 1) <= target:
        r += 1
    return r


def build_student(arch, d, budget, seed):
    torch.manual_seed(seed)
    if arch == "swiglu":
        m = max(1, int(round(budget / (3 * d))))
        s = SwiGLUFFN(d, m)
        return s, swiglu_params(d, m), {"m": m}
    if arch.startswith("ll1_l"):
        L = int(arch[len("ll1_l"):])
        B = ll1_blocks_for_params(d, L, budget)
        s = LL1FFN(d, n_blocks=B, block_rank=L)
        return s, ll1_params(d, B, L), {"B": B, "L": L}
    if arch == "tucker":
        r = tucker_r_for_params(d, budget)
        s = TuckerFFN(d, r=r, s=r, diagonal_only=False)
        return s, tucker_params(d, r, r), {"r": r}
    raise ValueError(arch)


def fit_student(student, x_tr, y_tr, x_va, y_va, steps, lr, batch, device,
                log_tag=""):
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.01)
    n = x_tr.shape[0]
    student.train()
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        pred = student(x_tr[idx])
        loss = F.mse_loss(pred, y_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    student.eval()
    with torch.no_grad():
        val_mse = F.mse_loss(student(x_va), y_va).item()
    return val_mse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--budget", type=int, default=9216)
    ap.add_argument("--budget_scales", type=str, default="0.25,0.5,1,2")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--n_val", type=int, default=8000)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--teachers", type=str, default="cp,ll1_l4,tucker")
    ap.add_argument("--lsweep_archs", type=str,
                    default="swiglu,ll1_l1,ll1_l2,ll1_l4,ll1_l8,ll1_l16,tucker")
    ap.add_argument("--budget_archs", type=str, default="swiglu,ll1_l4,tucker")
    ap.add_argument("--results_dir", type=str, default="results/exp18")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    res_path = os.path.join(args.results_dir, "exp18_results.json")

    if not args.plot_only:
        results = []
        d, device = args.d, args.device
        for teacher_kind in args.teachers.split(","):
            teacher = make_teacher(teacher_kind, d, gen_seed=1234, device=device)
            gen = torch.Generator(device="cpu").manual_seed(777)
            x_tr = torch.randn(args.n_train, d, generator=gen).to(device)
            x_va = torch.randn(args.n_val, d, generator=gen).to(device)
            with torch.no_grad():
                y_tr = teacher(x_tr)
                y_va = teacher(x_va)
                scale = y_tr.std()
                y_tr = y_tr / scale
                y_va = y_va / scale
            var_y = y_va.var().item()
            log("info", f"teacher={teacher_kind} | y var={var_y:.3f} | "
                f"teacher params={sum(p.numel() for p in teacher.parameters())}")

            cells = []
            for arch in args.lsweep_archs.split(","):
                cells.append((arch, args.budget, "lsweep"))
            for sc in [float(s) for s in args.budget_scales.split(",")]:
                if sc == 1.0:
                    continue
                for arch in args.budget_archs.split(","):
                    cells.append((arch, int(round(args.budget * sc)), "budget"))

            for arch, budget, tag in cells:
                seed_mses = []
                meta = None
                for seed in range(args.n_seeds):
                    student, n_params, meta = build_student(arch, d, budget,
                                                            seed=seed * 101 + 7)
                    student = student.to(device)
                    t0 = time.time()
                    mse = fit_student(student, x_tr, y_tr, x_va, y_va,
                                      args.steps, args.lr, args.batch, device)
                    seed_mses.append(mse / var_y)
                    log("train", f"{teacher_kind} <- {arch} ({tag}) "
                        f"budget={budget} params={n_params} seed={seed} "
                        f"relMSE={mse/var_y:.3e} ({time.time()-t0:.0f}s)")
                results.append({
                    "teacher": teacher_kind, "arch": arch, "tag": tag,
                    "budget": budget, "params": n_params, "meta": meta,
                    "rel_mse_seeds": seed_mses,
                })
                with open(res_path, "w") as f:
                    json.dump(results, f, indent=2)
        log("done", f"all fits complete -> {res_path}")

    with open(res_path) as f:
        results = json.load(f)

    setup_plot_style()
    teachers = args.teachers.split(",")
    tnames = {"cp": "CP teacher (32 rank-1 atoms)",
              "ll1_l4": "LL1 teacher (16 blocks, L*=4)",
              "tucker": "dense Tucker teacher (r=s=16)"}

    # ── L-sweep figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(teachers), figsize=(4.2 * len(teachers), 3.6),
                             sharey=True)
    if len(teachers) == 1:
        axes = [axes]
    C_CP, C_LL1, C_TK = "#2c7fb8", "#31a354", "#d7301f"
    for ax, t in zip(axes, teachers):
        rows = [r for r in results if r["teacher"] == t and r["tag"] == "lsweep"]
        xs, means, mins, maxs = [], [], [], []
        for r in rows:
            if not r["arch"].startswith("ll1_l"):
                continue
            L = r["meta"]["L"]
            xs.append(L)
            arr = np.array(r["rel_mse_seeds"])
            means.append(arr.mean()); mins.append(arr.min()); maxs.append(arr.max())
        order = np.argsort(xs)
        xs = np.array(xs)[order]; means = np.array(means)[order]
        mins = np.array(mins)[order]; maxs = np.array(maxs)[order]
        # plot best-of-seeds as the primary line (representational capacity, as
        # in exp10); band spans min-max across seeds
        ax.plot(xs, mins, "o-", color=C_LL1, label="LL1 student (best of 3)")
        ax.fill_between(xs, mins, maxs, color=C_LL1, alpha=0.2)
        for r in rows:
            if r["arch"] == "swiglu":
                arr = np.array(r["rel_mse_seeds"])
                ax.axhline(arr.min(), color=C_CP, ls="--",
                           label=f"SwiGLU (m={r['meta']['m']})")
            if r["arch"] == "tucker":
                arr = np.array(r["rel_mse_seeds"])
                ax.axhline(arr.min(), color=C_TK, ls=":",
                           label=f"Tucker (r={r['meta']['r']})")
        if t == "ll1_l4":
            ax.axvline(4, color="gray", lw=0.8, ls="-.")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(xs); ax.set_xticklabels([str(int(x)) for x in xs])
        ax.set_xlabel("student block rank L")
        ax.set_title(tnames.get(t, t), fontsize=10)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("val MSE / Var(y)")
    plt.tight_layout()
    out = os.path.join(args.results_dir, "exp18_lsweep.png")
    plt.savefig(out, dpi=180); plt.close()
    log("done", f"saved {out}")

    # ── budget figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(teachers), figsize=(4.2 * len(teachers), 3.6),
                             sharey=True)
    if len(teachers) == 1:
        axes = [axes]
    colors = {"swiglu": "#2c7fb8", "ll1_l4": "#31a354", "tucker": "#d7301f"}
    for ax, t in zip(axes, teachers):
        for arch in args.budget_archs.split(","):
            rows = [r for r in results if r["teacher"] == t and r["arch"] == arch]
            pts = sorted({(r["params"], tuple(r["rel_mse_seeds"])) for r in rows})
            xs = [p for p, _ in pts]
            ms = [np.mean(s) for _, s in pts]
            lo = [np.min(s) for _, s in pts]
            hi = [np.max(s) for _, s in pts]
            ax.plot(xs, ms, "o-", color=colors.get(arch, "k"), label=arch)
            ax.fill_between(xs, lo, hi, color=colors.get(arch, "k"), alpha=0.15)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("student parameters")
        ax.set_title(tnames.get(t, t), fontsize=10)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("val MSE / Var(y)")
    plt.tight_layout()
    out = os.path.join(args.results_dir, "exp18_budget.png")
    plt.savefig(out, dpi=180); plt.close()
    log("done", f"saved {out}")


if __name__ == "__main__":
    main()
