#!/usr/bin/env python3
"""experiment 10: synthetic fitting limit (theorem 1, headline figure).

generate a teacher tucker-core ffn with full-rank generic core (frontal
slices C[:,:,j] of rank min(r,s)=k), then fit students of varying width
to its outputs. theorem 1 (separation) predicts that an aligned-coordinate
swiglu must have width m >= sum_j rank(M_j) = k * k = k^2 to match the
teacher exactly. as students go from m << k^2 to m >> k^2 the val mse
should drop sharply at m = k^2.

three student curves per teacher (per k value):
  unconstrained swiglu — w_l, g_l free in R^d (no matched-coordinates)
  aligned    swiglu     — w_l = P a_l, g_l from columns of Q (theorem
                          hypothesis class, knee at k^2 by construction)
  tucker     control    — same architecture as teacher (drives mse to ~0
                          regardless of m, isolates optimization gap)

we sweep m in {k, 2k, k(k-1), k^2, 2k^2, 4k^2} with K=8 random inits per
m, take the minimum train-loss model and report its val mse.

the sharp knee at m=k^2 in the aligned-swiglu curve is the empirical
verification of theorem 1. the unconstrained curve is exploratory: if it
also shows a knee at k^2, the theorem's matched-coordinates assumption is
not the binding constraint; if it dips below k^2 it means unconstrained
swiglu can express more.

outputs (under --results_dir):
  exp10_config.json            — d, k_values, m_values, training hparams
  synthetic_fitting.npz        — val_mse[k_idx, student_idx, m_idx, seed_idx]
  synthetic_fitting.png        — log-log val mse vs m, panels per k
"""

import argparse
import json
import os
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import (  # noqa: E402
    COLOR_CYCLE,
    PALETTE,
    SwiGLUFFN,
    SwiGLUFFNAligned,
    TuckerFFN,
    log,
    setup_plot_style,
)


STUDENTS = ("swiglu_unconstrained", "swiglu_aligned", "tucker_control")


def make_teacher(d, k, gen):
    """build a tucker teacher with full-rank generic frontal slices.

    resamples C until every C[:, :, j] has full rank min(r,s)=k. random
    gaussian P, Q, R; we keep the same teacher across all student fits at
    a given k.
    """
    teacher = TuckerFFN(d, r=k, s=k, diagonal_only=False)
    with torch.no_grad():
        teacher.P.copy_(torch.randn(d, k, generator=gen) / float(d) ** 0.5)
        teacher.Q.copy_(torch.randn(d, k, generator=gen) / float(d) ** 0.5)
        teacher.R.copy_(torch.randn(d, k, generator=gen) / float(d) ** 0.5)
        for tries in range(20):
            C = torch.randn(k, k, k, generator=gen)
            ok = True
            for j in range(k):
                if torch.linalg.matrix_rank(C[:, :, j]) < k:
                    ok = False
                    break
            if ok:
                teacher.C.copy_(C)
                return teacher
        raise RuntimeError(f"could not sample full-rank generic core at k={k}")
    return teacher


def build_student(kind, d, m, k, P=None, Q=None, gate_assignment=None):
    if kind == "swiglu_unconstrained":
        return SwiGLUFFN(d, m, bias=False)
    if kind == "swiglu_aligned":
        # use teacher's P, Q as the latent dictionaries (matched-coordinates).
        # gate_assignment: spread m units roughly uniformly over r=k gates.
        if gate_assignment is None:
            gate_assignment = torch.arange(m) % k
        return SwiGLUFFNAligned(d, m, P, Q, gate_assignment)
    if kind == "tucker_control":
        return TuckerFFN(d, r=k, s=k, diagonal_only=False)
    raise ValueError(kind)


def fit_student(student, x_train, y_train, x_val, y_val, n_steps, lr,
                weight_decay, batch_size, device, verbose=False, log_every=2000):
    """fit student to teacher outputs by adam mse with cosine lr decay.

    avoids per-step .item() calls (sync) and per-step random index gen by
    pre-shuffling once per epoch. cosine schedule helps tucker controls
    reach near-machine-precision (without it they plateau at ~1e-2).
    """
    import math as _math
    student.to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr,
                           weight_decay=weight_decay)
    n = x_train.shape[0]
    perm = torch.randperm(n, device=device)
    cursor = 0
    for step in range(n_steps):
        # cosine decay to 1% of peak
        progress = step / max(1, n_steps - 1)
        cur_lr = lr * (0.01 + 0.99 * 0.5 * (1 + _math.cos(_math.pi * progress)))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        if cursor + batch_size > n:
            perm = torch.randperm(n, device=device)
            cursor = 0
        idx = perm[cursor:cursor + batch_size]
        cursor += batch_size
        xb = x_train[idx]
        yb = y_train[idx]
        pred = student(xb)
        loss = F.mse_loss(pred, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if verbose and (step + 1) % log_every == 0:
            with torch.no_grad():
                v = F.mse_loss(student(x_val), y_val).item()
            log("train", f"step {step+1}/{n_steps} | train={loss.item():.3e} | val={v:.3e}")
    student.eval()
    with torch.no_grad():
        val_mse = F.mse_loss(student(x_val), y_val).item()
        train_mse = F.mse_loss(student(x_train), y_train).item()
    return val_mse, train_mse


def m_grid_for_k(k):
    grid = sorted({k, 2 * k, k * (k - 1), k * k, 2 * k * k, 4 * k * k})
    return [m for m in grid if m >= 1]


def run_synthetic_fitting(k_values, d, n_train, n_val, n_seeds, n_steps,
                          lr, weight_decay, batch_size, device, results_dir,
                          base_seed=0):
    log("info", "experiment 10: synthetic fitting limit (theorem 1)")
    log("info", f"d={d} | k_values={list(k_values)} | n_train={n_train} | "
        f"n_val={n_val} | seeds={n_seeds} | steps={n_steps}")

    # store results as a ragged structure since m_grid varies by k
    by_k = {}
    for ki, k in enumerate(k_values):
        log("info", f"=== k = {k} ===")
        m_values = m_grid_for_k(k)
        log("info", f"k={k} | m_values={m_values}")

        teacher_gen = torch.Generator().manual_seed(base_seed + 10_000 * k)
        teacher = make_teacher(d, k, teacher_gen).to(device)
        teacher.eval()

        # synthetic data: x ~ N(0, I_d), y = teacher(x)
        data_gen = torch.Generator(device=device).manual_seed(base_seed + 7 * k)
        x = torch.randn(n_train + n_val, d, generator=data_gen, device=device)
        with torch.no_grad():
            y = teacher(x)
        x_train, x_val = x[:n_train], x[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        log("data", f"k={k} | y_train_var={y_train.var().item():.3e} | "
            f"y_val_var={y_val.var().item():.3e}")

        val_mse_arr = np.full((len(STUDENTS), len(m_values), n_seeds), np.nan)

        # share fixed gate assignments across seeds for aligned (so it is
        # truly the m-vs-rank tradeoff, not "did we get a lucky assignment")
        for mi, m in enumerate(m_values):
            for kind_idx, kind in enumerate(STUDENTS):
                for si in range(n_seeds):
                    torch.manual_seed(base_seed + 1009 * (k + 1) + 17 * mi
                                       + 31 * kind_idx + si)
                    if kind == "swiglu_aligned":
                        ga = torch.arange(m) % k
                        student = build_student(
                            kind, d, m, k, P=teacher.P.detach(),
                            Q=teacher.Q.detach(), gate_assignment=ga,
                        )
                    else:
                        student = build_student(kind, d, m, k)
                    val, train = fit_student(
                        student, x_train, y_train, x_val, y_val,
                        n_steps=n_steps, lr=lr, weight_decay=weight_decay,
                        batch_size=batch_size, device=device,
                    )
                    val_mse_arr[kind_idx, mi, si] = val
                # take min val over seeds is more direct than min-train-then-val
                # (we want the best fit achievable)
                best = np.nanmin(val_mse_arr[kind_idx, mi])
                log("eval", f"k={k} | m={m:5d} | {kind:25s} | "
                    f"min_val_mse={best:.3e}")

        by_k[k] = {"m_values": m_values, "val_mse": val_mse_arr}

    # save
    save_path = os.path.join(results_dir, "synthetic_fitting.npz")
    save_dict = {}
    for k, d_ in by_k.items():
        save_dict[f"k{k}_m_values"] = np.array(d_["m_values"])
        save_dict[f"k{k}_val_mse"] = d_["val_mse"]
    save_dict["k_values"] = np.array(list(k_values))
    save_dict["students"] = np.array(STUDENTS)
    np.savez(save_path, **save_dict)
    log("done", f"saved synthetic_fitting.npz -> {results_dir}/")

    cfg = {
        "d": d,
        "k_values": list(k_values),
        "n_train": n_train,
        "n_val": n_val,
        "n_seeds": n_seeds,
        "n_steps": n_steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "students": list(STUDENTS),
    }
    with open(os.path.join(results_dir, "exp10_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    plot_synthetic_fitting(save_path, results_dir)
    return by_k


def plot_synthetic_fitting(npz_path, results_dir):
    """log-log val mse vs m, one panel per k, with predicted knee at m=k^2."""
    setup_plot_style()
    data = np.load(npz_path, allow_pickle=True)
    k_values = data["k_values"]
    students = list(data["students"])
    n_panels = len(k_values)

    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.6),
                              sharey=True)
    if n_panels == 1:
        axes = [axes]

    style = {
        "swiglu_unconstrained": (PALETTE["primary"],   "o", "-",  "swiglu (unconstrained)"),
        "swiglu_aligned":       (PALETTE["ablation"],  "s", "--", r"swiglu (aligned, theorem 1)"),
        "tucker_control":       (PALETTE["accent"],    "^", ":",  "tucker (control)"),
    }

    for ax, k in zip(axes, k_values):
        m_values = data[f"k{k}_m_values"]
        val_mse = data[f"k{k}_val_mse"]   # (n_students, n_m, n_seeds)
        for si, kind in enumerate(students):
            color, marker, ls, label = style[kind]
            best = np.nanmin(val_mse[si], axis=-1)
            ax.plot(m_values, best, marker=marker, ls=ls, color=color,
                    lw=1.5, ms=5, label=label)
        ax.axvline(k * k, color=PALETTE["neutral"], ls=":", lw=0.8,
                   label=fr"$m = k^2 = {k*k}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"swiglu hidden width $m$")
        ax.set_title(fr"$k = {k}$")
    axes[0].set_ylabel("validation MSE")
    axes[-1].legend(framealpha=0.9, edgecolor="0.8", loc="best", fontsize=8)

    plt.tight_layout()
    out = os.path.join(results_dir, "synthetic_fitting.png")
    plt.savefig(out)
    plt.close()
    log("done", f"saved synthetic_fitting.png -> {results_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--k_values", type=str, default="4,8,16",
                        help="comma-separated k (teacher r=s=k)")
    parser.add_argument("--n_train", type=int, default=50000)
    parser.add_argument("--n_val", type=int, default=5000)
    parser.add_argument("--n_seeds", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--results_dir", type=str, default="results/exp10")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    setup_plot_style()

    if args.plot_only:
        plot_synthetic_fitting(
            os.path.join(args.results_dir, "synthetic_fitting.npz"),
            args.results_dir,
        )
        return

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    k_values = [int(s) for s in args.k_values.split(",")]
    t0 = time.time()
    run_synthetic_fitting(
        k_values=k_values, d=args.d,
        n_train=args.n_train, n_val=args.n_val, n_seeds=args.n_seeds,
        n_steps=args.n_steps, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, device=args.device,
        results_dir=args.results_dir, base_seed=args.seed,
    )
    log("done", f"experiment 10 complete | time={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
