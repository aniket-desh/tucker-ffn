#!/usr/bin/env python3
"""experiment 24 (sprint 2, Exp D): superposition recovery — interpretability
with a ground-truth answer.

Generative model (all ground truth known):
  latent features z in R^K, each active independently w.p. p, value ~ N(0,1)
  input  x = F z                      (F: (d, K) random unit features; K >= d
                                       gives superposition)
  target y = sum_{(i,j) in P} (z_i * z_j) v_{ij}   (rank-one bilinear atoms:
                                       v_ij (x) f_i (x) f_j, |P| pairs)

Two pair topologies:
  random : pairs drawn uniformly — no shared structure (CP's home turf)
  hub    : G gate-features, each paired with L distinct main-features —
           exactly LL1's prior (one route gates a rank-L block)

Students at matched parameter budget: swiglu, swiglu + route-L1 (sparse CP),
ll1_l{2,4}, tucker. Metrics:

  rel. val MSE              function fit
  recovery rate             fraction of true pairs matched by a learned atom
                            with product-cosine > 0.8 (greedy assignment;
                            score = |cos(w,f_i)*cos(g,f_j)*cos(u,v_ij)|)
  mean matched score        average alignment of matched atoms
  causal specificity        ablate the atom matched to pair (i,j): error
                            increase on samples where z_i z_j != 0 vs samples
                            where it == 0 (ratio; ground-truth-causal test)

This is the test Thomas's "sparse CPD is more interpretable" claim implies:
if explicit sparsity helps the student recover the TRUE atoms, sparse CPD
interpretability is real, not aesthetic.

outputs: results/exp24/exp24_results.json + exp24_recovery.png
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
import torch.nn.functional as F_t

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import log, setup_plot_style  # noqa: E402
from lib.tucker_ffn import SwiGLUFFN, TuckerFFN  # noqa: E402
from lib.ll1_ffn import LL1FFN, ll1_blocks_for_params  # noqa: E402
from exp18_ll1_synthetic import tucker_r_for_params  # noqa: E402


def make_task(d, K, n_pairs, topology, p_active, seed, device):
    g = torch.Generator().manual_seed(seed)
    F = torch.randn(d, K, generator=g)
    F = F / F.norm(dim=0, keepdim=True)
    if topology == "random":
        idx = torch.randperm(K * (K - 1), generator=g)[:n_pairs]
        pairs = [(int(t // (K - 1)), int(t % (K - 1))) for t in idx]
        pairs = [(i, j if j < i else j + 1) for i, j in pairs]
    elif topology == "hub":
        # G hubs, each gating L mains: n_pairs = G * L with L = 4
        L = 4
        G = n_pairs // L
        gates = torch.randperm(K, generator=g)[:G].tolist()
        mains = torch.randperm(K, generator=g)[G:G + n_pairs].tolist()
        pairs = [(mains[h * L + l], gates[h]) for h in range(G) for l in range(L)]
    else:
        raise ValueError(topology)
    V = torch.randn(d, n_pairs, generator=g)
    V = V / V.norm(dim=0, keepdim=True)
    return F.to(device), pairs, V.to(device)


def sample_batch(F, pairs, V, p_active, n, device, gen=None):
    K = F.shape[1]
    z = torch.randn(n, K, device=device, generator=gen) * \
        (torch.rand(n, K, device=device, generator=gen) < p_active)
    x = z @ F.T
    prod = torch.stack([z[:, i] * z[:, j] for i, j in pairs], dim=1)  # (n, P)
    y = prod @ V.T
    return x, y, z, prod


def build_student(arch, d, budget, seed):
    torch.manual_seed(seed)
    if arch.startswith("swiglu"):
        m = max(1, round(budget / (3 * d)))
        return SwiGLUFFN(d, m), {"m": m}
    if arch.startswith("ll1_l"):
        L = int(arch[len("ll1_l"):])
        B = ll1_blocks_for_params(d, L, budget)
        return LL1FFN(d, n_blocks=B, block_rank=L), {"B": B, "L": L}
    if arch == "tucker":
        r = tucker_r_for_params(d, budget)
        return TuckerFFN(d, r=r, s=r), {"r": r}
    raise ValueError(arch)


def fit(student, data, steps, lr, batch, device, l1_lambda=0.0):
    x_tr, y_tr, x_va, y_va = data
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                       eta_min=lr * 0.01)
    n = x_tr.shape[0]
    student.train()
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        xb = x_tr[idx]
        pred = student(xb)
        loss = F_t.mse_loss(pred, y_tr[idx])
        if l1_lambda > 0 and hasattr(student, "gate_proj"):
            s = F_t.silu(student.gate_proj(xb))
            loss = loss + l1_lambda * s.abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    student.eval()
    with torch.no_grad():
        return F_t.mse_loss(student(x_va), y_va).item() / y_va.var().item()


@torch.no_grad()
def atom_factors(student):
    """list of (u, w, g) unit vectors per learned atom. LL1 blocks are
    SVD-canonicalized into rank-one atoms sharing the block gate."""
    if isinstance(student, SwiGLUFFN):
        W = student.up_proj.weight        # (m, d) rows w
        G = student.gate_proj.weight
        U = student.down_proj.weight.T    # (m, d) rows u
        return [(U[j], W[j], G[j]) for j in range(W.shape[0])]
    if isinstance(student, LL1FFN):
        A, G, U = student.block_factors()  # (B,d,L), (d,B), (B,L,d)
        out = []
        for b in range(student.n_blocks):
            V = U[b].T @ A[b].T            # (d, d) = sum_l u w^T
            P, S, Qt = torch.linalg.svd(A[b] @ U[b], full_matrices=False)
            # A[b] (d,L) @ U[b] (L,d) = sum_l a_l u_l^T -> rows main, cols out
            for l in range(student.block_rank):
                w = P[:, l]
                u = Qt[l]
                out.append((u * S[l].sqrt(), w * S[l].sqrt(), G[:, b]))
        return out
    if isinstance(student, TuckerFFN):
        out = []
        for j in range(student.r):
            Vj = student.R @ student.C[:, :, j]   # (d, r) out-by-main
            P, S, Qt = torch.linalg.svd(Vj, full_matrices=False)
            for l in range(min(4, len(S))):       # top-4 directions per gate
                out.append((P[:, l] * S[l].sqrt(),
                            (Qt[l] @ student.P.T), student.Q[:, j]))
        return out
    raise TypeError(type(student))


@torch.no_grad()
def recovery_metrics(student, F, pairs, V, thresh=0.8):
    atoms = atom_factors(student)
    nrm = lambda v: v / (v.norm() + 1e-9)
    scores = torch.zeros(len(pairs), len(atoms))
    for pi, (i, j) in enumerate(pairs):
        fi, fj, vij = nrm(F[:, i]), nrm(F[:, j]), nrm(V[:, pi])
        for ai, (u, w, g) in enumerate(atoms):
            s = abs(float(nrm(w) @ fi)) * abs(float(nrm(g) @ fj)) * \
                abs(float(nrm(u) @ vij))
            scores[pi, ai] = s
    # greedy assignment
    matched, used = [], set()
    flat = scores.flatten().argsort(descending=True)
    for f in flat:
        pi, ai = divmod(int(f), len(atoms))
        if pi in {m[0] for m in matched} or ai in used:
            continue
        matched.append((pi, ai, float(scores[pi, ai])))
        used.add(ai)
        if len(matched) == len(pairs):
            break
    vals = [v for _, _, v in matched]
    return {
        "recovery_rate": float(np.mean([v > thresh for v in vals])),
        "mean_matched_score": float(np.mean(vals)),
        "n_atoms": len(atoms),
        "matched": matched[:64],
    }


@torch.no_grad()
def causal_specificity(student, matched, F, pairs, V, p_active, device):
    """ablate the atom matched to a pair; compare MSE increase on samples
    where the pair is active vs inactive. Mean over up to 16 matched pairs
    with score > 0.5. (SwiGLU only: atoms map to hidden units 1:1.)"""
    if not isinstance(student, SwiGLUFFN):
        return None
    gen = torch.Generator(device=device).manual_seed(123)
    x, y, z, prod = sample_batch(F, pairs, V, p_active, 8192, device, gen)
    base = ((student(x) - y) ** 2).mean(-1)
    ratios = []
    for pi, ai, sc in matched[:16]:
        if sc < 0.5:
            continue
        w = student.up_proj.weight[ai].clone()
        student.up_proj.weight[ai] = 0
        abl = ((student(x) - y) ** 2).mean(-1)
        student.up_proj.weight[ai] = w
        act = prod[:, pi].abs() > 1e-6
        if act.sum() < 8 or (~act).sum() < 8:
            continue
        d_on = (abl - base)[act].mean().item()
        d_off = (abl - base)[~act].mean().item()
        ratios.append(d_on / (abs(d_off) + 1e-9))
    return float(np.median(ratios)) if ratios else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--K", type=int, default=96)          # K > d: superposition
    ap.add_argument("--n_pairs", type=int, default=32)
    ap.add_argument("--p_active", type=float, default=0.2)
    ap.add_argument("--budget", type=int, default=9216)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n_train", type=int, default=100_000)
    ap.add_argument("--n_val", type=int, default=10_000)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--topologies", type=str, default="random,hub")
    ap.add_argument("--archs", type=str,
                    default="swiglu,swiglu_l1,ll1_l2,ll1_l4,tucker")
    ap.add_argument("--l1_lambda", type=float, default=3e-3)
    ap.add_argument("--results_dir", type=str, default="results/exp24")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    res_path = os.path.join(args.results_dir, "exp24_results.json")

    if not args.plot_only:
        results = []
        for topo in args.topologies.split(","):
            F, pairs, V = make_task(args.d, args.K, args.n_pairs, topo,
                                    args.p_active, seed=7, device=args.device)
            gen = torch.Generator(device=args.device).manual_seed(11)
            x_tr, y_tr, _, _ = sample_batch(F, pairs, V, args.p_active,
                                            args.n_train, args.device, gen)
            x_va, y_va, _, _ = sample_batch(F, pairs, V, args.p_active,
                                            args.n_val, args.device, gen)
            scale = y_tr.std() + 1e-9
            y_tr, y_va = y_tr / scale, y_va / scale
            V_s = V / scale
            data = (x_tr, y_tr, x_va, y_va)
            for arch in args.archs.split(","):
                base_arch = "swiglu" if arch == "swiglu_l1" else arch
                lam = args.l1_lambda if arch == "swiglu_l1" else 0.0
                for seed in range(args.n_seeds):
                    student, meta = build_student(base_arch, args.d,
                                                  args.budget, seed * 17 + 3)
                    student = student.to(args.device)
                    t0 = time.time()
                    rel = fit(student, data, args.steps, args.lr, args.batch,
                              args.device, l1_lambda=lam)
                    rec = recovery_metrics(student, F, pairs, V_s)
                    spec = causal_specificity(student, rec.pop("matched"),
                                              F, pairs, V_s, args.p_active,
                                              args.device)
                    row = {"topology": topo, "arch": arch, "seed": seed,
                           "meta": meta, "rel_mse": rel, **rec,
                           "causal_specificity": spec}
                    results.append(row)
                    log("train", f"{topo:6s} {arch:10s} s{seed} relMSE={rel:.4f} "
                        f"recov={rec['recovery_rate']:.2f} "
                        f"score={rec['mean_matched_score']:.3f} "
                        f"spec={spec if spec is None else round(spec,1)} "
                        f"({time.time()-t0:.0f}s)")
                    with open(res_path, "w") as f:
                        json.dump(results, f, indent=2)
                    del student
                    torch.cuda.empty_cache()
        log("done", f"exp24 complete -> {res_path}")

    with open(res_path) as f:
        results = json.load(f)

    setup_plot_style()
    topos = sorted({r["topology"] for r in results})
    archs = [a for a in ["swiglu", "swiglu_l1", "ll1_l2", "ll1_l4", "tucker"]
             if any(r["arch"] == a for r in results)]
    colors = {"swiglu": "#2c7fb8", "swiglu_l1": "#16a0a0",
              "ll1_l2": "#74c476", "ll1_l4": "#31a354", "tucker": "#d7301f"}
    fig, axes = plt.subplots(1, 2 * len(topos), figsize=(4.4 * 2 * len(topos), 3.6))
    for ti, topo in enumerate(topos):
        for mi, metric in enumerate(["rel_mse", "recovery_rate"]):
            ax = axes[ti * 2 + mi]
            for ai, arch in enumerate(archs):
                vals = [r[metric] for r in results
                        if r["topology"] == topo and r["arch"] == arch]
                ax.bar(ai, np.mean(vals), yerr=np.std(vals),
                       color=colors[arch], capsize=3)
            ax.set_xticks(range(len(archs)))
            ax.set_xticklabels(archs, rotation=45, fontsize=7)
            ax.set_title(f"{topo}: {metric}", fontsize=9)
            if metric == "rel_mse":
                ax.set_yscale("log")
    plt.tight_layout()
    out = os.path.join(args.results_dir, "exp24_recovery.png")
    plt.savefig(out, dpi=180)
    plt.close()
    log("done", f"saved {out}")


if __name__ == "__main__":
    main()
