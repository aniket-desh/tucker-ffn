#!/usr/bin/env python3
"""SVD-constructed aligned-SwiGLU student at m = k^2 (Theorem 4.2 upper bound).

For each k in {4,8,16}: regenerate the teacher using exp10's protocol, then
analytically construct an aligned-SwiGLU width m=k^2 by SVD-decomposing each
gate's V_j = R C^(j) (d x r). Each rank-1 SVD term contributes one SwiGLU
hidden unit (gate g = Q[:,j], up u = sigma_nu * U[:,nu], W column W[:,unit]
chosen so W^T x picks out the right combination of P^T x). m = k * k since
rank(V_j) <= min(d, r) = k for each of r = k gates.

Forward equivalence:
    Tucker:   y_a = sum_j sigma(<Q[:,j], x>) [R C^(j)]_a (P^T x)
                  = sum_j sigma(<Q[:,j], x>) (V_j P^T x)_a
    SwiGLU:   y_d = sum_l U[d,l] * (W[:,l]^T x) * sigma(<G[:,l], x>)

Per gate j with V_j = sum_nu sigma_nu u_nu v_nu^T:
    (V_j P^T x)_d = sum_nu sigma_nu u_nu[d] (v_nu^T P^T x)
                  = sum_nu sigma_nu u_nu[d] ((P v_nu)^T x)
=> hidden unit (j, nu): G col = Q[:,j], W col = P v_nu, U col = sigma_nu u_nu

Should land at machine precision (~1e-13) without optimization.

Output: results/exp10/svd_construction.json  {k: {mse, m, n_seeds}}
        plus updated synthetic_fitting.png with new marker series.
"""
import argparse, json, os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from lib import PALETTE, TuckerFFN, log, setup_plot_style  # noqa: E402

# match exp10 teacher protocol
def make_teacher(d, k, gen):
    teacher = TuckerFFN(d, r=k, s=k, diagonal_only=False)
    with torch.no_grad():
        teacher.P.copy_(torch.randn(d, k, generator=gen) / float(d) ** 0.5)
        teacher.Q.copy_(torch.randn(d, k, generator=gen) / float(d) ** 0.5)
        teacher.R.copy_(torch.randn(d, k, generator=gen) / float(d) ** 0.5)
        for _ in range(20):
            C = torch.randn(k, k, k, generator=gen)
            ok = all(torch.linalg.matrix_rank(C[:, :, j]) >= k
                     for j in range(k))
            if ok:
                teacher.C.copy_(C)
                return teacher
        raise RuntimeError(f"could not sample full-rank generic core at k={k}")


def construct_aligned_svd(teacher: TuckerFFN):
    """Return (W, G, U) numpy arrays implementing teacher exactly via SVD.

    Each is (d, m) with m = k * k.
    """
    d, r = teacher.P.shape
    s = teacher.R.shape[1]
    R = teacher.R.detach().double().cpu().numpy()         # (d, s)
    C = teacher.core().detach().double().cpu().numpy()    # (s, r, r)
    P = teacher.P.detach().double().cpu().numpy()         # (d, r)
    Q = teacher.Q.detach().double().cpu().numpy()         # (d, r)
    Wcols, Gcols, Ucols = [], [], []
    for j in range(r):
        Vj = R @ C[:, :, j]                               # (d, r)
        U_, S_, Vt_ = np.linalg.svd(Vj, full_matrices=False)
        rho = min(d, r)
        for nu in range(rho):
            Wcols.append(P @ Vt_[nu, :])                  # (d,)
            Gcols.append(Q[:, j])                         # (d,)
            Ucols.append(S_[nu] * U_[:, nu])              # (d,)
    W = np.stack(Wcols, axis=1)                           # (d, m)
    G = np.stack(Gcols, axis=1)
    U = np.stack(Ucols, axis=1)
    return W, G, U


def silu(x):
    return x / (1.0 + np.exp(-x))


def aligned_swiglu_forward(W, G, U, X):
    """X (n, d) -> Y (n, d). y = sum_l U[:,l] (W[:,l]^T x) silu(G[:,l]^T x)."""
    pre_w = X @ W                                          # (n, m)
    pre_g = X @ G
    h = pre_w * silu(pre_g)
    return h @ U.T                                         # (n, d)


def evaluate(W, G, U, X, Y_teacher):
    Y_pred = aligned_swiglu_forward(W, G, U, X)
    return float(((Y_pred - Y_teacher) ** 2).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--k_values", type=str, default="4,8,16")
    parser.add_argument("--n_val", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results/exp10")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    k_values = [int(s) for s in args.k_values.split(",")]

    out = {}
    for k in k_values:
        teacher_gen = torch.Generator().manual_seed(args.seed + 10_000 * k)
        teacher = make_teacher(args.d, k, teacher_gen)
        teacher.eval()

        # synthetic val data with teacher protocol (exp10 uses x ~ N(0, I_d))
        data_gen = torch.Generator().manual_seed(args.seed + 7 * k)
        x = torch.randn(args.n_val, args.d, generator=data_gen)
        with torch.no_grad():
            y = teacher(x)
        X = x.double().numpy()
        Y = y.double().numpy()

        W, G, U = construct_aligned_svd(teacher)
        m = W.shape[1]
        mse = evaluate(W, G, U, X, Y)
        log("eval", f"k={k} | m=k^2={m:5d} | SVD-construction val mse = {mse:.3e}")
        out[str(k)] = {"k": k, "m": m, "val_mse": mse, "n_val": args.n_val}

    out_path = os.path.join(args.results_dir, "svd_construction.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log("done", f"saved svd_construction.json -> {args.results_dir}/")


if __name__ == "__main__":
    main()
