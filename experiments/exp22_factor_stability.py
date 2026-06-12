#!/usr/bin/env python3
"""experiment 22: factor stability across seeds (interpretability proxy #6).

For each architecture with >=2 trained seeds, match routed units across seed
pairs and measure how similar the matched factors are. Identifiable
decompositions (CP/LL1 atoms, under Kruskal/De Lathauwer-type conditions)
could in principle recur across seeds more than gauge-free latents (Tucker).
At 50M-params-from-scratch scale, all architectures may learn different
features per seed — the differential between architectures is the signal.

Matching:
  swiglu: per layer, greedy max-cosine matching of concatenated unit vectors
          [w_j; g_j; u_j] (each part L2-normalized) between seeds.
  ll1:    match blocks by gate direction g_b cosine; additionally report the
          mean principal-angle overlap of the matched blocks' V_b = U_b A_b^T
          column spaces.
  tucker: match latent gates by Q-column cosine; report matched-gate cosine
          and subspace overlap of V_j ranges (top-L left singular vectors).

Null baseline: same matching against a *shuffled* (random-rotation) copy —
reported alongside, so "stability" is relative to chance.

outputs: results/exp22/exp22_results.json
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import log  # noqa: E402
from lib.ll1_ffn import LL1FFN  # noqa: E402
from lib.tucker_ffn import SwiGLUFFN, TuckerFFN  # noqa: E402
from exp19_interp_proxies import load_ckpt  # noqa: E402


def greedy_match_cosine(X, Y):
    """X (n, d), Y (n, d) rows normalized. greedy max |cos| matching.
    returns mean matched |cos|."""
    S = (X @ Y.T).abs()
    n = S.shape[0]
    used_r, used_c = set(), set()
    vals = []
    flat = S.flatten().argsort(descending=True)
    for f in flat:
        r, c = divmod(f.item(), S.shape[1])
        if r in used_r or c in used_c:
            continue
        used_r.add(r); used_c.add(c)
        vals.append(S[r, c].item())
        if len(vals) == n:
            break
    return float(np.mean(vals))


def normed(W, dim=1):
    return W / (W.norm(dim=dim, keepdim=True) + 1e-9)


def subspace_overlap(V1, V2, L):
    """mean squared cosine of principal angles between top-L ranges."""
    U1 = torch.linalg.svd(V1, full_matrices=False)[0][:, :L]
    U2 = torch.linalg.svd(V2, full_matrices=False)[0][:, :L]
    s = torch.linalg.svdvals(U1.T @ U2)
    return (s ** 2).mean().item()


def unit_vectors(ffn):
    """per-unit concatenated factor vectors (n_units, dim), plus gate dirs."""
    if isinstance(ffn, SwiGLUFFN):
        W = normed(ffn.up_proj.weight)          # (m, d) rows w_j
        G = normed(ffn.gate_proj.weight)
        U = normed(ffn.down_proj.weight.T)      # (m, d) rows u_j
        return torch.cat([W, G, U], dim=1), G
    if isinstance(ffn, LL1FFN):
        G = normed(ffn.gate_proj.weight)        # (B, d)
        return G, G
    if isinstance(ffn, TuckerFFN):
        Q = normed(ffn.Q.T)                     # (r, d) gate latents
        return Q, Q
    raise TypeError(type(ffn))


def per_gate_V(ffn):
    if isinstance(ffn, LL1FFN):
        return ffn.per_gate_matrices(), ffn.block_rank
    if isinstance(ffn, TuckerFFN):
        V = torch.stack([ffn.R @ ffn.C[:, :, j] for j in range(ffn.r)])
        return V, 4  # compare top-4 ranges for tucker (its stable rank scale)
    return None, None


@torch.no_grad()
def compare_pair(model1, model2, max_units=512, max_v=64):
    out = []
    for li, (b1, b2) in enumerate(zip(model1.blocks, model2.blocks)):
        f1, f2 = b1.ffn, b2.ffn
        X, G1 = unit_vectors(f1)
        Y, G2 = unit_vectors(f2)
        if X.shape[0] > max_units:
            idx = torch.randperm(X.shape[0])[:max_units]
            # match a subsample of seed1 units against ALL seed2 units
            X = X[idx]
        mc = greedy_match_cosine(X, Y[:max(Y.shape[0], 1)])
        # null: random gaussian with same shapes
        null = greedy_match_cosine(
            normed(torch.randn_like(X)), normed(torch.randn_like(Y)))
        rec = {"layer": li, "matched_cos": mc, "null_cos": null}
        V1, L = per_gate_V(f1)
        if V1 is not None:
            V2, _ = per_gate_V(f2)
            # match gates by gate-dir cosine first
            S = (G1 @ G2.T).abs()
            n = min(max_v, S.shape[0])
            top_pairs = []
            used_c = set()
            for r in S.max(dim=1).values.argsort(descending=True)[:n]:
                c = S[r].argmax().item()
                if c in used_c:
                    continue
                used_c.add(c)
                top_pairs.append((r.item(), c))
            ov = [subspace_overlap(V1[r], V2[c], L) for r, c in top_pairs]
            null_ov = [subspace_overlap(torch.randn_like(V1[r]),
                                        torch.randn_like(V2[c]), L)
                       for r, c in top_pairs[:8]]
            rec["v_subspace_overlap"] = float(np.mean(ov))
            rec["v_subspace_null"] = float(np.mean(null_ov))
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt_pairs", type=str, required=True,
                    help="semicolon-separated 'path1,path2' pairs")
    ap.add_argument("--results_dir", type=str, default="results/exp22")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    results = []
    for pair in args.ckpt_pairs.split(";"):
        p1, p2 = pair.split(",")
        m1, c1 = load_ckpt(p1, args.device)
        m2, c2 = load_ckpt(p2, args.device)
        layers = compare_pair(m1, m2)
        mean_mc = float(np.mean([r["matched_cos"] for r in layers]))
        mean_null = float(np.mean([r["null_cos"] for r in layers]))
        rec = {"arch": c1["arch"], "seeds": [c1["seed"], c2["seed"]],
               "mean_matched_cos": mean_mc, "mean_null_cos": mean_null,
               "layers": layers}
        if "v_subspace_overlap" in layers[0]:
            rec["mean_v_overlap"] = float(
                np.mean([r["v_subspace_overlap"] for r in layers]))
            rec["mean_v_null"] = float(
                np.mean([r["v_subspace_null"] for r in layers]))
        results.append(rec)
        log("result", f"{c1['arch']} seeds {c1['seed']}vs{c2['seed']}: "
            f"matched_cos={mean_mc:.4f} (null {mean_null:.4f})"
            + (f" V_overlap={rec.get('mean_v_overlap'):.4f} "
               f"(null {rec.get('mean_v_null'):.4f})"
               if "mean_v_overlap" in rec else ""))
        del m1, m2
        torch.cuda.empty_cache()
        with open(os.path.join(args.results_dir, "exp22_results.json"), "w") as f:
            json.dump(results, f, indent=2)
    log("done", "exp22 complete")


if __name__ == "__main__":
    main()
