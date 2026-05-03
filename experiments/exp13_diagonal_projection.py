#!/usr/bin/env python3
"""experiment 13: diagonal-bottleneck cost on a trained tucker model.

two analyses on each tucker checkpoint from exp11:

(a) c-diagonal projection (the parameterization-level bottleneck):
      build C_diag with C_diag[a,i,j] = C[a,i,j] if a==i==j else 0.
      sweep lambda in [0, 1] interpolating C(lambda) = (1-lambda) C
      + lambda C_diag, evaluate val perplexity at each lambda.
      lambda=0 is the trained tucker; lambda=1 is forced superdiagonal,
      which makes the layer parameterization-equivalent to a swiglu of
      width r (note section IV recovery condition).

(b) per-gate V_j svd truncation (the rank-level bottleneck of theorem 1):
      for each gate j, replace V_j = R C^(j) by its rank-rho svd truncation,
      effectively replacing C^(j) by R^+ U_rho diag(s_rho) Vt_rho. sweep
      rho in {1, 2, ..., min(r, s)} and evaluate val perplexity. the
      trained tucker is rho = min(r,s); the rho=1 case forces V_j to be
      rank-1 per gate, which is exactly the constraint an aligned swiglu
      with width m = r imposes (one unit per gate). this curve traces
      "aligned-swiglu width m -> perplexity" from m = r (rho=1) up to
      m = sum_j rank(V_j) (rho = min(r,s)).

we apply the projection in all layers simultaneously (same lambda or rho
across the network).

inputs: trained tucker checkpoints from exp11 + a held-out validation
text (built like in exp11).

outputs (under --results_dir):
  diagonal_projection.json
  diagonal_projection_bar.png        — bar chart: trained vs lambda=1
  diagonal_projection_dose.png       — perplexity vs lambda
  rank_truncation.json
  rank_truncation_curve.png          — perplexity vs rho per layer (or pooled)
"""

import argparse
import glob
import json
import math
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from contextlib import contextmanager  # noqa: E402

from lib import COLOR_CYCLE, PALETTE, log, setup_plot_style  # noqa: E402
from lib.lm import LM, make_lm  # noqa: E402


def load_tucker_lm(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = sd["cfg"]
    model = make_lm("tucker", d=cfg["d"], n_heads=cfg["n_heads"],
                     n_layers=cfg["n_layers"], vocab_size=cfg["vocab_size"],
                     max_seq_len=cfg["seq_len"], r=cfg["tucker_r"],
                     s=cfg["tucker_s"])
    model.load_state_dict(sd["model_state_dict"])
    return model, cfg


def build_val_set(tokenizer_name, seq_len, n_seqs, device,
                  config_name="sample-10BT", seed=12345):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", config_name,
                       split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    needed = (seq_len + 1) * n_seqs
    buf = []
    for ex in ds:
        text = ex.get("text", "")
        if not text:
            continue
        ids = tok.encode(text)
        ids.append(tok.eos_token_id if tok.eos_token_id is not None
                    else tok.encode("\n")[0])
        buf.extend(ids)
        if len(buf) >= needed:
            break
    chunk = torch.tensor(buf[:needed], dtype=torch.long).view(n_seqs, seq_len + 1)
    return chunk[:, :-1].to(device), chunk[:, 1:].to(device)


@torch.no_grad()
def eval_perplexity(model, val_inp, val_tgt, batch_size, device):
    model.eval()
    losses = []
    for i in range(0, val_inp.size(0), batch_size):
        ib = val_inp[i:i + batch_size]
        tb = val_tgt[i:i + batch_size]
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(ib, targets=tb)
        losses.append(loss.item())
    return math.exp(float(np.mean(losses)))


@contextmanager
def projected_C(model, lam):
    """temporarily replace each layer's C with (1-lam)*C + lam*C_diag.

    operates only on TuckerFFN modules with non-diagonal C parameter.
    """
    saved = {}
    for li, blk in enumerate(model.blocks):
        ffn = blk.ffn
        if not hasattr(ffn, "C") or ffn.C is None:
            continue
        if ffn.diagonal_only:
            continue
        C = ffn.C.data
        s, r1, r2 = C.shape
        kdim = min(s, r1, r2)
        C_diag = torch.zeros_like(C)
        idx = torch.arange(kdim, device=C.device)
        C_diag[idx, idx, idx] = C[idx, idx, idx]
        saved[li] = C.clone()
        ffn.C.data = (1.0 - lam) * C + lam * C_diag
    try:
        yield
    finally:
        for li, C0 in saved.items():
            model.blocks[li].ffn.C.data = C0


@contextmanager
def truncated_per_gate(model, rho):
    """for each gate j and each layer, replace V_j = R C^(j) by its rank-rho
    SVD truncation, equivalently replacing C^(j) by C^(j)' s.t. R C^(j)' has
    rank <= rho.

    uses pseudo-inverse: V_j_trunc = U_rho diag(s_rho) Vt_rho. update
    C^(j) <- R^+ V_j_trunc. R is (d, s) and C^(j) is (s, r); using R^+ as
    (s, d) preserves the rank-rho range under R since V_j_trunc is in
    column-space of R only if V_j was. for stability we do least-squares
    update via lstsq.
    """
    saved = {}
    for li, blk in enumerate(model.blocks):
        ffn = blk.ffn
        if not hasattr(ffn, "C") or ffn.C is None:
            continue
        if ffn.diagonal_only:
            continue
        C = ffn.C.data                # (s, r, r)
        R = ffn.R.data.float()        # (d, s)
        s_dim, r_dim, r2_dim = C.shape
        # pre-factor R: solve R x = b for x via lstsq
        saved[li] = C.clone()
        for j in range(r2_dim):
            Cj = C[:, :, j].float()           # (s, r)
            Vj = R @ Cj                       # (d, r)
            U, S, Vt = torch.linalg.svd(Vj, full_matrices=False)
            rho_eff = min(rho, S.numel())
            Vj_trunc = U[:, :rho_eff] @ torch.diag(S[:rho_eff]) @ Vt[:rho_eff, :]
            # solve R Cj' = Vj_trunc  =>  Cj' = R^+ Vj_trunc
            Cj_new = torch.linalg.lstsq(R, Vj_trunc).solution  # (s, r)
            C[:, :, j] = Cj_new.to(C.dtype)
        ffn.C.data = C
    try:
        yield
    finally:
        for li, C0 in saved.items():
            model.blocks[li].ffn.C.data = C0


def run_diagonal_projection(model, val_inp, val_tgt, batch_size, device,
                              lambdas):
    """sweep lambda interpolating C -> C_diag, return list of (lambda, ppl)."""
    out = []
    for lam in lambdas:
        with projected_C(model, lam):
            ppl = eval_perplexity(model, val_inp, val_tgt, batch_size, device)
        log("eval", f"lambda={lam:.2f} | perplexity={ppl:.3f}")
        out.append({"lambda": lam, "perplexity": ppl})
    return out


def run_rank_truncation(model, val_inp, val_tgt, batch_size, device,
                         rho_values):
    out = []
    for rho in rho_values:
        with truncated_per_gate(model, rho):
            ppl = eval_perplexity(model, val_inp, val_tgt, batch_size, device)
        log("eval", f"rho={rho:3d} | perplexity={ppl:.3f}")
        out.append({"rho": rho, "perplexity": ppl})
    return out


def plot_results(results_by_ckpt, results_dir):
    setup_plot_style()
    # per-ckpt: bar (lambda=0 vs 1), dose curve, rank truncation curve
    fig, ax = plt.subplots(figsize=(6, 4))
    n = len(results_by_ckpt)
    width = 0.35
    xs = np.arange(n)
    full = []
    diag = []
    tags = []
    for i, (tag, d_) in enumerate(results_by_ckpt.items()):
        lam_results = d_["diagonal_projection"]
        full_ppl = lam_results[0]["perplexity"]
        diag_ppl = lam_results[-1]["perplexity"]
        full.append(full_ppl)
        diag.append(diag_ppl)
        tags.append(tag)
    ax.bar(xs - width/2, full, width=width, color=PALETTE["primary"],
            edgecolor="0.3", label=r"trained tucker ($\lambda=0$)")
    ax.bar(xs + width/2, diag, width=width, color=PALETTE["ablation"],
            edgecolor="0.3", label=r"diagonal-projected ($\lambda=1$)")
    for i, (f, dd) in enumerate(zip(full, diag)):
        ax.text(i - width/2, f * 1.02, f"{f:.2f}", ha="center", fontsize=9)
        ax.text(i + width/2, dd * 1.02, f"{dd:.2f}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(tags, rotation=15, ha="right")
    ax.set_ylabel("validation perplexity")
    ax.set_yscale("log")
    ax.legend(framealpha=0.9, edgecolor="0.8")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "diagonal_projection_bar.png"))
    plt.close()
    log("done", f"saved diagonal_projection_bar.png")

    # dose-response curve
    fig, ax = plt.subplots(figsize=(6, 4))
    for ti, (tag, d_) in enumerate(results_by_ckpt.items()):
        lam = [r["lambda"] for r in d_["diagonal_projection"]]
        ppl = [r["perplexity"] for r in d_["diagonal_projection"]]
        ax.plot(lam, ppl, marker="o", color=COLOR_CYCLE[ti % len(COLOR_CYCLE)],
                lw=1.5, label=tag)
    ax.set_xlabel(r"$\lambda$ (interpolation toward diagonal)")
    ax.set_ylabel("validation perplexity")
    ax.set_yscale("log")
    ax.legend(framealpha=0.9, edgecolor="0.8")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "diagonal_projection_dose.png"))
    plt.close()
    log("done", f"saved diagonal_projection_dose.png")

    # rank truncation curve
    fig, ax = plt.subplots(figsize=(6, 4))
    for ti, (tag, d_) in enumerate(results_by_ckpt.items()):
        if "rank_truncation" not in d_:
            continue
        rho = [r["rho"] for r in d_["rank_truncation"]]
        ppl = [r["perplexity"] for r in d_["rank_truncation"]]
        ax.plot(rho, ppl, marker="s", color=COLOR_CYCLE[ti % len(COLOR_CYCLE)],
                lw=1.5, label=tag)
    ax.set_xlabel(r"per-gate svd truncation rank $\rho$")
    ax.set_ylabel("validation perplexity")
    ax.set_yscale("log")
    ax.legend(framealpha=0.9, edgecolor="0.8")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "rank_truncation_curve.png"))
    plt.close()
    log("done", f"saved rank_truncation_curve.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_glob", type=str,
                        default="results/exp11/tucker_seed*/checkpoint_final.pt")
    parser.add_argument("--results_dir", type=str, default="results/exp13")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--n_val_seqs", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lambdas", type=str,
                        default="0.0,0.25,0.5,0.75,0.9,1.0")
    parser.add_argument("--rho_max", type=int, default=None,
                        help="max truncation rank (default: r); we sweep "
                             "1..rho_max via geometric grid")
    parser.add_argument("--plot_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.plot_only:
        path = os.path.join(args.results_dir, "results.json")
        with open(path) as f:
            r = json.load(f)
        plot_results(r, args.results_dir)
        return

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = sorted(glob.glob(args.ckpt_glob))
    if not paths:
        log("error", f"no checkpoints found at {args.ckpt_glob}")
        return
    log("info", f"found {len(paths)} checkpoints")

    log("info", "building val set")
    val_inp, val_tgt = build_val_set(
        args.tokenizer, args.seq_len, args.n_val_seqs, args.device,
    )
    log("info", f"val_inp={val_inp.shape}")

    lambdas = [float(x) for x in args.lambdas.split(",")]
    results_by_ckpt = {}

    for p in paths:
        tag = pathlib.Path(p).parent.name
        log("info", f"=== analyzing {tag} ({p}) ===")
        model, cfg = load_tucker_lm(p)
        model.to(args.device)

        # baseline
        ppl0 = eval_perplexity(model, val_inp, val_tgt, args.batch_size,
                                args.device)
        log("result", f"{tag} | trained perplexity = {ppl0:.3f}")

        # (a) lambda sweep
        log("info", "diagonal-projection lambda sweep")
        diag_results = run_diagonal_projection(
            model, val_inp, val_tgt, args.batch_size, args.device, lambdas,
        )

        # (b) rho sweep — geometric grid
        rho_max = args.rho_max or cfg["tucker_r"]
        # log-spaced ints
        rho_grid = sorted({1, 2, 4, max(1, rho_max // 16),
                            max(1, rho_max // 8), max(1, rho_max // 4),
                            max(1, rho_max // 2), rho_max})
        rho_grid = [r for r in rho_grid if 1 <= r <= rho_max]
        log("info", f"rho-truncation sweep: {rho_grid}")
        rank_results = run_rank_truncation(
            model, val_inp, val_tgt, args.batch_size, args.device, rho_grid,
        )

        results_by_ckpt[tag] = {
            "ckpt": p,
            "tucker_r": cfg["tucker_r"],
            "tucker_s": cfg["tucker_s"],
            "trained_perplexity": ppl0,
            "diagonal_projection": diag_results,
            "rank_truncation": rank_results,
        }
        with open(os.path.join(args.results_dir, "results.json"), "w") as f:
            json.dump(results_by_ckpt, f, indent=2)

    plot_results(results_by_ckpt, args.results_dir)


if __name__ == "__main__":
    main()
