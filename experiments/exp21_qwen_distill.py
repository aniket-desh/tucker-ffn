#!/usr/bin/env python3
"""experiment 21: distill pretrained SwiGLU FFN layers into CP/LL1/Tucker students.

Question: do *real, pretrained* transformer FFN input-output maps prefer rank-1
routed atoms (CP), low-rank routed blocks (LL1), or a dense latent core (Tucker)
when compressed to a fixed parameter budget?

Setup: capture residual-stream inputs x and FFN outputs y for selected layers
of Qwen2.5-0.5B (d=896, m=4864, ~13.1M params/FFN) on FineWeb-Edu text; fit
students at compression budgets by MSE; report relative val MSE and cosine sim.

This complements exp18 (synthetic, structure known) and exp11 (from-scratch
training): it asks which tensor structure best matches the function a real
trained FFN computes, independent of trainability-from-scratch effects.

outputs (under --results_dir): exp21_results.json, exp21_distill.png
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

from lib import log, setup_plot_style, PALETTE  # noqa: E402
from lib.model_utils import load_model_and_tokenizer, get_swiglu_layers  # noqa: E402
from exp18_ll1_synthetic import build_student, fit_student  # noqa: E402


def capture_layer_io(model_name, layers, n_tokens, seq_len, device,
                     batch_seqs=8):
    """stream fineweb-edu through the model, capture FFN (input, output) pairs
    for the requested layer indices. returns {layer: (X, Y)} on cpu."""
    from datasets import load_dataset
    model, tok = load_model_and_tokenizer(model_name, device)
    infos = get_swiglu_layers(model)
    sel = {i["layer_idx"]: i for i in infos if i["layer_idx"] in layers}
    store = {li: ([], []) for li in layers}
    hooks = []
    for li, info in sel.items():
        def mk(li):
            def hook(mod, inp, out):
                store[li][0].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu())
                o = out[0] if isinstance(out, tuple) else out
                store[li][1].append(o.detach().reshape(-1, o.shape[-1]).cpu())
            return hook
        hooks.append(sel[li]["mlp"].register_forward_hook(mk(li)))

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True).shuffle(seed=3, buffer_size=10_000)
    buf, seen = [], 0
    batch = []
    for ex in ds:
        ids = tok.encode(ex.get("text", ""))
        buf.extend(ids)
        while len(buf) >= seq_len:
            batch.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(batch) == batch_seqs:
                with torch.no_grad():
                    model(torch.tensor(batch, device=device))
                seen += batch_seqs * seq_len
                batch = []
        if seen >= n_tokens:
            break
    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return {li: (torch.cat(xs), torch.cat(ys)) for li, (xs, ys) in store.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", type=str, default="4,12,20")
    ap.add_argument("--n_tokens", type=int, default=120_000)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--budgets", type=str, default="600000,1200000,2400000")
    ap.add_argument("--archs", type=str,
                    default="swiglu,ll1_l1,ll1_l2,ll1_l4,ll1_l8,ll1_l16,tucker")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n_seeds", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--results_dir", type=str, default="results/exp21")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    res_path = os.path.join(args.results_dir, "exp21_results.json")
    layers = [int(s) for s in args.layers.split(",")]

    if not args.plot_only:
        io = capture_layer_io(args.model, layers, args.n_tokens, args.seq_len,
                              args.device)
        results = []
        for li in layers:
            X, Y = io[li]
            d = X.shape[1]
            # standardize: zero-mean inputs help optimization; outputs scaled
            # to unit std so rel-MSE is comparable across layers
            n_val = int(len(X) * args.val_frac)
            perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
            X, Y = X[perm].float(), Y[perm].float()
            y_scale = Y[n_val:].std()
            Y = Y / y_scale
            x_tr, y_tr = X[n_val:].to(args.device), Y[n_val:].to(args.device)
            x_va, y_va = X[:n_val].to(args.device), Y[:n_val].to(args.device)
            var_y = y_va.var().item()
            log("info", f"layer {li}: d={d} train={len(x_tr)} val={len(x_va)} "
                f"var_y={var_y:.3f}")
            for budget in [int(b) for b in args.budgets.split(",")]:
                for arch in args.archs.split(","):
                    mses, coss = [], []
                    meta = None
                    for seed in range(args.n_seeds):
                        student, n_params, meta = build_student(
                            arch, d, budget, seed=seed * 31 + 5)
                        student = student.to(args.device)
                        t0 = time.time()
                        mse = fit_student(student, x_tr, y_tr, x_va, y_va,
                                          args.steps, args.lr, args.batch,
                                          args.device)
                        with torch.no_grad():
                            pred = student(x_va)
                            cos = F.cosine_similarity(pred, y_va, dim=-1).mean().item()
                        mses.append(mse / var_y)
                        coss.append(cos)
                        log("train", f"L{li} <- {arch} budget={budget} "
                            f"params={n_params} seed={seed} "
                            f"relMSE={mse/var_y:.3e} cos={cos:.4f} "
                            f"({time.time()-t0:.0f}s)")
                    results.append({"layer": li, "arch": arch, "budget": budget,
                                    "params": n_params, "meta": meta,
                                    "rel_mse_seeds": mses, "cos_seeds": coss})
                    with open(res_path, "w") as f:
                        json.dump(results, f, indent=2)
            del x_tr, y_tr, x_va, y_va
            torch.cuda.empty_cache()
        log("done", f"exp21 fits complete -> {res_path}")

    with open(res_path) as f:
        results = json.load(f)

    setup_plot_style()
    budgets = sorted({r["budget"] for r in results})
    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 3.6),
                             sharey=True)
    if len(layers) == 1:
        axes = [axes]
    mid_budget = budgets[len(budgets) // 2]
    for ax, li in zip(axes, layers):
        # L-sweep at mid budget
        xs, means, los, his = [], [], [], []
        for r in results:
            if r["layer"] != li or r["budget"] != mid_budget:
                continue
            if r["arch"].startswith("ll1_l"):
                xs.append(r["meta"]["L"])
                arr = np.array(r["rel_mse_seeds"])
                means.append(arr.mean()); los.append(arr.min()); his.append(arr.max())
        order = np.argsort(xs)
        xs = np.array(xs)[order]; means = np.array(means)[order]
        los = np.array(los)[order]; his = np.array(his)[order]
        ax.plot(xs, means, "o-", color=PALETTE["primary"], label="LL1 student")
        ax.fill_between(xs, los, his, color=PALETTE["primary"], alpha=0.2)
        for r in results:
            if r["layer"] != li or r["budget"] != mid_budget:
                continue
            if r["arch"] == "swiglu":
                ax.axhline(np.mean(r["rel_mse_seeds"]), color=PALETTE["ablation"],
                           ls="--", label="SwiGLU")
            if r["arch"] == "tucker":
                ax.axhline(np.mean(r["rel_mse_seeds"]), color=PALETTE["accent"],
                           ls=":", label="Tucker")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(xs); ax.set_xticklabels([str(int(x)) for x in xs])
        ax.set_xlabel("student block rank L")
        ax.set_title(f"Qwen2.5-0.5B layer {li}", fontsize=10)
        ax.legend(fontsize=7)
    axes[0].set_ylabel(f"val MSE / Var(y)  (budget {mid_budget/1e6:.1f}M)")
    plt.tight_layout()
    out = os.path.join(args.results_dir, "exp21_distill.png")
    plt.savefig(out, dpi=180); plt.close()
    log("done", f"saved {out}")


if __name__ == "__main__":
    main()
