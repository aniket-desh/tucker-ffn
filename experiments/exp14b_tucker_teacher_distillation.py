#!/usr/bin/env python3
"""experiment 14b: distillation with a trained TUCKER teacher.

complementary to exp14 (which used a pretrained swiglu teacher and so put
swiglu students at a structural advantage). here the teacher is one ffn
layer of a trained tucker lm (from exp11). because the teacher's outputs
genuinely use cross-channel interactions that tucker can express but
swiglu can only approximate, this isolates the expressivity gap predicted
by theorem 1 from the optimization gap of full lm training.

we collect inputs to the chosen tucker layer by running the trained tucker
lm on streamed text and capturing the residual stream entering it. we then
fit two student architectures to the layer's outputs at matched parameter
budgets, sweeping the budget.

outputs (under --results_dir):
  tucker_teacher_distillation.json
  tucker_teacher_distillation.png
"""

import argparse
import glob
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

from lib import PALETTE, SwiGLUFFN, TuckerFFN, log, setup_plot_style  # noqa: E402
from lib.lm import make_lm  # noqa: E402


def load_tucker_lm(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = sd["cfg"]
    model = make_lm("tucker", d=cfg["d"], n_heads=cfg["n_heads"],
                     n_layers=cfg["n_layers"], vocab_size=cfg["vocab_size"],
                     max_seq_len=cfg["seq_len"], r=cfg["tucker_r"],
                     s=cfg["tucker_s"])
    model.load_state_dict(sd["model_state_dict"])
    return model, cfg


@torch.no_grad()
def collect_activations(model, layer_idx, tokenizer, device, n_tokens,
                         seq_len=512, batch_seqs=4):
    """capture (mlp_in, mlp_out) at one tucker layer over fineweb-edu text."""
    from datasets import load_dataset
    target = model.blocks[layer_idx].ffn

    captured_in, captured_out = [], []

    def hook(module, inp, out):
        x = inp[0].detach()
        if isinstance(out, tuple):
            y = out[0].detach()
        else:
            y = out.detach()
        captured_in.append(x.float().reshape(-1, x.shape[-1]).cpu())
        captured_out.append(y.float().reshape(-1, y.shape[-1]).cpu())

    h = target.register_forward_hook(hook)

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                       split="train", streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=10_000)
    eos = tokenizer.eos_token_id or 0
    buf = []
    n_collected = 0
    try:
        for ex in ds:
            text = ex.get("text", "")
            if not text:
                continue
            ids = tokenizer.encode(text)
            ids.append(eos)
            buf.extend(ids)
            while len(buf) >= seq_len * batch_seqs:
                chunk = torch.tensor(buf[:seq_len * batch_seqs],
                                      dtype=torch.long)
                buf = buf[seq_len * batch_seqs:]
                chunk = chunk.view(batch_seqs, seq_len).to(device)
                model(chunk)
                n_collected += seq_len * batch_seqs
                if n_collected >= n_tokens:
                    break
            if n_collected >= n_tokens:
                break
    finally:
        h.remove()

    X = torch.cat(captured_in, dim=0)[:n_tokens]
    Y = torch.cat(captured_out, dim=0)[:n_tokens]
    return X, Y


def fit_student(student, X_train, Y_train, X_val, Y_val, n_steps, lr,
                weight_decay, batch_size, device, warmup_frac=0.05,
                grad_clip=1.0):
    student.to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr,
                            weight_decay=weight_decay)
    n = X_train.shape[0]
    perm = torch.randperm(n, device=device)
    cursor = 0
    warmup_steps = max(1, int(warmup_frac * n_steps))
    for step in range(n_steps):
        if step < warmup_steps:
            cur_lr = lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, n_steps - warmup_steps - 1)
            cur_lr = lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        if cursor + batch_size > n:
            perm = torch.randperm(n, device=device)
            cursor = 0
        idx = perm[cursor:cursor + batch_size]
        cursor += batch_size
        loss = F.mse_loss(student(X_train[idx]), Y_train[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        opt.step()
    student.eval()
    with torch.no_grad():
        v = F.mse_loss(student(X_val), Y_val).item()
    var_y = Y_val.var().item() + 1e-12
    return {"val_mse": v, "fvu": v / var_y}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str,
                        default="results/exp11/tucker_seed0/checkpoint_final.pt")
    parser.add_argument("--teacher_layer", type=int, default=4)
    parser.add_argument("--n_train", type=int, default=80000)
    parser.add_argument("--n_val", type=int, default=8000)
    parser.add_argument("--n_steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--results_dir", type=str, default="results/exp14b")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    setup_plot_style()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.ckpt):
        log("error", f"checkpoint not found: {args.ckpt}")
        return
    log("info", f"loading tucker teacher from {args.ckpt}")
    model, cfg = load_tucker_lm(args.ckpt)
    model.to(device)
    model.eval()
    d = cfg["d"]
    r_teacher = cfg["tucker_r"]
    s_teacher = cfg["tucker_s"]
    log("info", f"teacher d={d} | r=s={r_teacher} | layer={args.teacher_layer}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")

    log("info", f"collecting {args.n_train + args.n_val} activations")
    t0 = time.time()
    X, Y = collect_activations(
        model, args.teacher_layer, tok, device,
        args.n_train + args.n_val, seq_len=512, batch_seqs=4,
    )
    log("info", f"collected X={X.shape} Y={Y.shape} | time={time.time()-t0:.1f}s")
    del model
    torch.cuda.empty_cache()

    X_train, X_val = X[:args.n_train].to(device), X[args.n_train:].to(device)
    Y_train, Y_val = Y[:args.n_train].to(device), Y[args.n_train:].to(device)
    log("data", f"y_var={Y_val.var().item():.3e}")

    # sweep budgets that bracket the teacher (smaller and equal)
    r_grid = [max(4, r_teacher // 4), r_teacher // 2, r_teacher,
               int(r_teacher * 1.5)]
    sweep = []
    for r in r_grid:
        # tucker(d, r, r): params = d*(3r) + r^3
        tp = d * 3 * r + r ** 3
        # match: 3*d*m = tp => m = tp / (3d)
        m = max(1, int(round(tp / (3 * d))))
        sp = 3 * d * m
        sweep.append({"r": r, "s": r, "m_swiglu": m,
                       "tucker_params": tp, "swiglu_params": sp})

    log("info", f"matched-budget sweep:")
    for s in sweep:
        log("info", f"  tucker r=s={s['r']:3d} ({s['tucker_params']:8d}) | "
            f"swiglu m={s['m_swiglu']:5d} ({s['swiglu_params']:8d})")

    results = []
    for entry in sweep:
        log("info", f"=== tucker r=s={entry['r']} / swiglu m={entry['m_swiglu']} ===")
        per_arch = {"swiglu": [], "tucker": []}
        for seed in range(args.n_seeds):
            torch.manual_seed(seed + 17 * entry["m_swiglu"])
            sw = SwiGLUFFN(d, entry["m_swiglu"], bias=False)
            r = fit_student(sw, X_train, Y_train, X_val, Y_val,
                             args.n_steps, args.lr, 0.0, args.batch_size,
                             device)
            per_arch["swiglu"].append(r)
            log("eval", f"  swiglu seed={seed} | val_mse={r['val_mse']:.3e} | fvu={r['fvu']:.3e}")
            torch.manual_seed(seed + 31 * entry["r"])
            tk = TuckerFFN(d, r=entry["r"], s=entry["s"])
            r2 = fit_student(tk, X_train, Y_train, X_val, Y_val,
                              args.n_steps, args.lr, 0.0, args.batch_size,
                              device)
            per_arch["tucker"].append(r2)
            log("eval", f"  tucker seed={seed} | val_mse={r2['val_mse']:.3e} | fvu={r2['fvu']:.3e}")
        results.append({**entry, "results": per_arch})

    out_path = os.path.join(args.results_dir, "tucker_teacher_distillation.json")
    with open(out_path, "w") as f:
        json.dump({"teacher_ckpt": args.ckpt, "teacher_layer": args.teacher_layer,
                    "d": d, "r_teacher": r_teacher, "sweep": results}, f, indent=2)
    log("done", f"saved tucker_teacher_distillation.json -> {args.results_dir}/")

    # plot
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [e["tucker_params"] for e in results]
    sw_means = [np.mean([r["val_mse"] for r in e["results"]["swiglu"]]) for e in results]
    sw_stds = [np.std([r["val_mse"] for r in e["results"]["swiglu"]]) for e in results]
    tk_means = [np.mean([r["val_mse"] for r in e["results"]["tucker"]]) for e in results]
    tk_stds = [np.std([r["val_mse"] for r in e["results"]["tucker"]]) for e in results]
    ax.errorbar(xs, sw_means, yerr=sw_stds, marker="o", color=PALETTE["primary"],
                lw=1.5, label="SwiGLU (student)", capsize=3)
    ax.errorbar(xs, tk_means, yerr=tk_stds, marker="s", color=PALETTE["ablation"],
                lw=1.5, label="Tucker (student)", capsize=3)
    ax.set_xlabel("FFN parameter budget")
    ax.set_ylabel("Distillation val MSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(framealpha=0.9, edgecolor="0.8")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir,
                              "tucker_teacher_distillation.pdf"),
                 bbox_inches="tight", pad_inches=0.02)
    plt.savefig(os.path.join(args.results_dir,
                              "tucker_teacher_distillation.png"),
                 bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close()
    log("done", f"saved tucker_teacher_distillation.png -> {args.results_dir}/")


if __name__ == "__main__":
    main()
