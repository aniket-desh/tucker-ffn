#!/usr/bin/env python3
"""experiment 14: ffn distillation gap on real residual-stream activations.

instead of comparing swiglu vs tucker via end-to-end lm training (exp11),
isolate representational capacity by distilling a single trained-swiglu ffn
layer from a pretrained model into student ffns of matched parameter
budgets.

teacher: a swiglu ffn picked from a pretrained llama/qwen model (mid-depth
layer where routing variance is non-trivial). inputs x are sampled from
the residual stream entering that layer on real text (fineweb-edu); outputs
y are the teacher ffn's outputs at those x. this gives a realistic
activation distribution rather than the synthetic gaussians of exp10.

students:
  swiglu (m')      same architecture as teacher, possibly smaller width
  tucker (r,s)     tucker-core ffn at matched parameter budget
  swiglu (m_t)     control: matched-width swiglu == teacher (should
                  recover near-zero loss on enough data)

we sweep matched-budget pairs (m', (r, s)) with 3*d*m' ~= d*(2r+s)+s*r^2,
fit each student to the teacher by mse, report val mse. tucker should
match teacher more accurately at the same parameter budget if its richer
interaction structure is useful for representing realistic activations.

outputs (under --results_dir):
  distillation.json
  distillation.png         — val mse vs param budget, swiglu vs tucker
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
    PALETTE,
    SwiGLUFFN,
    TuckerFFN,
    detect_device,
    get_swiglu_layers,
    load_model_and_tokenizer,
    log,
    setup_plot_style,
)


@torch.no_grad()
def collect_activations(model, layers_info, layer_idx, tokenizer, device,
                         n_tokens, batch_seqs=4, seq_len=512):
    """run pretrained model on streamed text, capture (mlp_in, mlp_out) at the
    chosen layer until we have n_tokens (token, dim) samples."""
    from datasets import load_dataset
    info = layers_info[layer_idx]
    mlp = info["mlp"]

    captured_in, captured_out = [], []

    def hook(module, inp, out):
        x = inp[0].detach()
        if isinstance(out, tuple):
            y = out[0].detach()
        else:
            y = out.detach()
        captured_in.append(x.float().reshape(-1, x.shape[-1]).cpu())
        captured_out.append(y.float().reshape(-1, y.shape[-1]).cpu())

    h = mlp.register_forward_hook(hook)

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                       split="train", streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=10_000)

    buf = []
    eos = tokenizer.eos_token_id or 0
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
                model(chunk)  # populates hook
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
    """fit student to teacher outputs with linear warmup + cosine lr decay
    and gradient clipping. resilient to high-budget swiglu instability and
    tucker's noisy core-grad scale.

    when student is a TuckerFFN we put the core C on its own param group at
    lr_core = lr (same), but we keep the option to scale it separately if
    instability emerges. this is purely an optimization tweak; the model
    architecture and parameterization (note section IV) are unchanged.
    """
    import math as _math
    student.to(device)
    # split params: tucker C gets its own group; everything else default
    core_params = [p for n, p in student.named_parameters()
                    if n in ("C", "c_diag")]
    other_params = [p for n, p in student.named_parameters()
                     if n not in ("C", "c_diag")]
    if core_params:
        opt = torch.optim.Adam(
            [{"params": other_params, "lr": lr},
             {"params": core_params,  "lr": lr * 1.0}],
            weight_decay=weight_decay,
        )
    else:
        opt = torch.optim.Adam(student.parameters(), lr=lr,
                                weight_decay=weight_decay)
    n = X_train.shape[0]
    perm = torch.randperm(n, device=device)
    cursor = 0
    warmup_steps = max(1, int(warmup_frac * n_steps))
    for step in range(n_steps):
        # linear warmup then cosine decay to 1% of peak
        if step < warmup_steps:
            cur_lr = lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, n_steps - warmup_steps - 1)
            cur_lr = lr * (0.01 + 0.99 * 0.5 * (1 + _math.cos(_math.pi * progress)))
        # respect the per-group base lr scale if any (here both are == lr)
        for pg in opt.param_groups:
            base_scale = pg.get("base_scale", 1.0)
            pg["lr"] = cur_lr * base_scale
        if cursor + batch_size > n:
            perm = torch.randperm(n, device=device)
            cursor = 0
        idx = perm[cursor:cursor + batch_size]
        cursor += batch_size
        xb = X_train[idx]
        yb = Y_train[idx]
        pred = student(xb)
        loss = F.mse_loss(pred, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        opt.step()
    student.eval()
    with torch.no_grad():
        v = F.mse_loss(student(X_val), Y_val).item()
        t = F.mse_loss(student(X_train[:X_val.shape[0]]),
                        Y_train[:X_val.shape[0]]).item()
    var_y = Y_val.var().item() + 1e-12
    return {"val_mse": v, "train_mse": t, "fvu": v / var_y}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--teacher_layer", type=int, default=12)
    parser.add_argument("--n_train", type=int, default=80000)
    parser.add_argument("--n_val", type=int, default=10000)
    parser.add_argument("--n_steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--results_dir", type=str, default="results/exp14")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    setup_plot_style()
    device = args.device or detect_device()

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    layers_info = get_swiglu_layers(model)
    info = layers_info[args.teacher_layer]
    d = info["gate_proj"].weight.shape[1]
    m_teacher = info["gate_proj"].weight.shape[0]
    log("info", f"teacher d={d} | m_teacher={m_teacher}")

    # collect activations
    log("info", f"collecting {args.n_train + args.n_val} activations from "
        f"layer {args.teacher_layer}")
    t0 = time.time()
    X, Y = collect_activations(
        model, layers_info, args.teacher_layer, tokenizer, device,
        args.n_train + args.n_val, batch_seqs=4, seq_len=512,
    )
    log("info", f"collected X={X.shape} Y={Y.shape} | time={time.time()-t0:.1f}s")
    # free big model
    del model
    torch.cuda.empty_cache()

    X_train, X_val = X[:args.n_train].to(device), X[args.n_train:].to(device)
    Y_train, Y_val = Y[:args.n_train].to(device), Y[args.n_train:].to(device)
    log("data", f"y_var={Y_val.var().item():.3e}")

    # decide on a sweep of matched-param budgets:
    # use widths m' such that m' < m_teacher to make the comparison interesting
    m_grid = []
    for frac in [0.05, 0.1, 0.2, 0.4, 0.8]:
        m_grid.append(max(1, int(round(m_teacher * frac))))

    sweep = []
    for m_p in m_grid:
        # find tucker (r=s) such that d(2r+s)+s*r^2 ~= 3*d*m'
        target = 3 * d * m_p
        # solve r^3 + 3 d r ~ target  =>  r ~ target^{1/3}
        r = max(2, int(round((target) ** (1.0 / 3.0))))
        # adjust r to bracket target
        best_r = r
        best_err = abs(d * (2 * r + r) + r * r * r - target)
        for cand in range(max(2, r - 6), r + 6):
            err = abs(d * (3 * cand) + cand ** 3 - target)
            if err < best_err:
                best_err = err
                best_r = cand
        r = best_r
        sweep.append({"m_swiglu": m_p, "r": r, "s": r,
                       "swiglu_params": 3 * d * m_p,
                       "tucker_params": d * (3 * r) + r ** 3})

    log("info", f"matched-budget sweep:")
    for s in sweep:
        log("info", f"  m={s['m_swiglu']:5d} ({s['swiglu_params']:8d}) | "
            f"tucker r=s={s['r']:3d} ({s['tucker_params']:8d}) | "
            f"mismatch={(s['tucker_params']-s['swiglu_params'])/s['swiglu_params']*100:+.1f}%")

    results = []
    for entry in sweep:
        log("info", f"=== budget m={entry['m_swiglu']} / r=s={entry['r']} ===")
        per_arch = {"swiglu": [], "tucker": []}
        for seed in range(args.n_seeds):
            torch.manual_seed(seed + 17 * entry["m_swiglu"])
            sw = SwiGLUFFN(d, entry["m_swiglu"], bias=False)
            r = fit_student(sw, X_train, Y_train, X_val, Y_val,
                             args.n_steps, args.lr, args.weight_decay,
                             args.batch_size, device)
            per_arch["swiglu"].append(r)
            log("eval", f"  swiglu seed={seed} | val_mse={r['val_mse']:.3e} | fvu={r['fvu']:.3e}")

            torch.manual_seed(seed + 31 * entry["r"])
            tk = TuckerFFN(d, r=entry["r"], s=entry["s"])
            r2 = fit_student(tk, X_train, Y_train, X_val, Y_val,
                              args.n_steps, args.lr, args.weight_decay,
                              args.batch_size, device)
            per_arch["tucker"].append(r2)
            log("eval", f"  tucker seed={seed} | val_mse={r2['val_mse']:.3e} | fvu={r2['fvu']:.3e}")
        results.append({**entry, "results": per_arch})

    out_path = os.path.join(args.results_dir, "distillation.json")
    with open(out_path, "w") as f:
        json.dump({"teacher_layer": args.teacher_layer,
                    "model": args.model, "d": d, "m_teacher": m_teacher,
                    "sweep": results}, f, indent=2)
    log("done", f"saved distillation.json -> {args.results_dir}/")

    # plot
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [e["swiglu_params"] for e in results]
    sw_means = [np.mean([r["val_mse"] for r in e["results"]["swiglu"]]) for e in results]
    sw_stds = [np.std([r["val_mse"] for r in e["results"]["swiglu"]]) for e in results]
    tk_means = [np.mean([r["val_mse"] for r in e["results"]["tucker"]]) for e in results]
    tk_stds = [np.std([r["val_mse"] for r in e["results"]["tucker"]]) for e in results]
    ax.errorbar(xs, sw_means, yerr=sw_stds, marker="o", color=PALETTE["primary"],
                lw=1.5, label="swiglu (student)", capsize=3)
    ax.errorbar(xs, tk_means, yerr=tk_stds, marker="s", color=PALETTE["ablation"],
                lw=1.5, label="tucker (student, matched params)", capsize=3)
    ax.set_xlabel("ffn parameter budget")
    ax.set_ylabel(f"distillation val MSE (teacher = {args.model} layer {args.teacher_layer})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(framealpha=0.9, edgecolor="0.8")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "distillation.png"))
    plt.close()
    log("done", f"saved distillation.png -> {args.results_dir}/")


if __name__ == "__main__":
    main()
