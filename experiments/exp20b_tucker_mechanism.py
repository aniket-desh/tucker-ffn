#!/usr/bin/env python3
"""experiment 20b: what circuit does the Tucker-FFN model use for induction?

exp20 found tucker models reach 100% second-half accuracy with a much lower
canonical induction score (attention at offset T/2-1 from second-half queries).
This probe retrains the same models and measures:

  1. attention-offset profiles: mean attention mass from second-half query
     positions to each relative offset, per layer/head.
  2. FFN bypass at inference: accuracy with all FFN outputs zeroed (residual
     stream passthrough) — does the model need its FFN for the task?
  3. best-head ablation: accuracy with the highest-induction-score layer-2 head
     zeroed.

kinds: tucker (3 seeds) + swiglu (1 seed, control).
outputs: results/exp20b/exp20b_results.json + attention profile figure.
"""

import argparse
import json
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import log, setup_plot_style, PALETTE  # noqa: E402
from exp20_induction import TinyLM, make_batch  # noqa: E402


@torch.no_grad()
def offset_profile(model, inp, half_len):
    """mean attention mass at each offset, from second-half queries.
    returns (n_layers, n_heads, max_offset+1)."""
    _, attns, _ = model(inp, need_attn=True)
    t = inp.shape[1]
    out = []
    for att in attns:                       # (b, h, t, t)
        prof = torch.zeros(att.shape[1], t, device=att.device)
        for q in range(half_len, t):
            # attention to key j = offset q - j
            a = att[:, :, q, :q + 1].mean(0)             # (h, q+1)
            offs = q - torch.arange(q + 1, device=att.device)
            prof[:, offs] += a
        prof /= (t - half_len)
        out.append(prof)
    return torch.stack(out)                  # (L, H, t)


@torch.no_grad()
def acc_with_ablations(model, inp, tgt, half_len):
    logits, attns, _ = model(inp, need_attn=True)
    pred = logits.argmax(-1)
    base = (pred[:, half_len - 1:] == tgt[:, half_len - 1:]).float().mean().item()

    # ffn bypass
    if model.has_ffn:
        model.has_ffn = False
        logits2, _, _ = model(inp)
        model.has_ffn = True
        pred2 = logits2.argmax(-1)
        no_ffn = (pred2[:, half_len - 1:] == tgt[:, half_len - 1:]).float().mean().item()
    else:
        no_ffn = base

    # best layer-2 induction head ablation: zero its value-output contribution
    att = attns[-1]
    t = att.shape[-1]
    qpos = torch.arange(half_len, t, device=att.device)
    kpos = qpos - (half_len - 1)
    scores = att[:, :, qpos, kpos].mean(dim=(0, 2))      # (h,)
    best_h = scores.argmax().item()
    lyr = model.layers[-1]["attn"]
    hd = lyr.hd
    W = lyr.out.weight.data.clone()
    lyr.out.weight.data[:, best_h * hd:(best_h + 1) * hd] = 0
    logits3, _, _ = model(inp)
    lyr.out.weight.data.copy_(W)
    pred3 = logits3.argmax(-1)
    no_head = (pred3[:, half_len - 1:] == tgt[:, half_len - 1:]).float().mean().item()
    return base, no_ffn, no_head, best_h, scores.max().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=128)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--half_len", type=int, default=64)
    ap.add_argument("--ffn_budget", type=int, default=131072)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--runs", type=str,
                    default="tucker:0,tucker:1,tucker:2,swiglu:0")
    ap.add_argument("--results_dir", type=str, default="results/exp20b")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    results = []
    profiles = {}
    for spec in args.runs.split(","):
        kind, seed = spec.split(":")
        seed = int(seed)
        torch.manual_seed(seed)
        model = TinyLM(args.vocab, args.d, args.n_heads, 2, kind,
                       args.ffn_budget).to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                betas=(0.9, 0.95), weight_decay=0.01)
        for step in range(args.steps):
            inp, tgt = make_batch(args.bs, args.half_len, args.vocab,
                                  args.device)
            logits, _, _ = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab),
                                   tgt.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        gen = torch.Generator(device=args.device).manual_seed(99)
        inp, tgt = make_batch(args.eval_bs, args.half_len, args.vocab,
                              args.device, gen)
        prof = offset_profile(model, inp, args.half_len)
        base, no_ffn, no_head, best_h, best_score = acc_with_ablations(
            model, inp, tgt, args.half_len)
        profiles[f"{kind}_s{seed}"] = prof.cpu().numpy()
        rec = {"kind": kind, "seed": seed, "acc": base,
               "acc_no_ffn": no_ffn, "acc_no_best_head": no_head,
               "best_head": best_h, "best_head_induction_score": best_score,
               "l2_offset_top3": [
                   {"head": h,
                    "top_offsets": prof[1, h].argsort(descending=True)[:3].tolist(),
                    "top_mass": prof[1, h].sort(descending=True).values[:3].tolist()}
                   for h in range(prof.shape[1])]}
        results.append(rec)
        log("result", f"{kind} s{seed}: acc={base:.3f} no_ffn={no_ffn:.3f} "
            f"no_best_head={no_head:.3f} best_h={best_h} "
            f"ind={best_score:.3f}")
        with open(os.path.join(args.results_dir, "exp20b_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    np.savez(os.path.join(args.results_dir, "offset_profiles.npz"), **profiles)

    # figure: layer-2 offset profiles, tucker vs swiglu
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, key, title in [(axes[0], "swiglu_s0", "SwiGLU seed 0"),
                           (axes[1], "tucker_s0", "Tucker seed 0")]:
        if key not in profiles:
            continue
        prof = profiles[key]
        for h in range(prof.shape[1]):
            ax.plot(prof[1, h], label=f"L2 head {h}", lw=1.0)
        ax.axvline(args.half_len - 1, color="k", ls=":", lw=0.8,
                   label=f"offset {args.half_len-1} (induction)")
        ax.axvline(args.half_len, color="gray", ls="--", lw=0.8,
                   label=f"offset {args.half_len} (prev. occurrence)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("relative offset (query - key)")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("mean attention mass (2nd-half queries)")
    plt.tight_layout()
    out = os.path.join(args.results_dir, "exp20b_offsets.png")
    plt.savefig(out, dpi=180)
    plt.close()
    log("done", f"saved {out}")


if __name__ == "__main__":
    main()
