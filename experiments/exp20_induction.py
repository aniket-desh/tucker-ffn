#!/usr/bin/env python3
"""experiment 20: induction-head pilot across FFN tensor structures.

Task: repeated random sequences. Each sample is [s ; s] with s drawn uniformly
from a vocab of size V (length T/2 each). Predicting the second half requires
in-context copying (induction); every sample uses fresh random tokens, so the
task cannot be memorized — a 2-layer attention circuit (previous-token head ->
induction head) is the canonical solution (Olsson et al. 2022).

Models: 2-layer transformers, identical attention (d=128, 4 heads, rope),
differing only in FFN: none (attention-only control), swiglu, ll1_l4, tucker
(all FFNs at matched parameter budget).

Measured every eval_every steps on a fixed eval batch:
  - second-half next-token accuracy (induction accuracy)
  - induction score: mean layer-2 attention mass at offset T/2-1 (from query
    position i in the second half to the token *after* the previous occurrence
    of the current token, which for [s;s] is position i - (T/2 - 1))
  - FFN routing contrast: effective active routed units (exp-entropy) on
    second-half vs first-half positions.

outputs (under --results_dir): exp20_results.json, exp20_curves.png
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import log, setup_plot_style, PALETTE, COLOR_CYCLE  # noqa: E402
from lib.ll1_ffn import LL1FFN, ll1_blocks_for_params  # noqa: E402
from lib.tucker_ffn import SwiGLUFFN, TuckerFFN  # noqa: E402
from lib.lm import RMSNorm, _rope_cache, _apply_rope  # noqa: E402


class Attn(nn.Module):
    """causal attention that can return attention probabilities."""

    def __init__(self, d, n_heads):
        super().__init__()
        self.d, self.n_heads, self.hd = d, n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin, need_attn=False):
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.hd)
        q, k, v = (z.transpose(1, 2) for z in qkv.unbind(dim=2))
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if need_attn:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            mask = torch.full((t, t), float("-inf"), device=x.device).triu(1)
            att = (att + mask).softmax(dim=-1)              # (b, h, t, t)
            o = att @ v
        else:
            att = None
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(b, t, self.d)
        return self.out(o), att


def build_ffn(kind, d, budget):
    if kind == "none":
        return None
    if kind == "swiglu":
        m = max(1, round(budget / (3 * d)))
        return SwiGLUFFN(d, m)
    if kind.startswith("ll1_l"):
        L = int(kind[len("ll1_l"):])
        return LL1FFN(d, n_blocks=ll1_blocks_for_params(d, L, budget),
                      block_rank=L)
    if kind == "tucker":
        r = 1
        while d * 3 * (r + 1) + (r + 1) ** 3 <= budget:
            r += 1
        return TuckerFFN(d, r=r, s=r)
    raise ValueError(kind)


class TinyLM(nn.Module):
    def __init__(self, vocab, d, n_heads, n_layers, ffn_kind, ffn_budget):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "norm1": RMSNorm(d), "attn": Attn(d, n_heads),
                "norm2": RMSNorm(d),
                "ffn": build_ffn(ffn_kind, d, ffn_budget) or nn.Identity(),
            }))
        self.has_ffn = ffn_kind != "none"
        self.norm_f = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        nn.init.normal_(self.embed.weight, std=1.0 / math.sqrt(d))
        nn.init.normal_(self.head.weight, std=1.0 / math.sqrt(d))
        self.d, self.n_heads = d, n_heads

    def forward(self, idx, need_attn=False, capture_ffn_in=False):
        b, t = idx.shape
        cos, sin = _rope_cache(t, self.d // self.n_heads, idx.device)
        x = self.embed(idx)
        attns, ffn_ins = [], []
        for lyr in self.layers:
            a, att = lyr["attn"](lyr["norm1"](x), cos, sin, need_attn)
            x = x + a
            attns.append(att)
            h = lyr["norm2"](x)
            if capture_ffn_in:
                ffn_ins.append(h.detach())
            if self.has_ffn:
                x = x + lyr["ffn"](h)
        return self.head(self.norm_f(x)), attns, ffn_ins


def make_batch(bs, half_len, vocab, device, gen=None):
    s = torch.randint(0, vocab, (bs, half_len), device=device, generator=gen)
    seq = torch.cat([s, s], dim=1)
    return seq[:, :-1], seq[:, 1:]


@torch.no_grad()
def evaluate(model, eval_inp, eval_tgt, half_len):
    logits, attns, ffn_ins = model(eval_inp, need_attn=True,
                                   capture_ffn_in=True)
    pred = logits.argmax(dim=-1)
    # positions half_len-1 .. end of inp predict the second half
    acc = (pred[:, half_len - 1:] == eval_tgt[:, half_len - 1:]).float().mean()
    # induction score: layer-2 attention from query positions i >= half_len
    # to key position i - (half_len - 1), per head; report max over heads
    att = attns[-1]                                   # (b, h, t, t)
    t = att.shape[-1]
    qpos = torch.arange(half_len, t, device=att.device)
    kpos = qpos - (half_len - 1)
    ind = att[:, :, qpos, kpos].mean(dim=(0, 2))      # (h,)
    # ffn routing contrast (layer 2): eff active units, 2nd vs 1st half
    contrast = None
    ffn = model.layers[-1]["ffn"]
    if not isinstance(ffn, nn.Identity):
        h = ffn_ins[-1]                                # (b, t, d)
        first = h[:, :half_len - 1].reshape(-1, h.shape[-1])
        second = h[:, half_len - 1:].reshape(-1, h.shape[-1])
        from exp19_interp_proxies import unit_contributions, eff_count
        c1, _ = unit_contributions(ffn, first.float())
        c2, _ = unit_contributions(ffn, second.float())
        contrast = (eff_count(c2), eff_count(c1))
    return acc.item(), ind.max().item(), contrast


def train_one(kind, seed, args, device):
    torch.manual_seed(seed)
    model = TinyLM(args.vocab, args.d, args.n_heads, 2, kind,
                   args.ffn_budget).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    gen_eval = torch.Generator(device=device).manual_seed(99)
    eval_inp, eval_tgt = make_batch(args.eval_bs, args.half_len, args.vocab,
                                    device, gen_eval)
    hist = []
    t0 = time.time()
    for step in range(args.steps):
        inp, tgt = make_batch(args.bs, args.half_len, args.vocab, device)
        logits, _, _ = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, args.vocab), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % args.eval_every == 0 or step == 0:
            acc, ind, contrast = evaluate(model, eval_inp, eval_tgt,
                                          args.half_len)
            hist.append({"step": step + 1, "loss": loss.item(),
                         "acc2nd": acc, "induction_score": ind,
                         "eff_active_2nd_vs_1st": contrast})
    acc, ind, contrast = evaluate(model, eval_inp, eval_tgt, args.half_len)
    log("result", f"{kind} seed={seed}: acc={acc:.3f} ind={ind:.3f} "
        f"params={n_params/1e3:.0f}K time={time.time()-t0:.0f}s "
        f"contrast={contrast}")
    return {"kind": kind, "seed": seed, "n_params": n_params,
            "final_acc": acc, "final_induction": ind,
            "final_contrast": contrast, "history": hist}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--half_len", type=int, default=32)
    ap.add_argument("--ffn_budget", type=int, default=131072)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kinds", type=str,
                    default="none,swiglu,ll1_l4,tucker")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--results_dir", type=str, default="results/exp20")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    res_path = os.path.join(args.results_dir, "exp20_results.json")

    if not args.plot_only:
        results = []
        for kind in args.kinds.split(","):
            for seed in [int(s) for s in args.seeds.split(",")]:
                results.append(train_one(kind, seed, args, args.device))
                with open(res_path, "w") as f:
                    json.dump(results, f, indent=2)

    with open(res_path) as f:
        results = json.load(f)

    setup_plot_style()
    kinds = args.kinds.split(",")
    colors = {"none": "gray", "swiglu": PALETTE["ablation"],
              "ll1_l4": PALETTE["primary"], "tucker": PALETTE["accent"]}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for kind in kinds:
        rows = [r for r in results if r["kind"] == kind]
        if not rows:
            continue
        steps = [h["step"] for h in rows[0]["history"]]
        for mi, metric in enumerate(["acc2nd", "induction_score"]):
            curves = np.array([[h[metric] for h in r["history"]]
                               for r in rows])
            m = curves.mean(0); sd = curves.std(0)
            axes[mi].plot(steps, m, color=colors.get(kind, "k"), label=kind)
            axes[mi].fill_between(steps, m - sd, m + sd,
                                  color=colors.get(kind, "k"), alpha=0.15)
    axes[0].set_ylabel("second-half accuracy")
    axes[1].set_ylabel("induction score (best head, L2)")
    for ax in axes:
        ax.set_xlabel("training step")
        ax.legend(fontsize=7)
    plt.tight_layout()
    out = os.path.join(args.results_dir, "exp20_curves.png")
    plt.savefig(out, dpi=180); plt.close()
    log("done", f"saved {out}")


if __name__ == "__main__":
    main()
