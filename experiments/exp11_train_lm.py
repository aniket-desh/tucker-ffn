#!/usr/bin/env python3
"""experiment 11: train swiglu vs tucker language models from scratch.

trains a small llama-style transformer (~50M params) on a streamed slice of
fineweb-edu sample-10BT, comparing two ffn architectures at matched
parameter count:

  swiglu (m=1493)            standard glu ffn
  tucker (r=s=128)           tucker-core ffn (note section IV)
                             both at d*(2r+s) + s*r^2 = 3*d*m

both architectures use rmsnorm + rope + sdpa attention; only the ffn
differs. we train two seeds per architecture and log val loss every
~1M tokens.

this is the q6 head-to-head comparison: at matched params from scratch,
does tucker achieve lower loss? framing is conservative — workshop scale
is small enough that optimization may not find tucker's expressivity gain;
exp14 (distillation) is the cleaner capacity test if exp11 is mixed.

outputs (under --results_dir):
  {arch}_seed{seed}/config.json
  {arch}_seed{seed}/checkpoint_final.pt
  {arch}_seed{seed}/loss_log.json
  loss_curves.png
  loss_curves.tex
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
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import COLOR_CYCLE, PALETTE, log, setup_plot_style  # noqa: E402
from lib.lm import (  # noqa: E402
    LMConfig,
    matched_swiglu_for_tucker,
    make_lm,
)


def cosine_lr(step, max_steps, peak_lr, warmup, min_lr_frac=0.1):
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, progress)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_frac + (1 - min_lr_frac) * cos)


def stream_fineweb_edu(tokenizer, seq_len, batch_size, device,
                        config_name="sample-10BT", buffer_seqs=8,
                        seed=0, split="train"):
    """yield (input_ids, targets) batches forever from streamed fineweb-edu.

    we tokenize on the fly into a long buffer of token ids and slice it into
    seq_len chunks. simple, no packing, but adequate for small models.
    """
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", config_name,
        split=split, streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    buf = []
    for example in ds:
        text = example.get("text", "")
        if not text:
            continue
        ids = tokenizer.encode(text)
        ids.append(tokenizer.eos_token_id if tokenizer.eos_token_id is not None
                    else tokenizer.encode("\n")[0])
        buf.extend(ids)
        # emit when buffer is large enough for a batch
        while len(buf) >= (seq_len + 1) * batch_size * buffer_seqs:
            chunk = torch.tensor(buf[:(seq_len + 1) * batch_size * buffer_seqs],
                                  dtype=torch.long)
            buf = buf[(seq_len + 1) * batch_size * buffer_seqs:]
            chunk = chunk.view(buffer_seqs * batch_size, seq_len + 1)
            for i in range(0, chunk.size(0), batch_size):
                ib = chunk[i:i + batch_size]
                inp = ib[:, :-1].to(device, non_blocking=True)
                tgt = ib[:, 1:].to(device, non_blocking=True)
                yield inp, tgt


def build_val_set(tokenizer, seq_len, n_seqs, device, config_name="sample-10BT",
                  seed=12345):
    """grab a fixed validation set (deterministic shuffle, take first n_seqs)."""
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", config_name,
        split="train", streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    buf = []
    needed = (seq_len + 1) * n_seqs
    for example in ds:
        text = example.get("text", "")
        if not text:
            continue
        ids = tokenizer.encode(text)
        ids.append(tokenizer.eos_token_id if tokenizer.eos_token_id is not None
                    else tokenizer.encode("\n")[0])
        buf.extend(ids)
        if len(buf) >= needed:
            break
    chunk = torch.tensor(buf[:needed], dtype=torch.long).view(n_seqs, seq_len + 1)
    return chunk[:, :-1].to(device), chunk[:, 1:].to(device)


@torch.no_grad()
def eval_loss(model, val_inp, val_tgt, batch_size, device):
    model.eval()
    losses = []
    for i in range(0, val_inp.size(0), batch_size):
        ib = val_inp[i:i + batch_size]
        tb = val_tgt[i:i + batch_size]
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(ib, targets=tb)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train_one(arch, seed, args, device, val_inp, val_tgt, tokenizer,
              results_dir):
    log("info", f"=== train arch={arch} seed={seed} ===")
    out_dir = os.path.join(results_dir, f"{arch}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    if arch == "swiglu":
        model = make_lm("swiglu", d=args.d, n_heads=args.n_heads,
                         n_layers=args.n_layers, vocab_size=args.vocab_size,
                         max_seq_len=args.seq_len, m=args.swiglu_m)
    elif arch == "tucker":
        model = make_lm("tucker", d=args.d, n_heads=args.n_heads,
                         n_layers=args.n_layers, vocab_size=args.vocab_size,
                         max_seq_len=args.seq_len, r=args.tucker_r,
                         s=args.tucker_s,
                         diagonal_bias_init=args.tucker_diagonal_bias_init)
    elif arch == "tucker_diag":
        model = make_lm("tucker", d=args.d, n_heads=args.n_heads,
                         n_layers=args.n_layers, vocab_size=args.vocab_size,
                         max_seq_len=args.seq_len, r=args.tucker_r,
                         s=args.tucker_s, diagonal_only=True)
    else:
        raise ValueError(arch)

    model.to(device)
    n_params = model.num_params()
    log("info", f"{arch} params = {n_params/1e6:.2f}M | non-embed = "
        f"{model.num_params(exclude_embed=True)/1e6:.2f}M")

    # split params: tucker C (and c_diag) get a configurable lr scale and
    # zero weight decay (the core's job is to learn structure, not regularize
    # toward zero). this is purely an optimization tweak; the parameterization
    # itself (note section IV) is unchanged.
    if arch.startswith("tucker"):
        core_params, other_params = [], []
        for name, p in model.named_parameters():
            if name.endswith(".C") or name.endswith(".c_diag"):
                core_params.append(p)
            else:
                other_params.append(p)
        param_groups = [
            {"params": other_params, "lr": args.peak_lr,
             "weight_decay": 0.1, "base_scale": 1.0},
            {"params": core_params,  "lr": args.peak_lr * args.tucker_core_lr_scale,
             "weight_decay": 0.0,
             "base_scale": args.tucker_core_lr_scale},
        ]
        opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), fused=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.peak_lr,
                                 betas=(0.9, 0.95), weight_decay=0.1, fused=True)

    n_tokens_per_batch = args.batch_size * args.seq_len
    max_steps = math.ceil(args.max_tokens / n_tokens_per_batch)
    eval_every = max(1, math.ceil(args.eval_every_tokens / n_tokens_per_batch))
    log("info", f"max_steps={max_steps} | tokens_per_step={n_tokens_per_batch} | "
        f"eval_every={eval_every} steps")

    cfg_dump = {
        "arch": arch,
        "seed": seed,
        "d": args.d, "n_heads": args.n_heads, "n_layers": args.n_layers,
        "vocab_size": args.vocab_size, "seq_len": args.seq_len,
        "swiglu_m": args.swiglu_m,
        "tucker_r": args.tucker_r, "tucker_s": args.tucker_s,
        "tucker_diagonal_bias_init": bool(args.tucker_diagonal_bias_init),
        "tucker_core_lr_scale": float(args.tucker_core_lr_scale),
        "batch_size": args.batch_size, "max_tokens": args.max_tokens,
        "peak_lr": args.peak_lr, "warmup_steps": args.warmup_steps,
        "n_params_total": n_params,
        "n_params_non_embed": model.num_params(exclude_embed=True),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg_dump, f, indent=2)

    data_iter = stream_fineweb_edu(
        tokenizer, args.seq_len, args.batch_size, device,
        seed=seed, config_name=args.fineweb_config,
    )

    loss_log = []  # list of (step, tokens, train_loss, val_loss_or_none)
    t0 = time.time()
    last_log_t = t0
    train_losses_window = []
    model.train()
    for step in range(max_steps):
        lr = cosine_lr(step, max_steps, args.peak_lr, args.warmup_steps)
        for pg in opt.param_groups:
            base_scale = pg.get("base_scale", 1.0)
            pg["lr"] = lr * base_scale

        try:
            inp, tgt = next(data_iter)
        except StopIteration:
            log("error", "data iterator exhausted; restarting")
            data_iter = stream_fineweb_edu(
                tokenizer, args.seq_len, args.batch_size, device,
                seed=seed + step, config_name=args.fineweb_config,
            )
            inp, tgt = next(data_iter)

        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(inp, targets=tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_losses_window.append(loss.item())

        if (step + 1) % eval_every == 0 or step == max_steps - 1:
            val = eval_loss(model, val_inp, val_tgt, args.batch_size, device)
            tr = float(np.mean(train_losses_window[-50:]))
            tokens_seen = (step + 1) * n_tokens_per_batch
            now = time.time()
            tps = (now - last_log_t)
            last_log_t = now
            log("train", f"step {step+1:5d}/{max_steps} | "
                f"tok={tokens_seen/1e6:6.2f}M | "
                f"lr={lr:.2e} | train={tr:.3f} | val={val:.3f} | "
                f"step_dt={tps/eval_every*1000:.0f}ms")
            loss_log.append({
                "step": step + 1,
                "tokens": tokens_seen,
                "train_loss": tr,
                "val_loss": val,
                "lr": lr,
            })
            with open(os.path.join(out_dir, "loss_log.json"), "w") as f:
                json.dump(loss_log, f, indent=2)

    final_val = eval_loss(model, val_inp, val_tgt, args.batch_size, device)
    log("result", f"{arch} seed={seed} final_val_loss={final_val:.3f} "
        f"(perplexity={math.exp(final_val):.2f})")

    # save final checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": cfg_dump,
        "final_val_loss": final_val,
    }, os.path.join(out_dir, "checkpoint_final.pt"))
    log("done", f"saved checkpoint -> {out_dir}/checkpoint_final.pt | "
        f"time={time.time() - t0:.1f}s")
    return final_val


def plot_loss_curves(results_dir, archs, seeds):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    style = {
        "swiglu": (PALETTE["primary"],  "o", "swiglu (matched params)"),
        "tucker": (PALETTE["ablation"], "s", "tucker (matched params)"),
        "tucker_diag": (PALETTE["accent"], "^", "tucker (diagonal-only)"),
    }
    for arch in archs:
        color, marker, label = style[arch]
        all_curves = []
        tokens_axis = None
        for seed in seeds:
            path = os.path.join(results_dir, f"{arch}_seed{seed}",
                                 "loss_log.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                log_ = json.load(f)
            tokens_axis = np.array([d["tokens"] for d in log_]) / 1e6
            val = np.array([d["val_loss"] for d in log_])
            all_curves.append(val)
        if not all_curves:
            continue
        all_curves = np.stack(all_curves)
        mean = all_curves.mean(axis=0)
        std = all_curves.std(axis=0) if all_curves.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(tokens_axis, mean, marker=marker, color=color, lw=1.5, ms=3.5,
                label=label)
        if all_curves.shape[0] > 1:
            ax.fill_between(tokens_axis, mean - std, mean + std,
                             color=color, alpha=0.15)
    ax.set_xlabel("tokens (M)")
    ax.set_ylabel("validation loss")
    ax.legend(framealpha=0.9, edgecolor="0.8")
    plt.tight_layout()
    out = os.path.join(results_dir, "loss_curves.png")
    plt.savefig(out)
    plt.close()
    log("done", f"saved loss_curves.png -> {results_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archs", type=str, default="swiglu,tucker",
                        help="comma-separated list of arch tags to train")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--swiglu_m", type=int, default=None,
                        help="if None, computed to match tucker(r=s)")
    parser.add_argument("--tucker_r", type=int, default=128)
    parser.add_argument("--tucker_s", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_tokens", type=int, default=200_000_000)
    parser.add_argument("--peak_lr", type=float, default=3e-4)
    parser.add_argument("--tucker_core_lr_scale", type=float, default=1.0,
                        help="lr multiplier for tucker C/c_diag relative to "
                             "the rest of the model. set <1 to dampen if the "
                             "core is unstable, >1 to push it harder.")
    parser.add_argument("--tucker_diagonal_bias_init", action="store_true",
                        help="initialize tucker C with superdiagonal bias "
                             "(swiglu-equivalent at init) so the model can "
                             "deviate into off-diagonal interactions only "
                             "when training prefers it. parameterization "
                             "unchanged.")
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--eval_every_tokens", type=int, default=2_000_000)
    parser.add_argument("--n_val_seqs", type=int, default=128)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--fineweb_config", type=str, default="sample-10BT")
    parser.add_argument("--results_dir", type=str, default="results/exp11")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--plot_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    archs = args.archs.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.plot_only:
        plot_loss_curves(args.results_dir, archs, seeds)
        return

    if args.swiglu_m is None:
        args.swiglu_m = matched_swiglu_for_tucker(
            args.d, args.tucker_r, args.tucker_s,
        )
    log("info", f"swiglu_m={args.swiglu_m} | tucker_r={args.tucker_r} | "
        f"tucker_s={args.tucker_s}")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    args.vocab_size = max(args.vocab_size, len(tokenizer))
    log("info", f"vocab_size={args.vocab_size} | tokenizer={args.tokenizer}")

    log("info", "building val set")
    val_inp, val_tgt = build_val_set(
        tokenizer, args.seq_len, args.n_val_seqs, args.device,
        config_name=args.fineweb_config,
    )
    log("info", f"val_inp={val_inp.shape}")

    for arch in archs:
        for seed in seeds:
            train_one(arch, seed, args, args.device, val_inp, val_tgt,
                       tokenizer, args.results_dir)
            print()

    plot_loss_curves(args.results_dir, archs, seeds)
    log("done", "exp11 complete")


if __name__ == "__main__":
    main()
