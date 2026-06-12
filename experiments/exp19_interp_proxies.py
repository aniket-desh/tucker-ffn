#!/usr/bin/env python3
"""experiment 19: interpretability proxies on trained sprint LMs.

For each trained checkpoint (swiglu / tucker / ll1_l*), stream a fixed
FineWeb-Edu validation set and measure, per FFN layer (theory_notes §7):

  1. per-token unit contributions c_u(x) where a "unit" is the architecture's
     natural routed object:
       swiglu : atom j,  c_j = |h_j(x)| * ||u_j||_2
       ll1    : block b, c_b = ||U_b r_b(x)||_2 * |silu(g_b^T x)| (computed
                exactly as the norm of the block's output contribution)
       tucker : gate j,  c_j = ||V_j p||_2 * |silu(q_j)|
     metrics: effective active count exp(H(p)), 90% mass fraction — reported
     both absolute and as a fraction of available units.
  2. gate sparsity: distribution of the sigmoid routing coefficient
     (sigma(g^T x) for swiglu/ll1 gates, sigma(q_j) for tucker).
  3. weight-based per-gate stable rank (tucker: V_j = R C^(j); ll1: U_b A_b^T,
     <= L by construction; swiglu: 1 by definition).
  4. ablation locality: delta val loss from zeroing single units, for a sample
     of units per layer (random + top-by-mean-contribution); the whole-model
     loss is recomputed per ablation.
  5. tucker core diagnostics: core entropy, energy fraction within the best
     block-diagonal mask (greedy assignment), superdiagonal energy fraction.

outputs (under --results_dir): exp19_results.json + per-metric figures.
"""

import argparse
import json
import math
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import log, setup_plot_style, PALETTE  # noqa: E402
from lib.lm import LM, LMConfig, FFNConfig  # noqa: E402
from lib.ll1_ffn import LL1FFN  # noqa: E402
from lib.tucker_ffn import SwiGLUFFN, TuckerFFN  # noqa: E402


def load_ckpt(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    arch = cfg["arch"]
    if arch == "swiglu":
        ffn = FFNConfig(kind="swiglu", m=cfg["swiglu_m"])
    elif arch.startswith("tucker"):
        ffn = FFNConfig(kind="tucker", r=cfg["tucker_r"], s=cfg["tucker_s"])
    elif arch.startswith("ll1_l"):
        L = int(arch[len("ll1_l"):])
        # infer B from state dict
        w = ck["model_state_dict"]["blocks.0.ffn.gate_proj.weight"]
        ffn = FFNConfig(kind="ll1", n_blocks=w.shape[0], block_rank=L)
    else:
        raise ValueError(arch)
    lmcfg = LMConfig(vocab_size=cfg["vocab_size"], d=cfg["d"],
                     n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                     max_seq_len=cfg["seq_len"], ffn=ffn)
    model = LM(lmcfg)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def unit_contributions(ffn, x):
    """x: (n, d) ffn inputs. returns (contrib (n, U), alpha (n, Ugates))."""
    if isinstance(ffn, SwiGLUFFN):
        gate_pre = ffn.gate_proj(x)
        h = ffn.up_proj(x) * F.silu(gate_pre)            # (n, m)
        u_norm = ffn.down_proj.weight.norm(dim=0)        # (m,)
        return h.abs() * u_norm, torch.sigmoid(gate_pre)
    if isinstance(ffn, LL1FFN):
        gate_pre = ffn.gate_proj(x)                      # (n, B)
        s = F.silu(gate_pre)
        r = ffn.up_proj(x).view(x.shape[0], ffn.n_blocks, ffn.block_rank)
        U = ffn.down_proj.weight.T.view(ffn.n_blocks, ffn.block_rank, ffn.d)
        block_out = torch.einsum("nbl,bld->nbd", r, U)   # (n, B, d)
        c = block_out.norm(dim=-1) * s.abs()             # (n, B)
        return c, torch.sigmoid(gate_pre)
    if isinstance(ffn, TuckerFFN):
        p = x @ ffn.P                                    # (n, r)
        q = x @ ffn.Q
        sq = F.silu(q)                                   # (n, r)
        # z_oj = sum_i C_oij p_i  -> (n, s, r_gates)
        zj = torch.einsum("oij,ni->noj", ffn.C, p)
        # y_j = R z_:,j * silu(q_j); contribution norm per gate j
        yj = torch.einsum("do,noj->ndj", ffn.R, zj)      # (n, d, r)
        c = yj.norm(dim=1) * sq.abs()                    # (n, r)
        return c, torch.sigmoid(q)
    raise TypeError(type(ffn))


def eff_count(contrib):
    """exp(entropy) per token, averaged. contrib (n, U) nonneg."""
    p = contrib / (contrib.sum(dim=1, keepdim=True) + 1e-12)
    H = -(p * (p + 1e-12).log()).sum(dim=1)
    return H.exp().mean().item()


def mass90(contrib):
    """mean fraction of units covering 90% of total |c| per token."""
    sorted_c, _ = contrib.sort(dim=1, descending=True)
    cum = sorted_c.cumsum(dim=1)
    tot = cum[:, -1:]
    need = (cum < 0.9 * tot).sum(dim=1).float() + 1
    return (need / contrib.shape[1]).mean().item()


def stable_ranks(ffn):
    """weight-based per-gate stable rank ||V||_F^2/||V||_op^2."""
    with torch.no_grad():
        if isinstance(ffn, TuckerFFN):
            V = torch.einsum("do,oir->rdi", ffn.R, ffn.C.permute(0, 1, 2))
            # careful: C is (s, r_main, r_gate); slice j: C[:, :, j] (s, r)
            V = torch.stack([ffn.R @ ffn.C[:, :, j] for j in range(ffn.r)])
        elif isinstance(ffn, LL1FFN):
            V = ffn.per_gate_matrices()
        else:
            return None
        fro = V.pow(2).sum(dim=(1, 2))
        op = torch.linalg.matrix_norm(V, ord=2) ** 2
        return (fro / (op + 1e-12)).cpu().numpy()


def tucker_core_diags(ffn):
    with torch.no_grad():
        C = ffn.C.abs()
        e = C.pow(2)
        tot = e.sum().item()
        s, r, _ = C.shape
        k = min(s, r)
        idx = torch.arange(k)
        diag_e = e[idx, idx, idx].sum().item()
        p = (e / e.sum()).flatten()
        H = -(p * (p + 1e-12).log()).sum().item()
        eff_frac = math.exp(H) / p.numel()
        # block-diagonal energy at block size L: partition latent indices into
        # contiguous blocks after sorting rows/cols by spectral co-clustering
        # proxy (greedy): for simplicity report energy in |o-i|<=L band and
        # gate-coupling band |i-j|<=L for L in {4, 16}
        bands = {}
        oi = torch.arange(s).view(-1, 1, 1) - torch.arange(r).view(1, -1, 1)
        ij = torch.arange(r).view(1, -1, 1) - torch.arange(r).view(1, 1, -1)
        for L in (4, 16):
            mask = (oi.abs() <= L) & (ij.abs() <= L)
            bands[f"band_L{L}"] = e[mask.expand_as(e)].sum().item() / tot
        return {"superdiag_energy_frac": diag_e / tot,
                "core_eff_entry_frac": eff_frac, **bands}


@torch.no_grad()
def ablation_locality(model, ffn_layer_idx, unit_ids, val_inp, val_tgt,
                      batch_size, device, base_loss):
    """delta val loss from zeroing one unit in one layer's ffn."""
    ffn = model.blocks[ffn_layer_idx].ffn
    deltas = []
    for u in unit_ids:
        handle = None
        if isinstance(ffn, SwiGLUFFN):
            def hook(mod, inp, out, u=u):
                out = out.clone(); out[..., u] = 0; return out
            handle = ffn.up_proj.register_forward_hook(hook)
        elif isinstance(ffn, LL1FFN):
            def hook(mod, inp, out, u=u):
                out = out.clone(); out[..., u] = 0; return out
            handle = ffn.gate_proj.register_forward_hook(hook)
        elif isinstance(ffn, TuckerFFN):
            def hook(mod, inp, out, u=u):
                return out
            # zero gate j: monkeypatch via Q-column zeroing is wrong (silu(0)=0
            # exactly zeroes the gate slice contribution). silu(q_j)=0 <=> q_j=0.
            # easiest: temporarily zero column j of Q AND rely on silu(0)=0.
            old = ffn.Q[:, u].clone()
            ffn.Q[:, u] = 0
        losses = []
        for i in range(0, val_inp.size(0), batch_size):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(val_inp[i:i+batch_size],
                                targets=val_tgt[i:i+batch_size])
            losses.append(loss.item())
        if handle is not None:
            handle.remove()
        if isinstance(ffn, TuckerFFN):
            ffn.Q[:, u] = old
        deltas.append(float(np.mean(losses)) - base_loss)
    return deltas


class TopKWrap(torch.nn.Module):
    """wrap an FFN so only the top-k routed units (by per-token contribution
    norm) contribute to the output. exact per-unit decomposition, no approx."""

    def __init__(self, ffn, k):
        super().__init__()
        self.ffn = ffn
        self.k = k

    def forward(self, x):
        ffn, k = self.ffn, self.k
        shape = x.shape[:-1]
        xf = x.reshape(-1, x.shape[-1])
        if isinstance(ffn, SwiGLUFFN):
            h = ffn.up_proj(xf) * F.silu(ffn.gate_proj(xf))      # (n, m)
            c = h.abs() * ffn.down_proj.weight.norm(dim=0)
            thresh = c.topk(k, dim=1).values[:, -1:]
            h = h * (c >= thresh)
            out = ffn.down_proj(h)
        elif isinstance(ffn, LL1FFN):
            s = F.silu(ffn.gate_proj(xf))                        # (n, B)
            r = ffn.up_proj(xf).view(-1, ffn.n_blocks, ffn.block_rank)
            U = ffn.down_proj.weight.T.view(ffn.n_blocks, ffn.block_rank,
                                            ffn.d)
            block_out = torch.einsum("nbl,bld->nbd", r, U) * s.unsqueeze(-1)
            c = block_out.norm(dim=-1)                           # (n, B)
            thresh = c.topk(k, dim=1).values[:, -1:]
            out = (block_out * (c >= thresh).unsqueeze(-1)).sum(dim=1)
        elif isinstance(ffn, TuckerFFN):
            p = xf @ ffn.P
            q = xf @ ffn.Q
            sq = F.silu(q)                                       # (n, r)
            out = torch.zeros(xf.shape[0], ffn.d, device=xf.device,
                              dtype=xf.dtype)
            # chunk over tokens to bound memory of (n, d, r) intermediate
            for i in range(0, xf.shape[0], 2048):
                zj = torch.einsum("oij,ni->noj", ffn.C, p[i:i+2048])
                yj = torch.einsum("do,noj->ndj", ffn.R, zj)      # (n, d, r)
                yj = yj * sq[i:i+2048].unsqueeze(1)
                c = yj.norm(dim=1)                               # (n, r)
                thresh = c.topk(k, dim=1).values[:, -1:]
                out[i:i+2048] = (yj * (c >= thresh).unsqueeze(1)).sum(dim=2)
        else:
            raise TypeError(type(ffn))
        return out.view(*shape, -1)


@torch.no_grad()
def topk_loss_curve(model, ks, val_inp, val_tgt, batch_size):
    """val loss with every layer's ffn restricted to top-k units."""
    out = {}
    orig = [blk.ffn for blk in model.blocks]
    for k in ks:
        for blk, ffn in zip(model.blocks, orig):
            blk.ffn = TopKWrap(ffn, min(k, _n_units(ffn)))
        losses = []
        for i in range(0, val_inp.size(0), batch_size):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(val_inp[i:i+batch_size],
                                targets=val_tgt[i:i+batch_size])
            losses.append(loss.item())
        out[k] = float(np.mean(losses))
    for blk, ffn in zip(model.blocks, orig):
        blk.ffn = ffn
    return out


def _n_units(ffn):
    if isinstance(ffn, SwiGLUFFN):
        return ffn.m
    if isinstance(ffn, LL1FFN):
        return ffn.n_blocks
    if isinstance(ffn, TuckerFFN):
        return ffn.r
    raise TypeError(type(ffn))


def build_val(tokenizer_name, seq_len, n_seqs, device):
    from transformers import AutoTokenizer
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from exp11_train_lm import build_val_set
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return build_val_set(tok, seq_len, n_seqs, device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", type=str, required=True,
                    help="comma-separated checkpoint paths")
    ap.add_argument("--n_val_seqs", type=int, default=32)
    ap.add_argument("--n_contrib_tokens", type=int, default=4096)
    ap.add_argument("--n_ablate_units", type=int, default=48)
    ap.add_argument("--ablate_layers", type=str, default="3",
                    help="comma-separated layer indices for ablation study")
    ap.add_argument("--topk_ks", type=str, default="1,2,4,8,16,32,64,128,256,512,1024",
                    help="top-k unit counts for the decomposability curve "
                         "(clipped per-arch to available units)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--results_dir", type=str, default="results/exp19")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = args.device
    val_inp, val_tgt = build_val("gpt2", args.seq_len, args.n_val_seqs, device)

    all_results = []
    for ckpt_path in args.ckpts.split(","):
        model, cfg = load_ckpt(ckpt_path, device)
        tag = f"{cfg['arch']}_seed{cfg['seed']}"
        log("info", f"=== {tag} ({ckpt_path}) ===")

        # capture ffn inputs on a slice of val data
        ffn_inputs = {}
        hooks = []
        for li, blk in enumerate(model.blocks):
            def mk(li):
                def hook(mod, inp, out):
                    ffn_inputs.setdefault(li, []).append(out.detach())
                return hook
            hooks.append(blk.norm2.register_forward_hook(mk(li)))
        base_losses = []
        with torch.no_grad():
            for i in range(0, val_inp.size(0), args.batch_size):
                with torch.amp.autocast(device_type="cuda",
                                        dtype=torch.bfloat16):
                    _, loss = model(val_inp[i:i+args.batch_size],
                                    targets=val_tgt[i:i+args.batch_size])
                base_losses.append(loss.item())
        for h in hooks:
            h.remove()
        base_loss = float(np.mean(base_losses))

        layer_metrics = []
        for li in range(len(model.blocks)):
            x = torch.cat([t.reshape(-1, t.shape[-1]).float()
                           for t in ffn_inputs[li]], 0)
            x = x[torch.randperm(x.shape[0])[:args.n_contrib_tokens]]
            ffn = model.blocks[li].ffn
            contrib, alpha = unit_contributions(ffn, x)
            sr = stable_ranks(ffn)
            lm = {
                "layer": li,
                "n_units": contrib.shape[1],
                "eff_active": eff_count(contrib),
                "eff_active_frac": eff_count(contrib) / contrib.shape[1],
                "mass90_frac": mass90(contrib),
                "alpha_mean": alpha.mean().item(),
                "alpha_frac_below_0.1": (alpha < 0.1).float().mean().item(),
                "alpha_frac_above_0.9": (alpha > 0.9).float().mean().item(),
                "stable_rank_mean": float(np.mean(sr)) if sr is not None else None,
                "stable_rank_max": float(np.max(sr)) if sr is not None else None,
            }
            if isinstance(ffn, TuckerFFN):
                lm.update(tucker_core_diags(ffn))
            layer_metrics.append(lm)
            del x, contrib, alpha

        # ablation locality on selected layers
        ablate = {}
        for li in [int(s) for s in args.ablate_layers.split(",")]:
            ffn = model.blocks[li].ffn
            x = torch.cat([t.reshape(-1, t.shape[-1]).float()
                           for t in ffn_inputs[li]], 0)[:args.n_contrib_tokens]
            contrib, _ = unit_contributions(ffn, x)
            mean_c = contrib.mean(0)
            n_units = mean_c.shape[0]
            top = mean_c.argsort(descending=True)[:args.n_ablate_units // 3].cpu()
            g = torch.Generator().manual_seed(0)
            rnd = torch.randperm(n_units, generator=g)[:args.n_ablate_units
                                                       - top.numel()]
            ids = torch.cat([top, rnd]).unique().tolist()
            deltas = ablation_locality(model, li, ids, val_inp, val_tgt,
                                       args.batch_size, device, base_loss)
            ablate[li] = {"unit_ids": ids, "delta_loss": deltas,
                          "kind": "top+random"}
            log("info", f"  layer {li}: ablation max dLoss="
                f"{max(deltas):.4f} median={np.median(deltas):.5f}")

        ks = sorted({min(int(s), _n_units(model.blocks[0].ffn))
                     for s in args.topk_ks.split(",")})
        topk = topk_loss_curve(model, ks, val_inp, val_tgt, args.batch_size)
        log("info", f"  topk curve: {topk}")

        all_results.append({
            "tag": tag, "arch": cfg["arch"], "seed": cfg["seed"],
            "ckpt": ckpt_path, "base_loss": base_loss,
            "layers": layer_metrics, "ablation": ablate,
            "topk_loss": topk,
        })
        with open(os.path.join(args.results_dir, "exp19_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        del model, ffn_inputs
        torch.cuda.empty_cache()
        log("done", f"{tag}: base_loss={base_loss:.4f}")

    log("done", "exp19 complete")


if __name__ == "__main__":
    main()
