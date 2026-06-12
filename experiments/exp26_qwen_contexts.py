#!/usr/bin/env python3
"""experiment 26 (sprint 2, Exp E + confound D): real-pretrained-layer
interpretability with context-specific metrics.

On Qwen2.5-0.5B, one FFN layer:

1. Mine top-activating contexts for a sample of SwiGLU atoms (contribution
   c_j = |h_j| * ||u_j||). Coherence proxy = token selectivity: the fraction
   of an atom's top-20 contexts that share the same current token (and same
   previous token). Compare against a shuffled-position null.
2. Local vs global causal effect: for the most selective atoms, zero the atom
   in the FULL model and measure delta log-prob of the next token ON the
   atom's top contexts vs on random positions. If local >> global, sprint-1's
   global ablations washed out real context-specific structure (confound D).
3. Distill the layer into LL1(L=4) and sparse-CP students (route-L1) at a
   1.2M budget; mine student atom/block contexts; same selectivity metric.
   Question: do student blocks inherit, sharpen, or lose the teacher's
   context structure? Does trained sparsity sharpen it?

outputs: results/exp26/exp26_results.json, top-context text dumps in
results/exp26/contexts_*.txt, figure exp26_selectivity.png
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

from lib import log, setup_plot_style  # noqa: E402
from lib.model_utils import load_model_and_tokenizer, get_swiglu_layers  # noqa: E402
from lib.tucker_ffn import SwiGLUFFN  # noqa: E402
from lib.ll1_ffn import LL1FFN, ll1_blocks_for_params  # noqa: E402
from exp18_ll1_synthetic import fit_student  # noqa: E402
from exp24_superposition import fit as fit_sparse  # noqa: E402


@torch.no_grad()
def capture_with_tokens(model, tok, layer_idx, n_tokens, seq_len, device,
                        batch_seqs=4):
    """returns X (n, d) ffn inputs, Y (n, d) outputs, ids (n,) token ids,
    and the list of (input_ids) batches for later causal tests."""
    from datasets import load_dataset
    info = [i for i in get_swiglu_layers(model) if i["layer_idx"] == layer_idx][0]
    store = {"x": [], "y": []}

    def hook(mod, inp, out):
        store["x"].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu())
        o = out[0] if isinstance(out, tuple) else out
        store["y"].append(o.detach().reshape(-1, o.shape[-1]).cpu())

    h = info["mlp"].register_forward_hook(hook)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train",
                      streaming=True).shuffle(seed=5, buffer_size=10_000)
    buf, batches, ids = [], [], []
    batch = []
    seen = 0
    for ex in ds:
        buf.extend(tok.encode(ex.get("text", "")))
        while len(buf) >= seq_len:
            batch.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(batch) == batch_seqs:
                t = torch.tensor(batch, device=device)
                model(t)
                batches.append(t.cpu())
                ids.append(t.reshape(-1).cpu())
                seen += batch_seqs * seq_len
                batch = []
        if seen >= n_tokens:
            break
    h.remove()
    return (torch.cat(store["x"]), torch.cat(store["y"]),
            torch.cat(ids), batches, info)


def token_selectivity(top_ids):
    """top_ids: list of (current_token, prev_token) for an atom's top
    contexts. Returns (frac sharing modal current token, frac sharing modal
    prev token)."""
    cur = [c for c, _ in top_ids]
    prv = [p for _, p in top_ids]
    fc = max(np.bincount(cur)) / len(cur) if cur else 0
    fp = max(np.bincount(prv)) / len(prv) if prv else 0
    return float(fc), float(fp)


@torch.no_grad()
def atom_contribs_swiglu_weights(x, W_up, W_gate, u_norm, atom_ids,
                                 chunk=16384):
    """contributions c_j = |up_j * silu(gate_j)| * ||u_j|| for selected atoms."""
    outs = []
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk]
        up = xb @ W_up[atom_ids].T
        gt = xb @ W_gate[atom_ids].T
        outs.append((up * F.silu(gt)).abs() * u_norm[atom_ids])
    return torch.cat(outs)


def mine(contrib, ids, k=20):
    """per selected atom: top-k positions, values, (cur, prev) token pairs."""
    res = []
    for a in range(contrib.shape[1]):
        vals, pos = contrib[:, a].topk(k)
        pairs = [(int(ids[p]), int(ids[max(0, p - 1)])) for p in pos.tolist()]
        res.append({"positions": pos.tolist(), "values": vals.tolist(),
                    "token_pairs": pairs})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n_tokens", type=int, default=200_000)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--n_atoms", type=int, default=256)
    ap.add_argument("--n_causal_atoms", type=int, default=12)
    ap.add_argument("--budget", type=int, default=1_200_000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--l1_lambda", type=float, default=1e-3)
    ap.add_argument("--results_dir", type=str, default="results/exp26")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    dev = args.device

    model, tok = load_model_and_tokenizer(args.model, dev)
    X, Y, ids, batches, info = capture_with_tokens(
        model, tok, args.layer, args.n_tokens, args.seq_len, dev)
    d = X.shape[1]
    log("info", f"captured {X.shape[0]} positions at layer {args.layer}")

    # ── teacher atoms ────────────────────────────────────────────────────────
    mlp = info["mlp"]
    W_up = mlp.up_proj.weight.detach().float().cpu()
    W_gate = mlp.gate_proj.weight.detach().float().cpu()
    u_norm = mlp.down_proj.weight.detach().float().norm(dim=0).cpu()
    g = torch.Generator().manual_seed(0)
    atom_ids = torch.randperm(W_up.shape[0], generator=g)[:args.n_atoms]
    contrib = atom_contribs_swiglu_weights(X.float(), W_up, W_gate, u_norm,
                                           atom_ids)
    mined_T = mine(contrib, ids)
    sel_T = [token_selectivity(m["token_pairs"]) for m in mined_T]

    # null: random positions
    null_sel = []
    for _ in range(args.n_atoms):
        pos = torch.randperm(len(ids), generator=g)[:20]
        pairs = [(int(ids[p]), int(ids[max(0, p - 1)])) for p in pos.tolist()]
        null_sel.append(token_selectivity(pairs))

    # ── students ────────────────────────────────────────────────────────────
    n_val = int(0.1 * len(X))
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(1))
    Xs, Ys = X[perm].float(), Y[perm].float()
    scale = Ys[n_val:].std()
    Ys = Ys / scale
    x_tr, y_tr = Xs[n_val:].to(dev), Ys[n_val:].to(dev)
    x_va, y_va = Xs[:n_val].to(dev), Ys[:n_val].to(dev)

    students = {}
    torch.manual_seed(3)
    m_cp = max(1, round(args.budget / (3 * d)))
    students["swiglu_student"] = (SwiGLUFFN(d, m_cp).to(dev), 0.0)
    torch.manual_seed(3)
    students["sparse_cp_student"] = (SwiGLUFFN(d, m_cp).to(dev), args.l1_lambda)
    torch.manual_seed(3)
    B = ll1_blocks_for_params(d, 4, args.budget)
    students["ll1_l4_student"] = (LL1FFN(d, n_blocks=B, block_rank=4).to(dev), 0.0)

    student_stats = {}
    for name, (stu, lam) in students.items():
        rel = fit_sparse(stu, (x_tr, y_tr, x_va, y_va), args.steps, 2e-3, 512,
                         dev, l1_lambda=lam)
        # student unit contributions on the SAME positions (use X unpermuted)
        from exp19_interp_proxies import unit_contributions
        cs = []
        for i in range(0, X.shape[0], 16384):
            c, _ = unit_contributions(stu, X[i:i + 16384].float().to(dev))
            cs.append(c.cpu())
        cs = torch.cat(cs)
        # sample units for mining
        uids = torch.randperm(cs.shape[1], generator=g)[:args.n_atoms]
        mined_S = mine(cs[:, uids], ids)
        sel_S = [token_selectivity(m["token_pairs"]) for m in mined_S]
        student_stats[name] = {
            "rel_mse": rel,
            "n_units": cs.shape[1],
            "sel_cur_mean": float(np.mean([s[0] for s in sel_S])),
            "sel_cur_p90": float(np.quantile([s[0] for s in sel_S], 0.9)),
            "sel_prev_mean": float(np.mean([s[1] for s in sel_S])),
        }
        log("result", f"{name}: relMSE={rel:.4f} "
            f"sel_cur={student_stats[name]['sel_cur_mean']:.3f}")
        del stu
        torch.cuda.empty_cache()

    # ── causal: local vs global ablation in the FULL model ─────────────────
    sel_arr = np.array([s[0] for s in sel_T])
    top_sel = np.argsort(-sel_arr)[:args.n_causal_atoms]
    causal = []
    for a_i in top_sel.tolist():
        atom = int(atom_ids[a_i])
        top_pos = mined_T[a_i]["positions"][:10]
        # measure delta logprob at top positions and at random positions
        rnd_pos = torch.randperm(len(ids), generator=g)[:10].tolist()

        def eval_pos(positions, ablate):
            deltas = []
            handle = None
            if ablate:
                def hook(mod, inp, out):
                    out = out.clone()
                    out[..., atom] = 0
                    return out
                handle = mlp.up_proj.register_forward_hook(hook)
            with torch.no_grad():
                lps = {}
                for p in positions:
                    b, r = divmod(p, args.seq_len * 4)  # batches of 4 seqs
                    if b >= len(batches):
                        continue
                    inp = batches[b].to(dev)
                    logits = model(inp).logits.float().log_softmax(-1)
                    s, t = divmod(r, args.seq_len)
                    if t + 1 >= args.seq_len:
                        continue
                    nxt = inp[s, t + 1]
                    lps[p] = logits[s, t, nxt].item()
            if handle:
                handle.remove()
            return lps

        base_top = eval_pos(top_pos, False)
        abl_top = eval_pos(top_pos, True)
        base_rnd = eval_pos(rnd_pos, False)
        abl_rnd = eval_pos(rnd_pos, True)
        loc = np.mean([base_top[p] - abl_top[p] for p in base_top
                       if p in abl_top]) if base_top else float("nan")
        glob = np.mean([base_rnd[p] - abl_rnd[p] for p in base_rnd
                        if p in abl_rnd]) if base_rnd else float("nan")
        causal.append({"atom": atom, "selectivity": float(sel_arr[a_i]),
                       "local_dlogp": float(loc), "global_dlogp": float(glob)})
        log("info", f"atom {atom}: sel={sel_arr[a_i]:.2f} "
            f"local dlogp={loc:.4f} global dlogp={glob:.4f}")

    out = {
        "layer": args.layer,
        "teacher": {
            "sel_cur_mean": float(np.mean([s[0] for s in sel_T])),
            "sel_cur_p90": float(np.quantile([s[0] for s in sel_T], 0.9)),
            "sel_prev_mean": float(np.mean([s[1] for s in sel_T])),
            "null_cur_mean": float(np.mean([s[0] for s in null_sel])),
        },
        "students": student_stats,
        "causal": causal,
    }
    with open(os.path.join(args.results_dir, "exp26_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # text dump of a few atoms' contexts
    with open(os.path.join(args.results_dir, "contexts_teacher.txt"), "w") as f:
        for a_i in top_sel.tolist()[:8]:
            f.write(f"=== atom {int(atom_ids[a_i])} sel={sel_arr[a_i]:.2f} ===\n")
            for p, v in zip(mined_T[a_i]["positions"][:10],
                            mined_T[a_i]["values"][:10]):
                lo = max(0, p - 10)
                txt = tok.decode(ids[lo:p + 1].tolist())
                f.write(f"  [{v:8.2f}] ...{txt!r}\n")
            f.write("\n")

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(0, 1, 21)
    ax.hist([s[0] for s in sel_T], bins=bins, alpha=0.5,
            label=f"teacher atoms (mean {out['teacher']['sel_cur_mean']:.2f})",
            color="#2c7fb8", density=True)
    ax.hist([s[0] for s in null_sel], bins=bins, alpha=0.5,
            label=f"null (mean {out['teacher']['null_cur_mean']:.2f})",
            color="gray", density=True)
    ax.set_xlabel("token selectivity of top-20 contexts")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "exp26_selectivity.png"), dpi=180)
    log("done", "exp26 complete")


if __name__ == "__main__":
    main()
