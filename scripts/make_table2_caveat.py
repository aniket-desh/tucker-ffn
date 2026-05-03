#!/usr/bin/env python3
"""Within-run std of val loss over the last 10 checkpoints, per arch.
Emits snippets/table2_caveat.tex (one LaTeX sentence)."""
import json, numpy as np, pathlib


def last_n_std(loss_log_path, n=10):
    with open(loss_log_path) as f:
        log_ = json.load(f)
    vals = [d["val_loss"] for d in log_[-n:]]
    return float(np.std(vals)), float(np.mean(vals)), len(vals)


sw_std, sw_mean, sw_n = last_n_std("results/exp11/swiglu_seed0/loss_log.json")
tk_std, tk_mean, tk_n = last_n_std("results/exp11_hc_v3/tucker_seed0/loss_log.json")

snippet = (
    "Single seed; within-run std of val.\\ loss over the last "
    "{n} checkpoints is $\\approx{sw:.3f}$ (SwiGLU) and $\\approx{tk:.3f}$ (Tucker), "
    "comparable to or larger than the 0.005-nat gap, so the table should be "
    "read as a sanity check rather than a superiority claim.\n"
).format(n=sw_n, sw=sw_std, tk=tk_std)

pathlib.Path("snippets").mkdir(exist_ok=True)
pathlib.Path("snippets/table2_caveat.tex").write_text(snippet)
print(snippet)
print(f"swiglu mean={sw_mean:.4f} std={sw_std:.4f} (n={sw_n})")
print(f"tucker mean={tk_mean:.4f} std={tk_std:.4f} (n={tk_n})")
