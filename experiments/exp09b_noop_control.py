#!/usr/bin/env python3
"""experiment 9b: no-op pairing control for exp09 framing.

if we permute gate_proj rows AND up_proj rows AND down_proj cols ALL by
the same π, the cp decomposition is just relabeled:
    atom_j = u_{pi(j)} . w_{pi(j)} . g_{pi(j)} = original atom_{pi(j)}.
the set of rank-one atoms is unchanged, only their hidden-index labeling
moves. perplexity should equal baseline (modulo numerical noise).

this gives the paper a sanity check: any cost above baseline in exp09
joint or u_only is necessarily attributable to *breaking* a pairing,
not to "permuting weights is bad on its own".

exp09 itself does not include this because the existing permuted_layer
context manager only wires gate and down. we extend it inline here.

outputs (under --results_dir):
  noop_control.json
"""

import argparse
import json
import os
import pathlib
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import (  # noqa: E402
    add_common_args,
    compute_perplexity,
    log,
    prepare_run,
)


@contextmanager
def permuted_all(layers_info, layer_idx, perm):
    """permute gate rows + up rows + down cols all by same perm (no-op)."""
    info = layers_info[layer_idx]
    gp = info["gate_proj"]
    up = info["up_proj"]
    dp = info["down_proj"]

    perm = perm.to(gp.weight.device)
    saved = {
        "gw": gp.weight.data.clone(),
        "uw": up.weight.data.clone(),
        "dw": dp.weight.data.clone(),
    }
    if gp.bias is not None:
        saved["gb"] = gp.bias.data.clone()
    if up.bias is not None:
        saved["ub"] = up.bias.data.clone()

    gp.weight.data = gp.weight.data[perm, :].contiguous()
    up.weight.data = up.weight.data[perm, :].contiguous()
    dp.weight.data = dp.weight.data[:, perm].contiguous()
    if gp.bias is not None:
        gp.bias.data = gp.bias.data[perm].contiguous()
    if up.bias is not None:
        up.bias.data = up.bias.data[perm].contiguous()
    try:
        yield
    finally:
        gp.weight.data = saved["gw"]
        up.weight.data = saved["uw"]
        dp.weight.data = saved["dw"]
        if "gb" in saved:
            gp.bias.data = saved["gb"]
        if "ub" in saved:
            up.bias.data = saved["ub"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--n_seeds", type=int, default=4,
                        help="number of random permutations per layer")
    parser.add_argument("--out_name", type=str, default="noop_control.json")
    args = parser.parse_args()

    t0 = time.time()
    ctx = prepare_run(args, capture_activations=False)
    model = ctx["model"]
    layers_info = ctx["layers_info"]
    eval_ids = ctx["eval_ids"]
    device = ctx["device"]

    m = layers_info[0]["gate_proj"].weight.shape[0]
    n_layers = len(layers_info)

    ppl_baseline = compute_perplexity(model, eval_ids, device)
    log("result", f"baseline | perplexity={ppl_baseline:.4f}")

    raw = np.zeros((n_layers, args.n_seeds))
    for li in range(n_layers):
        for si in range(args.n_seeds):
            gen = torch.Generator().manual_seed(args.seed + 1000 * si + li)
            pi = torch.randperm(m, generator=gen)
            with permuted_all(layers_info, li, pi):
                ppl = compute_perplexity(model, eval_ids, device)
            raw[li, si] = ppl
            log("eval", f"layer {li:02d} | seed {si} | "
                f"noop_ppl={ppl:.4f} | delta={ppl-ppl_baseline:+.2e}")
        log("result", f"layer {li:02d} | mean_noop={raw[li].mean():.4f} | "
            f"max_dev={np.abs(raw[li]-ppl_baseline).max():.2e}")

    results = {
        "baseline": ppl_baseline,
        "n_seeds": args.n_seeds,
        "n_layers": n_layers,
        "noop_per_layer_mean": raw.mean(axis=1).tolist(),
        "noop_per_layer_max_dev": np.abs(raw - ppl_baseline).max(axis=1).tolist(),
        "raw": raw.tolist(),
    }
    out = os.path.join(args.results_dir, args.out_name)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log("done", f"saved {args.out_name} -> {args.results_dir}/")
    log("done", f"experiment 9b complete | time={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
