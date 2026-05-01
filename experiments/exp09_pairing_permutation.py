#!/usr/bin/env python3
"""experiment 9: same-index pairing permutation.

tests whether the cp decomposition's same-index coupling between W and G
is *binding* — i.e., whether the model has learned a specific pairing of
up-projection rows w_j and gate-projection rows g_j that the rest of the
layer depends on, or whether the choice of pairing is a parameterization
the model would be indifferent to.

per layer (one at a time, all others intact), apply a random permutation
π and measure perplexity under two conditions:

  joint  : permute gate_proj rows AND down_proj cols by the same π
           — leaves G and U paired (both move to π(j)) but breaks the
             W-G and W-U same-index couplings
  u_only : permute down_proj cols only by π_u
           — control: breaks U pairing without touching W-G

both conditions perturb the same number of parameters in the same way
(rearrange a random m-permutation of one or two matrices). if same-index
W-G coupling is binding, joint should cost much more perplexity than
u_only at the same layer; if it is just a labeling, the two should be
similar.

note: permuting gate_proj rows AND up_proj rows by the same π would be
a true no-op (it just relabels channels), so to actually break the
pairing we permute G but not W. permuting U by the same π keeps G,U
together; permuting U alone is the control. see docs/swiglu.pdf §III for
the cp decomposition framing.

outputs (under --results_dir):
  pairing_permutation.json
  pairing_permutation.png
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import (
    PALETTE,
    add_common_args,
    compute_perplexity,
    log,
    permuted_layer,
    prepare_run,
)


def plot_pairing_permutation(results, results_dir):
    """plot per-layer perplexity for joint vs u_only, with baseline reference."""
    ppl_baseline = results["baseline"]
    n_layers = len(results["joint"]["mean"])
    layers_x = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(11, 4))

    for cond, color, marker, label in [
        ("joint",  PALETTE["ablation"],  "o", r"joint $\pi_G = \pi_U$ (breaks W-G coupling)"),
        ("u_only", PALETTE["secondary"], "s", r"$\pi_U$ only (control: breaks U pairing)"),
    ]:
        mean = np.array(results[cond]["mean"])
        std = np.array(results[cond]["std"])
        ax.plot(layers_x, mean, marker=marker, color=color, lw=1.5, ms=4,
                label=label)
        ax.fill_between(layers_x, mean - std, mean + std,
                        color=color, alpha=0.15)

    ax.axhline(ppl_baseline, color=PALETTE["primary"], ls="--", lw=0.8,
               label=f"baseline = {ppl_baseline:.1f}")
    ax.set_xlabel("permuted layer")
    ax.set_ylabel("perplexity")
    ax.set_yscale("log")
    ax.set_xticks(layers_x)
    ax.legend(framealpha=0.9, edgecolor="0.8", loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pairing_permutation.png"))
    plt.close()
    log("done", f"saved pairing_permutation.png -> {results_dir}/")


def run_pairing_permutation(model, layers_info, eval_ids, device, results_dir,
                            n_seeds=3, base_seed=0):
    log("info", f"experiment 9: same-index pairing permutation | n_seeds={n_seeds}")

    m = layers_info[0]["gate_proj"].weight.shape[0]
    n_layers = len(layers_info)

    ppl_baseline = compute_perplexity(model, eval_ids, device)
    log("result", f"baseline | perplexity={ppl_baseline:.2f}")

    conds = ("joint", "u_only")
    raw = {c: np.zeros((n_layers, n_seeds)) for c in conds}

    for li in range(n_layers):
        for si in range(n_seeds):
            gen = torch.Generator().manual_seed(base_seed + 1000 * si + li)
            pi = torch.randperm(m, generator=gen)
            pi_u = torch.randperm(m, generator=gen)

            # joint: same π applied to gate rows and down cols
            with permuted_layer(layers_info, li, perm_g=pi, perm_u=pi):
                ppl_joint = compute_perplexity(model, eval_ids, device)
            raw["joint"][li, si] = ppl_joint

            # u_only: independent π_u on down cols, gate untouched
            with permuted_layer(layers_info, li, perm_u=pi_u):
                ppl_u = compute_perplexity(model, eval_ids, device)
            raw["u_only"][li, si] = ppl_u

            log("eval", f"layer {li:02d}/{n_layers-1:02d} | seed {si} | "
                f"joint={ppl_joint:.2e} | u_only={ppl_u:.2e}")

        means = {c: raw[c][li].mean() for c in conds}
        log("result", f"layer {li:02d} | joint_mean={means['joint']:.2e} | "
            f"u_only_mean={means['u_only']:.2e} | "
            f"joint/u_only={means['joint']/means['u_only']:.2f}")

    results = {
        "baseline": ppl_baseline,
        "n_seeds": n_seeds,
        "n_layers": n_layers,
        "joint": {
            "mean": raw["joint"].mean(axis=1).tolist(),
            "std":  raw["joint"].std(axis=1).tolist(),
            "raw":  raw["joint"].tolist(),
        },
        "u_only": {
            "mean": raw["u_only"].mean(axis=1).tolist(),
            "std":  raw["u_only"].std(axis=1).tolist(),
            "raw":  raw["u_only"].tolist(),
        },
    }

    with open(os.path.join(results_dir, "pairing_permutation.json"), "w") as f:
        json.dump(results, f, indent=2)
    log("done", f"saved pairing_permutation.json -> {results_dir}/")

    plot_pairing_permutation(results, results_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--n_seeds", type=int, default=3,
                        help="number of random permutations per layer")
    args = parser.parse_args()

    t0 = time.time()
    ctx = prepare_run(args, capture_activations=False)
    run_pairing_permutation(
        ctx["model"], ctx["layers_info"], ctx["eval_ids"],
        ctx["device"], args.results_dir, n_seeds=args.n_seeds,
        base_seed=args.seed,
    )
    log("done", f"experiment 9 complete | time={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
