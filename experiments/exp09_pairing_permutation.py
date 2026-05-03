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
    """plot per-layer perplexity for all permutation conditions, with baseline
    + optional noop overlay. supports both the legacy (joint, u_only) layout
    and the symmetric (joint, u_only, g_only, w_only) layout."""
    ppl_baseline = results["baseline"]
    conds = results.get("conditions", ["joint", "u_only"])
    n_layers = len(results[conds[0]]["mean"])
    layers_x = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(11, 4))

    floor = ppl_baseline * 0.7
    style = {
        "joint":  (PALETTE["ablation"],  "o",
                   r"joint $\pi_G = \pi_U$ (breaks W-G)"),
        "u_only": (PALETTE["secondary"], "s",
                   r"$\pi_U$ only (breaks U pairings)"),
        "g_only": (PALETTE["accent"],    "^",
                   r"$\pi_G$ only (breaks G pairings)"),
        "w_only": (PALETTE["primary"],   "D",
                   r"$\pi_W$ only (breaks W pairings)"),
    }
    for cond in conds:
        if cond not in style:
            continue
        color, marker, label = style[cond]
        mean = np.array(results[cond]["mean"])
        std = np.array(results[cond]["std"])
        ax.plot(layers_x, mean, marker=marker, color=color, lw=1.5, ms=4,
                label=label)
        ax.fill_between(layers_x, np.maximum(mean - std, floor), mean + std,
                        color=color, alpha=0.12)

    # optional no-op control overlay
    noop_path = os.path.join(results_dir, "noop_control.json")
    if os.path.exists(noop_path):
        with open(noop_path) as f:
            noop = json.load(f)
        noop_means = np.array(noop["noop_per_layer_mean"])
        ax.plot(np.arange(len(noop_means)), noop_means,
                marker="x", ls=":", color=PALETTE["accent"], lw=1.0, ms=4,
                label=r"no-op: permute $\pi_G=\pi_W=\pi_U$ together")

    ax.axhline(ppl_baseline, color=PALETTE["primary"], ls="--", lw=0.8,
               label=f"baseline = {ppl_baseline:.1f}")
    ax.set_xlabel("permuted layer")
    ax.set_ylabel("perplexity")
    ax.set_yscale("log")
    ax.set_xticks(layers_x)
    ax.set_ylim(bottom=floor)
    ax.legend(framealpha=0.95, edgecolor="0.8", loc="lower center",
              ncol=2, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pairing_permutation.png"))
    plt.close()
    log("done", f"saved pairing_permutation.png -> {results_dir}/")


def run_pairing_permutation(model, layers_info, eval_ids, device, results_dir,
                            n_seeds=8, base_seed=0):
    log("info", f"experiment 9: same-index pairing permutation | n_seeds={n_seeds}")

    m = layers_info[0]["gate_proj"].weight.shape[0]
    n_layers = len(layers_info)

    ppl_baseline = compute_perplexity(model, eval_ids, device)
    log("result", f"baseline | perplexity={ppl_baseline:.2f}")

    # four conditions:
    #   joint   permute G and U by same π — breaks W-G coupling, keeps G-U
    #   u_only  permute U alone           — keeps W-G, breaks G-U and W-U
    #   g_only  permute G alone           — breaks W-G and G-U, keeps W-U
    #   w_only  permute W alone           — breaks W-G and W-U, keeps G-U
    # the no-op (g, u, w) joint permutation is exp09b and is exactly baseline.
    conds = ("joint", "u_only", "g_only", "w_only")
    raw = {c: np.zeros((n_layers, n_seeds)) for c in conds}

    for li in range(n_layers):
        for si in range(n_seeds):
            gen = torch.Generator().manual_seed(base_seed + 1000 * si + li)
            pi   = torch.randperm(m, generator=gen)
            pi_u = torch.randperm(m, generator=gen)
            pi_g = torch.randperm(m, generator=gen)
            pi_w = torch.randperm(m, generator=gen)

            # joint: same π applied to gate rows and down cols
            with permuted_layer(layers_info, li, perm_g=pi, perm_u=pi):
                ppl_joint = compute_perplexity(model, eval_ids, device)
            raw["joint"][li, si] = ppl_joint

            # u_only: independent π_u on down cols, gate/up untouched
            with permuted_layer(layers_info, li, perm_u=pi_u):
                ppl_u = compute_perplexity(model, eval_ids, device)
            raw["u_only"][li, si] = ppl_u

            # g_only: independent π_g on gate rows, up/down untouched
            with permuted_layer(layers_info, li, perm_g=pi_g):
                ppl_g = compute_perplexity(model, eval_ids, device)
            raw["g_only"][li, si] = ppl_g

            # w_only: independent π_w on up rows, gate/down untouched
            with permuted_layer(layers_info, li, perm_w=pi_w):
                ppl_w = compute_perplexity(model, eval_ids, device)
            raw["w_only"][li, si] = ppl_w

            log("eval", f"layer {li:02d}/{n_layers-1:02d} | seed {si} | "
                f"joint={ppl_joint:.2e} | u_only={ppl_u:.2e} | "
                f"g_only={ppl_g:.2e} | w_only={ppl_w:.2e}")

        means = {c: raw[c][li].mean() for c in conds}
        log("result",
            f"layer {li:02d} | joint={means['joint']:.2e} | "
            f"u_only={means['u_only']:.2e} | "
            f"g_only={means['g_only']:.2e} | "
            f"w_only={means['w_only']:.2e}")

    results = {
        "baseline": ppl_baseline,
        "n_seeds": n_seeds,
        "n_layers": n_layers,
        "conditions": list(conds),
    }
    for c in conds:
        results[c] = {
            "mean": raw[c].mean(axis=1).tolist(),
            "std":  raw[c].std(axis=1).tolist(),
            "raw":  raw[c].tolist(),
        }

    # per-condition summary stats
    for c in conds:
        m = np.array(results[c]["mean"])
        log("summary", f"{c:8s} | mean={m.mean():.2e} | max={m.max():.2e} "
            f"(layer {int(m.argmax())}) | min={m.min():.2e} "
            f"(layer {int(m.argmin())})")
    # legacy joint/u_only ratio (kept for backward-compatible SUMMARY parsing)
    joint_means = np.array(results["joint"]["mean"])
    u_means = np.array(results["u_only"]["mean"])
    ratio = joint_means / np.maximum(u_means, 1e-30)
    results["ratio_joint_over_u_only"] = ratio.tolist()
    results["geomean_ratio"] = float(np.exp(np.log(ratio).mean()))
    log("summary", f"geomean(joint/u_only) across layers = "
        f"{results['geomean_ratio']:.2e}")

    with open(os.path.join(results_dir, "pairing_permutation.json"), "w") as f:
        json.dump(results, f, indent=2)
    log("done", f"saved pairing_permutation.json -> {results_dir}/")

    plot_pairing_permutation(results, results_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--n_seeds", type=int, default=8,
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
