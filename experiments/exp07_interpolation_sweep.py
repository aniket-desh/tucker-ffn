#!/usr/bin/env python3
"""experiment 7: interpolation sweep (dose-response).

sweep interpolation strength from normal routing to fully ablated.
uses alpha^(lam)_j(x) = (1-lam)*sigmoid(g_j^T x) + lam*E[alpha_j].
this gives a dose-response curve showing how gradually removing
input-dependent routing degrades model quality.

outputs (under --results_dir):
  interpolation_sweep.json
  interpolation_sweep.png
"""

import argparse
import json
import os
import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import (
    PALETTE,
    add_common_args,
    compute_channel_quantities,
    compute_perplexity,
    interpolated_routing,
    log,
    prepare_run,
)


def plot_interpolation_sweep(lambdas, ppls, results_dir):
    """plot dose-response curve: perplexity vs interpolation strength."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lambdas, ppls, "o-", color=PALETTE["primary"], ms=4, lw=1.5)
    ax.set_xlabel(r"$\lambda$ (0 = normal, 1 = fully ablated)")
    ax.set_ylabel("perplexity")
    ax.set_yscale("log")

    ax.axhline(ppls[0], color=PALETTE["neutral"], ls="--", lw=0.8, alpha=0.5)
    ax.annotate(f"baseline = {ppls[0]:.1f}", xy=(0.02, ppls[0]),
                xytext=(0.12, ppls[0] * 0.6), fontsize=9, color=PALETTE["neutral"])

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "interpolation_sweep.png"))
    plt.close()
    log("done", f"saved interpolation_sweep.png -> {results_dir}/")


def run_interpolation_sweep(model, layers_info, mlp_inputs, eval_ids, device,
                            results_dir, lambdas=None):
    if lambdas is None:
        lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    log("info", f"experiment 7: interpolation sweep | lambdas={lambdas}")

    mean_alphas = {}
    for info in layers_info:
        idx = info["layer_idx"]
        _, _, alpha, _ = compute_channel_quantities(mlp_inputs[idx], info)
        mean_alphas[idx] = alpha.mean(dim=0)

    ppls = []
    for lam in lambdas:
        if lam == 0.0:
            ppl = compute_perplexity(model, eval_ids, device)
        else:
            with interpolated_routing(model, layers_info, lam, mean_alphas):
                ppl = compute_perplexity(model, eval_ids, device)
        ppls.append(ppl)
        log("result", f"lambda={lam:.2f} | perplexity={ppl:.2f}")

    results = {"lambdas": lambdas, "ppls": ppls}
    with open(os.path.join(results_dir, "interpolation_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)
    log("done", f"saved interpolation_sweep.json -> {results_dir}/")

    plot_interpolation_sweep(lambdas, ppls, results_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--lambdas", type=str, default=None,
        help="comma-separated lambda values (default: 0.0..1.0 in 0.1 steps)",
    )
    args = parser.parse_args()

    lambdas = None
    if args.lambdas is not None:
        lambdas = [float(s) for s in args.lambdas.split(",")]

    ctx = prepare_run(args, capture_activations=True)
    run_interpolation_sweep(
        ctx["model"], ctx["layers_info"], ctx["mlp_inputs"],
        ctx["eval_ids"], ctx["device"], args.results_dir, lambdas=lambdas,
    )


if __name__ == "__main__":
    main()
