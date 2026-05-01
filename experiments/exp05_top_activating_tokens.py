#!/usr/bin/env python3
"""experiment 5: top-activating tokens.

find tokens that most strongly activate high-variance routing channels.
for channels with highest Var_x[alpha_j], find tokens where
alpha_j(x) * |c_j(x)| is largest. if the routed-cp decomposition
captures meaningful structure, top-activating tokens for a given channel
should cluster semantically (note section on interpretability).

outputs (under --results_dir):
  top_activating_tokens.json

this experiment depends on per-channel variances. when run standalone it
recomputes them from the captured activations.
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import (
    add_common_args,
    compute_channel_quantities,
    log,
    prepare_run,
)


def run_top_activating(layers_info, mlp_inputs, all_vars, analysis_ids,
                       tokenizer, results_dir, top_k_channels=10,
                       top_k_tokens=15):
    log("info", f"experiment 5: top-activating token analysis | "
        f"top_k_channels={top_k_channels} | top_k_tokens={top_k_tokens}")

    token_ids = analysis_ids.squeeze(0).tolist()
    n_layers = len(layers_info)
    # print detailed output for a few representative layers
    print_layers = {0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1}

    results = []

    for info in layers_info:
        idx = info["layer_idx"]
        _, _, alpha, c = compute_channel_quantities(mlp_inputs[idx], info)

        # score: alpha_j(x) * |c_j(x)|
        score = alpha * c.abs()

        var_j = all_vars[idx]
        top_ch = np.argsort(var_j)[::-1][:top_k_channels]

        layer_data = []
        for ch_idx in top_ch:
            ch = int(ch_idx)
            s = score[:, ch].numpy()
            top_tok = np.argsort(s)[::-1][:top_k_tokens]

            entries = []
            for ti in top_tok:
                ti = int(ti)
                ctx_start = max(0, ti - 3)
                ctx_end = min(len(token_ids), ti + 4)
                tok_str = tokenizer.decode([token_ids[ti]])
                ctx_str = tokenizer.decode(token_ids[ctx_start:ctx_end])
                entries.append({
                    "pos": ti,
                    "token": tok_str,
                    "context": ctx_str,
                    "alpha": float(alpha[ti, ch]),
                    "c_abs": float(c[ti, ch].abs()),
                    "score": float(s[ti]),
                })

            layer_data.append({
                "channel": ch,
                "variance": float(var_j[ch]),
                "top_tokens": entries,
            })

        results.append({"layer": idx, "channels": layer_data})

        if idx in print_layers:
            log("info", f"layer {idx:02d} — top channels by routing variance:")
            for cd in layer_data[:3]:
                toks = [e["token"].strip() or repr(e["token"]) for e in cd["top_tokens"][:8]]
                print(f"  ch {cd['channel']:4d} (var={cd['variance']:.4f}): {toks}")

    with open(os.path.join(results_dir, "top_activating_tokens.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("done", f"saved top_activating_tokens.json -> {results_dir}/")

    return results


def _compute_variances(layers_info, mlp_inputs):
    """recompute per-channel alpha variances from captured activations."""
    all_vars = []
    for info in layers_info:
        _, _, alpha, _ = compute_channel_quantities(mlp_inputs[info["layer_idx"]], info)
        all_vars.append(alpha.var(dim=0).numpy())
    return all_vars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--top_k_channels", type=int, default=10)
    parser.add_argument("--top_k_tokens", type=int, default=15)
    args = parser.parse_args()

    ctx = prepare_run(args, capture_activations=True)
    all_vars = _compute_variances(ctx["layers_info"], ctx["mlp_inputs"])
    run_top_activating(
        ctx["layers_info"], ctx["mlp_inputs"], all_vars,
        ctx["analysis_ids"], ctx["tokenizer"], args.results_dir,
        top_k_channels=args.top_k_channels, top_k_tokens=args.top_k_tokens,
    )


if __name__ == "__main__":
    main()
