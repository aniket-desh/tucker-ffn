#!/usr/bin/env python3
"""experiment 1: decomposition sanity check.

verify the channel decomposition reproduces the forward pass.
reconstructs y = sum_j u_j * c_j(x) from extracted weight matrices and
compares to the actual mlp output captured via hooks. catches any indexing
or convention errors (wrong matrix, transposed, missing bias).

key identity: c_j(x) = h_j(x) because SiLU(z) = z * sigmoid(z), so the
reconstruction y = U^T c = down_proj(h) is exactly what the standard
forward computes.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import (
    add_common_args,
    compute_channel_quantities,
    get_weight_and_bias,
    log,
    prepare_run,
)


def run_sanity_check(layers_info, mlp_inputs, mlp_outputs):
    log("info", "experiment 1: decomposition sanity check")
    all_pass = True

    for info in layers_info:
        idx = info["layer_idx"]
        x = mlp_inputs[idx]
        y_actual = mlp_outputs[idx]
        if y_actual.dim() == 3:
            y_actual = y_actual.squeeze(0)
        y_actual = y_actual.float()

        _, _, _, c = compute_channel_quantities(x, info)

        U, u_bias = get_weight_and_bias(info["down_proj"])
        # y = U^T c, where U^T = down_proj.weight (d, m)
        # batched: y = c @ down_proj.weight^T = c @ U
        y_decomp = c @ U.T
        if u_bias is not None:
            y_decomp = y_decomp + u_bias

        max_err = (y_actual - y_decomp).abs().max().item()
        scale = y_actual.abs().max().item() + 1e-10
        rel_err = max_err / scale
        passed = rel_err < 1e-4

        status = "pass" if passed else "FAIL"
        log("result", f"layer {idx:02d} | max_abs_err={max_err:.2e} | rel_err={rel_err:.2e} | {status}")
        if not passed:
            all_pass = False

    tag = "done" if all_pass else "error"
    msg = "all layers pass decomposition sanity check" if all_pass else "some layers FAILED"
    log(tag, msg)
    return all_pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    ctx = prepare_run(args, capture_activations=True)
    run_sanity_check(ctx["layers_info"], ctx["mlp_inputs"], ctx["mlp_outputs"])


if __name__ == "__main__":
    main()
