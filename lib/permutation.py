"""weight-permutation context manager for the same-index pairing experiment.

unlike lib/routing.py (which swaps mlp.act_fn), this permutes the actual
weight matrices of one swiglu mlp:
  - permuting gate_proj rows by π replaces g_j with g_{π(j)} at hidden index j
  - permuting down_proj columns by π_u replaces u_j with u_{π_u(j)}
  - up_proj (W) is left untouched

joint π = π_u tests breaking the W-G same-index coupling while keeping
G and U paired (the cp diagonal moves together for G,U). π_u alone is
the control: breaks U pairing without disturbing W-G.
"""

from contextlib import contextmanager


@contextmanager
def permuted_layer(layers_info, layer_idx, perm_g=None, perm_u=None):
    """temporarily permute gate_proj rows and/or down_proj cols of one layer.

    args:
        layers_info: list from get_swiglu_layers
        layer_idx:   index into layers_info (not the layer_idx field, but the
                     position in the list — they match for standard models)
        perm_g:      long tensor of size m, permutation of gate_proj rows
        perm_u:      long tensor of size m, permutation of down_proj columns

    bias handling:
        gate_proj.bias (if present) is on the m-dim output, so it permutes
        with the rows of gate_proj.weight.
        down_proj.bias is on the d-dim output, so column permutation of
        down_proj.weight does not touch it.
    """
    info = layers_info[layer_idx]
    gp = info["gate_proj"]
    dp = info["down_proj"]

    saved = {}

    if perm_g is not None:
        perm_g = perm_g.to(gp.weight.device)
        saved["gate_w"] = gp.weight.data.clone()
        gp.weight.data = gp.weight.data[perm_g, :].contiguous()
        if gp.bias is not None:
            saved["gate_b"] = gp.bias.data.clone()
            gp.bias.data = gp.bias.data[perm_g].contiguous()

    if perm_u is not None:
        perm_u = perm_u.to(dp.weight.device)
        saved["down_w"] = dp.weight.data.clone()
        dp.weight.data = dp.weight.data[:, perm_u].contiguous()

    try:
        yield
    finally:
        if "gate_w" in saved:
            gp.weight.data = saved["gate_w"]
        if "gate_b" in saved:
            gp.bias.data = saved["gate_b"]
        if "down_w" in saved:
            dp.weight.data = saved["down_w"]
