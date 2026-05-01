"""act_fn replacements for routing-ablation experiments.

each module is a drop-in for silu(z) = z * sigmoid(z) inside swiglu mlps:
swapping mlp.act_fn temporarily lets us test what happens when the routing
coefficient alpha_j(x) = sigmoid(g_j^T x) is replaced by something simpler.
"""

from contextlib import contextmanager

import torch


# ── constant routing (experiments 4, 6) ─────────────────────────────────────

class ConstantRouting(torch.nn.Module):
    """drop-in replacement for silu that uses a fixed routing coefficient.

    silu(z) = z * sigmoid(z) gets replaced with z * alpha_const,
    stripping out input-dependent routing while preserving the linear
    component of the gate signal.
    """

    def __init__(self, alpha):
        super().__init__()
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = alpha

    def forward(self, z):
        return z * self.alpha


def _resolve_const_alpha(mode, idx, mlp, mean_alphas):
    if mode == "uniform":
        return 0.5
    if mode == "mean":
        return mean_alphas[idx].to(next(mlp.parameters()).device)
    if mode == "ones":
        return 1.0
    raise ValueError(f"unknown ablation mode: {mode}")


@contextmanager
def ablated_routing(model, layers_info, mode, mean_alphas=None):
    """replace input-dependent sigmoid routing with constants in all layers.

    standard swiglu forward:
      y = down_proj( act_fn(gate_proj(x)) * up_proj(x) )
        = down_proj( silu(gate_pre) * up_pre )
        = down_proj( gate_pre * sigmoid(gate_pre) * up_pre )

    ablated forward (replace sigmoid(gate_pre) -> alpha_const):
      y = down_proj( gate_pre * alpha_const * up_pre )

    we achieve this by temporarily swapping mlp.act_fn.  standard act_fn is
    silu(z) = z * sigmoid(z).  we replace with z * alpha_const, which strips
    out the input-dependent routing.

    modes:
      "uniform" — alpha = 0.5 for all channels
      "mean"    — alpha = E_x[sigmoid(g_j^T x)] per channel (from calibration)
      "ones"    — alpha = 1.0 (pure bilinear, no routing at all)
    """
    saved = []

    for info in layers_info:
        mlp = info["mlp"]
        saved.append((mlp, mlp.act_fn))
        idx = info["layer_idx"]
        alpha = _resolve_const_alpha(mode, idx, mlp, mean_alphas)
        mlp.act_fn = ConstantRouting(alpha)

    try:
        yield
    finally:
        for mlp, orig in saved:
            mlp.act_fn = orig


@contextmanager
def ablated_single_layer(model, layers_info, target_layer_idx, mode,
                         mean_alphas=None):
    """replace routing in a single layer, leaving all others intact."""
    info = layers_info[target_layer_idx]
    mlp = info["mlp"]
    saved = mlp.act_fn
    idx = info["layer_idx"]

    alpha = _resolve_const_alpha(mode, idx, mlp, mean_alphas)
    mlp.act_fn = ConstantRouting(alpha)
    try:
        yield
    finally:
        mlp.act_fn = saved


# ── interpolated routing (experiment 7) ─────────────────────────────────────

class InterpolatedRouting(torch.nn.Module):
    """silu with routing interpolated toward a constant.

    computes z * [(1-lam)*sigmoid(z) + lam*alpha_const] instead of z*sigmoid(z).
    at lam=0 this is standard silu; at lam=1 it is fully ablated.
    """

    def __init__(self, lam, alpha_const):
        super().__init__()
        self.lam = lam
        if isinstance(alpha_const, torch.Tensor):
            self.register_buffer("alpha_const", alpha_const)
        else:
            self.alpha_const = alpha_const

    def forward(self, z):
        return z * ((1 - self.lam) * torch.sigmoid(z) + self.lam * self.alpha_const)


@contextmanager
def interpolated_routing(model, layers_info, lam, mean_alphas):
    """interpolate all layers: alpha^(lam) = (1-lam)*sigmoid(z) + lam*E[alpha]."""
    saved = []
    for info in layers_info:
        mlp = info["mlp"]
        saved.append((mlp, mlp.act_fn))
        idx = info["layer_idx"]
        alpha_const = mean_alphas[idx].to(next(mlp.parameters()).device)
        mlp.act_fn = InterpolatedRouting(lam, alpha_const)
    try:
        yield
    finally:
        for mlp, orig in saved:
            mlp.act_fn = orig


# ── channel-subset routing (experiment 8) ───────────────────────────────────

class ChannelSubsetRouting(torch.nn.Module):
    """ablate routing for a specific subset of channels, keep others intact.

    for channels in the ablation mask, replaces sigmoid(z) with a constant.
    for all other channels, keeps standard sigmoid(z).
    """

    def __init__(self, mask, alpha_const):
        super().__init__()
        # mask: bool tensor (m,), True = ablate this channel
        self.register_buffer("mask", mask)
        if isinstance(alpha_const, torch.Tensor):
            self.register_buffer("alpha_const", alpha_const)
        else:
            self.alpha_const = alpha_const

    def forward(self, z):
        sig = torch.sigmoid(z)
        # blend: ablated channels get alpha_const, others keep sigmoid
        alpha = torch.where(self.mask, self.alpha_const, sig)
        return z * alpha


@contextmanager
def channel_subset_ablation(model, layers_info, masks, mean_alphas):
    """ablate routing for specific channel subsets per layer.

    masks: dict layer_idx -> bool tensor (m,), True = ablate
    """
    saved = []
    for info in layers_info:
        mlp = info["mlp"]
        saved.append((mlp, mlp.act_fn))
        idx = info["layer_idx"]
        device = next(mlp.parameters()).device
        mask = masks[idx].to(device)
        alpha_const = mean_alphas[idx].to(device)
        mlp.act_fn = ChannelSubsetRouting(mask, alpha_const)
    try:
        yield
    finally:
        for mlp, orig in saved:
            mlp.act_fn = orig
