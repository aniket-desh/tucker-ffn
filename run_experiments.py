#!/usr/bin/env python3
"""
swiglu tensor decomposition analysis.

empirical validation that every swiglu ffn computes an exact input-dependent
cp decomposition of a third-order interaction tensor (note eq 10):

  A(x) = sum_j alpha_j(x) * u_j (x) w_j (x) g_j,   alpha_j(x) = sigmoid(g_j^T x)

the output of each swiglu layer decomposes as:

  y = sum_j alpha_j(x) * u_j * (w_j^T x) * (g_j^T x)

where u_j, w_j, g_j are columns of down_proj, up_proj, gate_proj respectively.
each channel contribution is c_j(x) = alpha_j(x) * (w_j^T x) * (g_j^T x).
see docs/note.pdf for full theory.

experiments:
  1. numerical sanity check — verify decomposition matches forward pass
  2. routing coefficient statistics — alpha_j(x) distributions across layers
  3. channel sparsity — concentration of channel contributions |c_j(x)|
  4. routing ablation — perplexity with alpha replaced by constants (all layers)
  5. top-activating tokens — semantic clustering of high-variance channels
  6. layerwise ablation — ablate routing one layer at a time
  7. interpolation sweep — dose-response from normal to fully ablated routing
  8. channel-subset ablation — ablate high/low variance or contribution channels

note: pythia models use standard gelu mlp, NOT swiglu. default model is
Qwen/Qwen2.5-0.5B which has the required gate_proj / up_proj / down_proj
structure. any llama/mistral/gemma-family model also works.

usage:
  python run_experiments.py
  python run_experiments.py --model Qwen/Qwen2.5-0.5B --max_tokens 4096
  python run_experiments.py --experiments 1,2,3

dependencies: torch, transformers, datasets, numpy, matplotlib
"""

import argparse
import json
import os
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ── plot theme ───────────────────────────────────────────────────────────────
# siam preprint style, modeled after Singh et al. (arXiv:2510.17734).
# no titles, all spines, serif fonts, muted distinguishable colors,
# markers + linestyle variation, light dotted grid, 300 dpi.

_PALETTE = {
    "primary": "#1f77b4",      # steel blue — main data series
    "secondary": "#d4820e",    # dull orange — secondary
    "accent": "#2ca02c",       # muted green — highlights
    "neutral": "#7f7f7f",      # gray — reference lines, annotations
    "ablation": "#c44e52",     # muted red — ablation conditions
    "fill": "#1f77b4",         # same as primary, used with low alpha
    "black": "#2d2d2d",        # near-black for primary emphasis
}

# layer-position colormap: early layers cool, late layers warm
_LAYER_CMAP = "coolwarm"

# ordered cycle for multi-series plots (matches siam convention)
_COLOR_CYCLE = ["#2d2d2d", "#1f77b4", "#d4820e", "#2ca02c", "#c44e52", "#9467bd"]
_MARKER_CYCLE = ["o", "s", "^", "v", "D", "p"]
_LS_CYCLE = ["-", "--", "-.", ":", "-", "--"]


def setup_plot_style():
    """apply siam-style theme: serif fonts, all spines, light grid, 300 dpi."""
    plt.rcParams.update({
        # fonts — match latex computer modern
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif",
                        "Times New Roman", "Times"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        # spines — all four visible (siam standard)
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.6,
        # ticks — inward
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        # grid — light dotted, barely visible
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        # figure
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.05,
        # lines
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
    })


# ── logging ──────────────────────────────────────────────────────────────────

def log(tag, msg):
    """bracketed-tag logging. see docs/format.md for convention."""
    print(f"[{tag}] {msg}")


# ── model and data ───────────────────────────────────────────────────────────

def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(model_name, device):
    log("info", f"loading {model_name}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log("info", f"loaded | params={n_params:.1f}M | device={device} | time={time.time() - t0:.1f}s")
    return model, tokenizer


def get_swiglu_layers(model):
    """find swiglu mlp layers in the model.

    expects llama / qwen2 / mistral / gemma style architecture where each
    transformer block has an mlp with gate_proj, up_proj, down_proj.

    weight convention (matching note eqs 1-2):
      gate_proj.weight  (m, d)  — rows are g_j^T   (gate directions)
      up_proj.weight    (m, d)  — rows are w_j^T   (up-projection directions)
      down_proj.weight  (d, m)  — columns are u_j   (output directions)
    """
    layers = []

    model_base = getattr(model, "model", None)
    if model_base is not None and hasattr(model_base, "layers"):
        for i, block in enumerate(model_base.layers):
            mlp = block.mlp
            if all(hasattr(mlp, a) for a in ("gate_proj", "up_proj", "down_proj")):
                layers.append({
                    "layer_idx": i,
                    "mlp": mlp,
                    "gate_proj": mlp.gate_proj,
                    "up_proj": mlp.up_proj,
                    "down_proj": mlp.down_proj,
                })

    if not layers:
        raise ValueError(
            "no swiglu layers found. this model may not use swiglu architecture. "
            "pythia uses gelu mlp, not swiglu. "
            "try Qwen/Qwen2.5-0.5B or a llama-family model."
        )

    d = layers[0]["gate_proj"].weight.shape[1]
    m = layers[0]["gate_proj"].weight.shape[0]
    log("info", f"found {len(layers)} swiglu layers | d_model={d} | d_intermediate={m}")
    return layers


def load_text_data(tokenizer, max_tokens=4096):
    """load wikitext-2 test split, return two non-overlapping token chunks.

    first chunk is for analysis / calibration (experiments 1-3, 5).
    second chunk is held out for ablation evaluation (experiment 4).
    """
    log("info", "loading wikitext-2-raw-v1 test split")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join([t for t in ds["text"] if t.strip()])
    all_ids = tokenizer.encode(text)
    log("data", f"total_tokens_available={len(all_ids)}")

    n = min(max_tokens, len(all_ids) // 2)
    analysis_ids = torch.tensor(all_ids[:n], dtype=torch.long).unsqueeze(0)
    eval_ids = torch.tensor(all_ids[n : 2 * n], dtype=torch.long).unsqueeze(0)
    log("data", f"analysis_tokens={analysis_ids.shape[1]} | eval_tokens={eval_ids.shape[1]}")
    return analysis_ids, eval_ids


# ── activation capture ───────────────────────────────────────────────────────

def capture_mlp_io(model, input_ids, layers_info, device):
    """run forward pass, capture each mlp layer's input and output via hooks."""
    captured_in = {}
    captured_out = {}
    hooks = []

    for info in layers_info:
        idx = info["layer_idx"]

        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                captured_in[layer_idx] = inp[0].detach().cpu()
                # some models return tuples from mlp forward
                if isinstance(out, tuple):
                    captured_out[layer_idx] = out[0].detach().cpu()
                else:
                    captured_out[layer_idx] = out.detach().cpu()
            return hook_fn

        h = info["mlp"].register_forward_hook(make_hook(idx))
        hooks.append(h)

    with torch.no_grad():
        model(input_ids.to(device))

    for h in hooks:
        h.remove()

    return captured_in, captured_out


# ── weight extraction helpers ────────────────────────────────────────────────

def get_weight_and_bias(linear):
    """extract weight and optional bias from nn.Linear, moved to cpu."""
    w = linear.weight.detach().cpu().float()
    b = linear.bias.detach().cpu().float() if linear.bias is not None else None
    return w, b


def compute_channel_quantities(x, info):
    """compute the cp decomposition quantities for one layer.

    from note eq (4):
      h_j(x) = (w_j^T x) * SiLU(g_j^T x) = (w_j^T x)(g_j^T x) * sigma(g_j^T x)

    so the channel contribution c_j(x) = alpha_j(x) * (w_j^T x) * (g_j^T x)
    equals the hidden activation h_j(x), and the routing coefficient is
    alpha_j(x) = sigmoid(g_j^T x).

    args:
        x: mlp input, shape (1, seq_len, d) or (seq_len, d)
        info: layer info dict from get_swiglu_layers

    returns:
        gate_pre  (seq_len, m)  — g_j^T x
        up_pre    (seq_len, m)  — w_j^T x
        alpha     (seq_len, m)  — sigmoid(g_j^T x)
        c         (seq_len, m)  — channel contributions c_j(x)
    """
    if x.dim() == 3:
        x = x.squeeze(0)
    x = x.float()

    G, g_bias = get_weight_and_bias(info["gate_proj"])
    W, w_bias = get_weight_and_bias(info["up_proj"])

    gate_pre = x @ G.T
    if g_bias is not None:
        gate_pre = gate_pre + g_bias

    up_pre = x @ W.T
    if w_bias is not None:
        up_pre = up_pre + w_bias

    alpha = torch.sigmoid(gate_pre)
    c = alpha * gate_pre * up_pre

    return gate_pre, up_pre, alpha, c


# ── experiment 1: decomposition sanity check ─────────────────────────────────

def run_sanity_check(layers_info, mlp_inputs, mlp_outputs):
    """verify the channel decomposition reproduces the forward pass.

    reconstructs y = sum_j u_j * c_j(x) from extracted weight matrices
    and compares to the actual mlp output captured via hooks. this catches
    any indexing or convention errors (wrong matrix, transposed, missing bias).

    the key identity: c_j(x) = h_j(x) because SiLU(z) = z * sigmoid(z),
    so the reconstruction is y = U^T c = down_proj(h), which is exactly
    what the standard forward computes.
    """
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


# ── experiment 2: routing statistics ─────────────────────────────────────────

def plot_routing_stats(variances, means, hists, results_dir):
    """generate all routing statistics plots from precomputed arrays.

    args:
        variances: (n_layers, m) array of per-channel variances
        means:     (n_layers, m) array of per-channel means
        hists:     list of (hist_counts, bin_edges) tuples per layer
        results_dir: output directory
    """
    n_layers = variances.shape[0]

    # ── plot 1: alpha distribution for representative layers ──
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    picks = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
    cmap = plt.get_cmap(_LAYER_CMAP)
    layer_colors = [cmap(i / max(n_layers - 1, 1)) for i in range(n_layers)]

    for ax, li in zip(axes.flat, picks):
        h, e = hists[li]
        centers = (e[:-1] + e[1:]) / 2
        ax.fill_between(centers, h, alpha=0.25, color=layer_colors[li])
        ax.plot(centers, h, color=layer_colors[li], lw=1.5)
        ax.set_xlabel(r"$\alpha_j(x)$")
        ax.set_ylabel("count")
        ax.set_xlim(0, 1)
        mean_a = means[li].mean()
        ax.axvline(mean_a, color=_PALETTE["neutral"], ls="--", lw=0.8,
                   label=f"layer {li}, mean = {mean_a:.2f}")
        ax.legend(framealpha=0.9, edgecolor="0.8")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "alpha_distribution_by_layer.png"))
    plt.close()
    log("done", f"saved alpha_distribution_by_layer.png -> {results_dir}/")

    # ── plot 2: per-channel variance heatmap across layers ──
    sort_idx = variances.mean(axis=0).argsort()[::-1]
    n_show = min(200, variances.shape[1])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(variances[:, sort_idx[:n_show]], aspect="auto",
                   cmap="viridis", interpolation="nearest")
    ax.set_xlabel(f"channel (sorted by mean variance, top {n_show})")
    ax.set_ylabel("layer")
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("variance")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "routing_variance_heatmap.png"))
    plt.close()
    log("done", f"saved routing_variance_heatmap.png -> {results_dir}/")

    # ── plot 3: mean variance by layer (with iqr band) ──
    mean_var_per_layer = variances.mean(axis=1)
    q25 = np.percentile(variances, 25, axis=1)
    q75 = np.percentile(variances, 75, axis=1)
    layers_x = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.fill_between(layers_x, q25, q75, alpha=0.15, color=_PALETTE["primary"],
                    label="25th\u201375th percentile")
    ax.plot(layers_x, mean_var_per_layer, "o-", color=_PALETTE["primary"],
            ms=4, lw=1.5, label="mean")
    ax.set_xlabel("layer")
    ax.set_ylabel(r"$\mathrm{Var}_x[\alpha_j(x)]$")
    ax.legend(framealpha=0.9, edgecolor="0.8")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "routing_variance_by_layer.png"))
    plt.close()
    log("done", f"saved routing_variance_by_layer.png -> {results_dir}/")


def run_routing_stats(layers_info, mlp_inputs, results_dir):
    """compute and plot routing coefficient statistics.

    alpha_j(x) = sigmoid(g_j^T x) is the per-channel routing coefficient.
    per-channel variance s_j = Var_x[alpha_j(x)] measures whether channel j
    is a static bilinear atom (low s_j) or a genuine input-dependent router
    (high s_j). this is the core empirical observable from the routed-cp
    picture (note section on interpretability).
    """
    log("info", "experiment 2: routing coefficient statistics")

    n_layers = len(layers_info)
    all_vars = []
    all_means = []
    layer_hists = []

    for info in layers_info:
        idx = info["layer_idx"]
        _, _, alpha, _ = compute_channel_quantities(mlp_inputs[idx], info)

        var_j = alpha.var(dim=0).numpy()
        mean_j = alpha.mean(dim=0).numpy()
        all_vars.append(var_j)
        all_means.append(mean_j)

        hist, edges = np.histogram(alpha.numpy().ravel(), bins=100, range=(0, 1))
        layer_hists.append((hist, edges))

        log("eval", f"layer {idx:02d} | mean_alpha={mean_j.mean():.3f} | "
            f"mean_var={var_j.mean():.2e} | max_var={var_j.max():.2e}")

    # save numerical results (including histograms for plot-only regeneration)
    variances = np.stack(all_vars)
    means = np.stack(all_means)
    hist_counts = np.stack([h for h, _ in layer_hists])
    hist_edges = layer_hists[0][1]  # all layers share the same bin edges
    np.savez(
        os.path.join(results_dir, "routing_stats.npz"),
        variances=variances,
        means=means,
        hist_counts=hist_counts,
        hist_edges=hist_edges,
    )
    log("done", f"saved routing_stats.npz -> {results_dir}/")

    plot_routing_stats(variances, means, layer_hists, results_dir)

    return all_vars, all_means


# ── experiment 3: channel sparsity ───────────────────────────────────────────

def plot_channel_sparsity(frac_90_mean, frac_90_std, eff_channels, total_channels,
                          results_dir):
    """generate channel sparsity plots from precomputed arrays."""
    n_layers = len(frac_90_mean)
    layers_x = np.arange(n_layers)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax1.fill_between(layers_x,
                     frac_90_mean - frac_90_std,
                     frac_90_mean + frac_90_std,
                     alpha=0.15, color=_PALETTE["primary"])
    ax1.plot(layers_x, frac_90_mean, "o-", color=_PALETTE["primary"], ms=4, lw=1.5)
    ax1.set_xlabel("layer")
    ax1.set_ylabel("fraction of channels")
    ax1.set_ylim(0, 1)
    ax1.axhline(0.5, color=_PALETTE["neutral"], ls=":", lw=0.8, alpha=0.5)
    ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes, va="top")

    ax2.plot(layers_x, eff_channels, "s-", color=_PALETTE["accent"], ms=4, lw=1.5)
    ax2.axhline(total_channels, color=_PALETTE["neutral"], ls="--", lw=0.8, alpha=0.5,
                label=f"total = {total_channels}")
    ax2.set_xlabel("layer")
    ax2.set_ylabel("effective channels")
    ax2.legend(framealpha=0.9, edgecolor="0.8", loc="lower left")
    ax2.text(0.02, 0.95, "(b)", transform=ax2.transAxes, va="top")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "channel_sparsity.png"))
    plt.close()
    log("done", f"saved channel_sparsity.png -> {results_dir}/")


def run_channel_sparsity(layers_info, mlp_inputs, results_dir):
    """measure how concentrated channel contributions are per token.

    c_j(x) = alpha_j(x) * (w_j^T x) * (g_j^T x) is the scalar channel
    contribution (note eq 4). for each token we ask: how many channels
    carry most of the signal?

    two metrics:
      - frac_90: fraction of channels needed for 90% of sum_j |c_j(x)|
      - eff_channels: exp(entropy) of normalized |c_j(x)|, the effective
        number of active channels (= m if uniform, << m if sparse)
    """
    log("info", "experiment 3: channel sparsity and concentration")

    n_layers = len(layers_info)
    m = layers_info[0]["gate_proj"].weight.shape[0]
    frac_90_means = []
    frac_90_stds = []
    eff_ch_means = []

    for info in layers_info:
        idx = info["layer_idx"]
        _, _, _, c = compute_channel_quantities(mlp_inputs[idx], info)
        c_abs = c.abs()

        # fraction of channels for 90% of total |c_j|
        sorted_c, _ = c_abs.sort(dim=1, descending=True)
        cum = sorted_c.cumsum(dim=1)
        total = c_abs.sum(dim=1, keepdim=True)
        above_90 = cum >= 0.9 * total
        # argmax on bool gives first True; +1 because we want count not index
        first_above = above_90.float().argmax(dim=1) + 1
        frac_90 = first_above.float() / m

        # effective channel count: exp(entropy) of normalized |c_j|
        p = c_abs / (total + 1e-10)
        entropy = -(p * (p + 1e-10).log()).sum(dim=1)
        eff_channels = entropy.exp()

        f90_mean = frac_90.mean().item()
        f90_std = frac_90.std().item()
        eff_mean = eff_channels.mean().item()
        frac_90_means.append(f90_mean)
        frac_90_stds.append(f90_std)
        eff_ch_means.append(eff_mean)

        log("eval", f"layer {idx:02d} | frac_for_90pct={f90_mean:.3f} | eff_channels={eff_mean:.1f}/{m}")

    # save numerical
    frac_arr = np.array(frac_90_means)
    std_arr = np.array(frac_90_stds)
    eff_arr = np.array(eff_ch_means)
    np.savez(
        os.path.join(results_dir, "channel_sparsity_stats.npz"),
        frac_90_mean=frac_arr,
        frac_90_std=std_arr,
        eff_channels=eff_arr,
        total_channels=m,
    )
    log("done", f"saved channel_sparsity_stats.npz -> {results_dir}/")

    plot_channel_sparsity(frac_arr, std_arr, eff_arr, m, results_dir)

    return frac_90_means, eff_ch_means


# ── experiment 4: routing ablation ───────────────────────────────────────────

class _ConstantRouting(torch.nn.Module):
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


@contextmanager
def ablated_routing(model, layers_info, mode, mean_alphas=None):
    """replace input-dependent sigmoid routing with constants.

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
        if mode == "uniform":
            alpha = 0.5
        elif mode == "mean":
            alpha = mean_alphas[idx].to(next(mlp.parameters()).device)
        elif mode == "ones":
            alpha = 1.0
        else:
            raise ValueError(f"unknown ablation mode: {mode}")

        mlp.act_fn = _ConstantRouting(alpha)

    try:
        yield
    finally:
        for mlp, orig in saved:
            mlp.act_fn = orig


def compute_perplexity(model, input_ids, device):
    """compute perplexity via teacher-forced cross-entropy loss."""
    with torch.no_grad():
        out = model(input_ids.to(device), labels=input_ids.to(device))
    return torch.exp(out.loss).item()


def plot_ablation(results, results_dir):
    """generate ablation bar chart from results dict."""
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["baseline", r"$\alpha=0.5$" + "\n(uniform)",
             r"$\alpha=\mathbb{E}[\alpha_j]$" + "\n(per-ch mean)",
             r"$\alpha=1$" + "\n(bilinear)"]
    ppls = [results["baseline"], results["uniform"], results["mean"], results["ones"]]
    colors = [_PALETTE["primary"]] + [_PALETTE["ablation"]] * 3

    bars = ax.bar(names, ppls, color=colors, alpha=0.85, width=0.55,
                  edgecolor="0.3", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_ylabel("perplexity")

    for bar, val in zip(bars, ppls):
        label = f"{val:.1f}" if val < 1000 else f"{val:.1e}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.3, label,
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim(top=max(ppls) * 8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ablation_perplexity.png"))
    plt.close()
    log("done", f"saved ablation_perplexity.png -> {results_dir}/")


def run_ablation(model, layers_info, mlp_inputs, eval_ids, device, results_dir):
    """measure perplexity when routing coefficients are replaced by constants.

    this directly tests whether the input-dependent routing alpha_j(x) =
    sigmoid(g_j^T x) carries meaningful information beyond what a static
    bilinear interaction provides. large perplexity increase = routing
    matters; small increase = the gate mostly acts as a fixed scale.
    """
    log("info", "experiment 4: routing ablation")

    # calibrate: compute per-channel mean alpha from analysis data
    mean_alphas = {}
    for info in layers_info:
        idx = info["layer_idx"]
        _, _, alpha, _ = compute_channel_quantities(mlp_inputs[idx], info)
        mean_alphas[idx] = alpha.mean(dim=0)  # (m,)

    # baseline
    ppl_baseline = compute_perplexity(model, eval_ids, device)
    log("result", f"baseline | perplexity={ppl_baseline:.2f}")

    results = {"baseline": ppl_baseline}

    modes = [
        ("uniform", "alpha=0.5"),
        ("mean", "alpha=E[alpha_j]"),
        ("ones", "alpha=1.0 (pure bilinear)"),
    ]

    for mode, desc in modes:
        with ablated_routing(model, layers_info, mode, mean_alphas):
            ppl = compute_perplexity(model, eval_ids, device)
        delta = ppl - ppl_baseline
        log("result", f"{desc} | perplexity={ppl:.2f} | delta={delta:+.2f}")
        results[mode] = ppl

    # save
    with open(os.path.join(results_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log("done", f"saved ablation_results.json -> {results_dir}/")

    plot_ablation(results, results_dir)

    return results


# ── experiment 6: layerwise routing ablation ─────────────────────────────────

@contextmanager
def ablated_single_layer(model, layers_info, target_layer_idx, mode,
                         mean_alphas=None):
    """replace routing in a single layer, leaving all others intact."""
    info = layers_info[target_layer_idx]
    mlp = info["mlp"]
    saved = mlp.act_fn
    idx = info["layer_idx"]

    if mode == "mean":
        alpha = mean_alphas[idx].to(next(mlp.parameters()).device)
    elif mode == "uniform":
        alpha = 0.5
    else:
        alpha = 1.0

    mlp.act_fn = _ConstantRouting(alpha)
    try:
        yield
    finally:
        mlp.act_fn = saved


def plot_layerwise_ablation(results, results_dir):
    """plot per-layer perplexity delta for each ablation mode side by side."""
    ppl_baseline = results["baseline"]
    modes = [k for k in results if k not in ("baseline",)]
    n_layers = len(results[modes[0]])

    layers_x = np.arange(n_layers)
    n_modes = len(modes)
    bar_width = 0.8 / n_modes

    mode_style = {
        "mean":    {"color": _PALETTE["ablation"], "label": r"$\alpha = \mathbb{E}[\alpha_j]$"},
        "uniform": {"color": _PALETTE["secondary"], "label": r"$\alpha = 0.5$"},
        "ones":    {"color": _PALETTE["primary"],   "label": r"$\alpha = 1$ (bilinear)"},
    }

    fig, ax = plt.subplots(figsize=(11, 4))
    for i, mode in enumerate(modes):
        layer_ppls = results[mode]
        deltas = np.array([layer_ppls.get(j, layer_ppls.get(str(j))) - ppl_baseline
                           for j in range(n_layers)])
        offset = (i - (n_modes - 1) / 2) * bar_width
        style = mode_style.get(mode, {"color": _PALETTE["neutral"], "label": mode})
        ax.bar(layers_x + offset, deltas, width=bar_width, alpha=0.85,
               color=style["color"], edgecolor="0.3", linewidth=0.3,
               label=style["label"])

    ax.set_xlabel("layer")
    ax.set_ylabel(r"$\Delta$ perplexity")
    ax.set_yscale("symlog", linthresh=1)
    ax.axhline(0, color=_PALETTE["neutral"], lw=0.6)
    ax.legend(framealpha=0.9, edgecolor="0.8")
    ax.set_xticks(layers_x)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "layerwise_ablation.png"))
    plt.close()
    log("done", f"saved layerwise_ablation.png -> {results_dir}/")


def run_layerwise_ablation(model, layers_info, mlp_inputs, eval_ids, device,
                           results_dir):
    """ablate routing one layer at a time under three constant-alpha regimes.

    for each layer independently, replaces alpha_j(x) with a constant while
    keeping all other layers intact. runs three modes:
      - mean:    alpha = E_x[alpha_j(x)] per channel (calibrated)
      - uniform: alpha = 0.5
      - ones:    alpha = 1.0 (pure bilinear)

    the perplexity delta per layer reveals which layers depend most on
    input-dependent routing vs static bilinear interaction, and how the
    sensitivity profile differs across ablation types.
    """
    log("info", "experiment 6: layerwise routing ablation (3 modes)")

    # calibrate mean alphas
    mean_alphas = {}
    for info in layers_info:
        idx = info["layer_idx"]
        _, _, alpha, _ = compute_channel_quantities(mlp_inputs[idx], info)
        mean_alphas[idx] = alpha.mean(dim=0)

    ppl_baseline = compute_perplexity(model, eval_ids, device)
    log("result", f"baseline | perplexity={ppl_baseline:.2f}")

    n_layers = len(layers_info)
    ablation_modes = [
        ("mean",    "E[alpha_j]"),
        ("uniform", "alpha=0.5"),
        ("ones",    "alpha=1.0"),
    ]

    results = {"baseline": ppl_baseline}

    for mode, desc in ablation_modes:
        layer_ppls = {}
        log("info", f"  mode: {desc}")
        for li in range(n_layers):
            with ablated_single_layer(model, layers_info, li, mode, mean_alphas):
                ppl = compute_perplexity(model, eval_ids, device)
            delta = ppl - ppl_baseline
            layer_ppls[li] = ppl
            log("result", f"  layer {li:02d}/{n_layers-1:02d} | {desc} | "
                f"perplexity={ppl:.2f} | delta={delta:+.2f}")
        results[mode] = layer_ppls

    with open(os.path.join(results_dir, "layerwise_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)
    log("done", f"saved layerwise_ablation.json -> {results_dir}/")

    plot_layerwise_ablation(results, results_dir)
    return results


# ── experiment 7: interpolation sweep ────────────────────────────────────────

class _InterpolatedRouting(torch.nn.Module):
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
        mlp.act_fn = _InterpolatedRouting(lam, alpha_const)
    try:
        yield
    finally:
        for mlp, orig in saved:
            mlp.act_fn = orig


def plot_interpolation_sweep(lambdas, ppls, results_dir):
    """plot dose-response curve: perplexity vs interpolation strength."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lambdas, ppls, "o-", color=_PALETTE["primary"], ms=4, lw=1.5)
    ax.set_xlabel(r"$\lambda$ (0 = normal, 1 = fully ablated)")
    ax.set_ylabel("perplexity")
    ax.set_yscale("log")

    ax.axhline(ppls[0], color=_PALETTE["neutral"], ls="--", lw=0.8, alpha=0.5)
    ax.annotate(f"baseline = {ppls[0]:.1f}", xy=(0.02, ppls[0]),
                xytext=(0.12, ppls[0] * 0.6), fontsize=9, color=_PALETTE["neutral"])

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "interpolation_sweep.png"))
    plt.close()
    log("done", f"saved interpolation_sweep.png -> {results_dir}/")


def run_interpolation_sweep(model, layers_info, mlp_inputs, eval_ids, device,
                            results_dir,
                            lambdas=None):
    """sweep interpolation strength from normal routing to fully ablated.

    uses alpha^(lam)_j(x) = (1-lam)*sigmoid(g_j^T x) + lam*E[alpha_j].
    this gives a dose-response curve showing how gradually removing
    input-dependent routing degrades model quality.
    """
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


# ── experiment 8: channel-subset ablation ────────────────────────────────────

class _ChannelSubsetRouting(torch.nn.Module):
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
        mlp.act_fn = _ChannelSubsetRouting(mask, alpha_const)
    try:
        yield
    finally:
        for mlp, orig in saved:
            mlp.act_fn = orig


def plot_channel_subset_ablation(results, results_dir):
    """bar chart comparing perplexity under different channel-subset ablations."""
    fig, ax = plt.subplots(figsize=(8, 4))

    conditions = list(results.keys())
    ppls = [results[c] for c in conditions]

    # color: baseline=primary, ablation conditions=cycle
    n = len(conditions)
    colors = [_PALETTE["primary"]] + [_COLOR_CYCLE[i % len(_COLOR_CYCLE)]
                                       for i in range(1, n)]

    bars = ax.bar(range(n), ppls, color=colors, alpha=0.85,
                  edgecolor="0.3", linewidth=0.4)
    ax.set_xticks(range(n))
    ax.set_xticklabels(conditions, fontsize=9, rotation=15, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("perplexity")

    for bar, val in zip(bars, ppls):
        label = f"{val:.1f}" if val < 1000 else f"{val:.1e}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.4, label,
                ha="center", va="bottom", fontsize=8)

    ax.set_ylim(top=max(ppls) * 8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "channel_subset_ablation.png"))
    plt.close()
    log("done", f"saved channel_subset_ablation.png -> {results_dir}/")


def run_channel_subset_ablation(model, layers_info, mlp_inputs, eval_ids,
                                device, results_dir):
    """ablate routing for specific channel subsets to test which channels matter.

    conditions:
      - top 10% by variance (high-variance = most input-dependent routing)
      - bottom 10% by variance (low-variance = near-static atoms)
      - top 10% by mean |c_j| (highest-contribution channels)
      - bottom 90% by variance (freeze the boring majority)

    this directly tests whether the small minority of high-variance channels
    carries most of the routing signal, as the cp decomposition suggests.
    """
    log("info", "experiment 8: channel-subset ablation")

    # compute per-channel statistics from analysis data
    mean_alphas = {}
    channel_vars = {}
    channel_mean_abs_c = {}
    m = layers_info[0]["gate_proj"].weight.shape[0]

    for info in layers_info:
        idx = info["layer_idx"]
        _, _, alpha, c = compute_channel_quantities(mlp_inputs[idx], info)
        mean_alphas[idx] = alpha.mean(dim=0)
        channel_vars[idx] = alpha.var(dim=0).numpy()
        channel_mean_abs_c[idx] = c.abs().mean(dim=0).numpy()

    ppl_baseline = compute_perplexity(model, eval_ids, device)
    log("result", f"baseline | perplexity={ppl_baseline:.2f}")

    k_top = max(1, m // 10)       # top 10%
    k_bot = max(1, m // 10)       # bottom 10%
    k_majority = m - k_top        # bottom 90%

    # build masks for each condition
    conditions = {}

    def make_masks(selector_fn, description):
        masks = {}
        for info in layers_info:
            idx = info["layer_idx"]
            mask = torch.zeros(m, dtype=torch.bool)
            indices = selector_fn(idx)
            mask[indices] = True
            masks[idx] = mask
        n_ablated = int(mask.sum())
        log("info", f"  {description} | channels_ablated={n_ablated}/{m}")
        return masks

    # condition 1: ablate top 10% by variance
    conditions["top 10%\nby variance"] = make_masks(
        lambda idx: np.argsort(channel_vars[idx])[-k_top:],
        "top 10% by variance",
    )

    # condition 2: ablate bottom 10% by variance
    conditions["bottom 10%\nby variance"] = make_masks(
        lambda idx: np.argsort(channel_vars[idx])[:k_bot],
        "bottom 10% by variance",
    )

    # condition 3: ablate top 10% by mean |c_j|
    conditions["top 10%\nby |c_j|"] = make_masks(
        lambda idx: np.argsort(channel_mean_abs_c[idx])[-k_top:],
        "top 10% by mean |c_j|",
    )

    # condition 4: ablate bottom 90% by variance (keep only the routers)
    conditions["bottom 90%\nby variance"] = make_masks(
        lambda idx: np.argsort(channel_vars[idx])[:k_majority],
        "bottom 90% by variance",
    )

    results = {"baseline": ppl_baseline}

    for name, masks in conditions.items():
        with channel_subset_ablation(model, layers_info, masks, mean_alphas):
            ppl = compute_perplexity(model, eval_ids, device)
        delta = ppl - ppl_baseline
        log("result", f"{name.replace(chr(10), ' ')} | perplexity={ppl:.2f} | "
            f"delta={delta:+.2f}")
        results[name] = ppl

    with open(os.path.join(results_dir, "channel_subset_ablation.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("done", f"saved channel_subset_ablation.json -> {results_dir}/")

    plot_channel_subset_ablation(results, results_dir)
    return results


# ── experiment 5: top-activating tokens ──────────────────────────────────────

def run_top_activating(layers_info, mlp_inputs, all_vars, analysis_ids,
                       tokenizer, results_dir, top_k_channels=10,
                       top_k_tokens=15):
    """find tokens that most strongly activate high-variance routing channels.

    for channels with highest Var_x[alpha_j], we find the tokens where
    alpha_j(x) * |c_j(x)| is largest. if the routed-cp decomposition
    captures meaningful structure, top-activating tokens for a given
    channel should cluster semantically (note section on interpretability).
    """
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


# ── plot-only mode ───────────────────────────────────────────────────────────

def replot_from_saved(exps, results_dir):
    """regenerate plots from saved .npz / .json files without loading a model."""
    t0 = time.time()

    if 2 in exps:
        path = os.path.join(results_dir, "routing_stats.npz")
        if os.path.exists(path):
            data = np.load(path)
            variances = data["variances"]
            means = data["means"]
            # reconstruct histogram tuples
            if "hist_counts" in data and "hist_edges" in data:
                hist_counts = data["hist_counts"]
                hist_edges = data["hist_edges"]
                hists = [(hist_counts[i], hist_edges) for i in range(len(hist_counts))]
            else:
                log("error", "routing_stats.npz missing histogram data — rerun experiment 2 to regenerate")
                hists = None
            if hists is not None:
                plot_routing_stats(variances, means, hists, results_dir)
        else:
            log("error", f"routing_stats.npz not found in {results_dir}/")

    if 3 in exps:
        path = os.path.join(results_dir, "channel_sparsity_stats.npz")
        if os.path.exists(path):
            data = np.load(path)
            total_ch = int(data["total_channels"]) if "total_channels" in data else 4864
            plot_channel_sparsity(
                data["frac_90_mean"], data["frac_90_std"],
                data["eff_channels"], total_ch, results_dir,
            )
        else:
            log("error", f"channel_sparsity_stats.npz not found in {results_dir}/")

    if 4 in exps:
        path = os.path.join(results_dir, "ablation_results.json")
        if os.path.exists(path):
            with open(path) as f:
                results = json.load(f)
            plot_ablation(results, results_dir)
        else:
            log("error", f"ablation_results.json not found in {results_dir}/")

    if 6 in exps:
        path = os.path.join(results_dir, "layerwise_ablation.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            plot_layerwise_ablation(data, results_dir)
        else:
            log("error", f"layerwise_ablation.json not found in {results_dir}/")

    if 7 in exps:
        path = os.path.join(results_dir, "interpolation_sweep.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            plot_interpolation_sweep(data["lambdas"], data["ppls"], results_dir)
        else:
            log("error", f"interpolation_sweep.json not found in {results_dir}/")

    if 8 in exps:
        path = os.path.join(results_dir, "channel_subset_ablation.json")
        if os.path.exists(path):
            with open(path) as f:
                results = json.load(f)
            plot_channel_subset_ablation(results, results_dir)
        else:
            log("error", f"channel_subset_ablation.json not found in {results_dir}/")

    log("done", f"plots regenerated | time={time.time() - t0:.1f}s")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="swiglu tensor decomposition analysis")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B",
        help="huggingface model name (must use swiglu architecture)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096,
        help="max tokens per data chunk (analysis and eval)",
    )
    parser.add_argument(
        "--experiments", type=str, default="1,2,3,4,5",
        help="comma-separated experiment numbers to run (default: all)",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="directory for output files",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="device: cpu / mps / cuda (default: auto-detect)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot_only", action="store_true",
        help="regenerate plots from saved results (no model loading)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)
    exps = set(int(e) for e in args.experiments.split(","))
    setup_plot_style()

    # ── plot-only mode: regenerate from saved data, no model needed ──
    if args.plot_only:
        log("info", f"plot-only mode | results_dir={args.results_dir}")
        replot_from_saved(exps, args.results_dir)
        return

    device = args.device or detect_device()

    log("info", "swiglu tensor decomposition analysis")
    log("info", f"model={args.model} | max_tokens={args.max_tokens} | device={device}")
    log("info", f"experiments={sorted(exps)} | results_dir={args.results_dir}")
    print()

    # ── setup ──
    t0_total = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    layers_info = get_swiglu_layers(model)
    analysis_ids, eval_ids = load_text_data(tokenizer, args.max_tokens)
    print()

    # ── capture mlp activations ──
    log("info", "running forward pass to capture mlp activations")
    t0 = time.time()
    mlp_inputs, mlp_outputs = capture_mlp_io(model, analysis_ids, layers_info, device)
    log("info", f"activations captured | n_layers={len(mlp_inputs)} | time={time.time() - t0:.1f}s")
    print()

    # ── experiment 1 ──
    if 1 in exps:
        run_sanity_check(layers_info, mlp_inputs, mlp_outputs)
        print()

    # ── experiment 2 (also needed by experiment 5) ──
    all_vars = None
    all_means = None
    if 2 in exps or 5 in exps:
        all_vars, all_means = run_routing_stats(layers_info, mlp_inputs, args.results_dir)
        print()

    # ── experiment 3 ──
    if 3 in exps:
        run_channel_sparsity(layers_info, mlp_inputs, args.results_dir)
        print()

    # ── experiment 4 ──
    if 4 in exps:
        run_ablation(model, layers_info, mlp_inputs, eval_ids, device, args.results_dir)
        print()

    # ── experiment 5 ──
    if 5 in exps:
        if all_vars is None:
            all_vars, all_means = run_routing_stats(
                layers_info, mlp_inputs, args.results_dir,
            )
        run_top_activating(
            layers_info, mlp_inputs, all_vars, analysis_ids,
            tokenizer, args.results_dir,
        )
        print()

    # ── experiment 6: layerwise ablation ──
    if 6 in exps:
        run_layerwise_ablation(
            model, layers_info, mlp_inputs, eval_ids, device, args.results_dir,
        )
        print()

    # ── experiment 7: interpolation sweep ──
    if 7 in exps:
        run_interpolation_sweep(
            model, layers_info, mlp_inputs, eval_ids, device, args.results_dir,
        )
        print()

    # ── experiment 8: channel-subset ablation ──
    if 8 in exps:
        run_channel_subset_ablation(
            model, layers_info, mlp_inputs, eval_ids, device, args.results_dir,
        )
        print()

    log("done", f"all experiments complete | total_time={time.time() - t0_total:.1f}s")


if __name__ == "__main__":
    main()
