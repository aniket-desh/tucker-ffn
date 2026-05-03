#!/usr/bin/env python3
"""assemble paper-ready figures and SUMMARY.md.

reads results from results/qwen25_05b/, results/exp10/, results/exp11/,
results/exp12/, results/exp13/, results/exp14/ and writes:

  results/figures/fig_synthetic_fitting.png      [exp10]
  results/figures/fig_diagonal_projection.png    [exp13]
  results/figures/fig_pairing_permutation.png    [exp09 polished]
  results/figures/fig_stable_rank_histogram.png  [exp12]
  results/figures/fig_routing_validation.png     [exp02 + exp04]
  results/figures/fig_lm_loss_curves.png         [exp11]
  results/figures/{name}.tex                      latex include stub
  results/SUMMARY.md                              numerical results index

each figure is a copy/regeneration of an earlier experiment plot, possibly
restyled/composited (e.g. fig_routing_validation merges exp02 + exp04).
"""

import argparse
import json
import os
import pathlib
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import COLOR_CYCLE, PALETTE, log, setup_plot_style  # noqa: E402


def write_tex_stub(name, fig_dir, caption):
    fname = os.path.join(fig_dir, f"{name}.tex")
    body = (
        r"\begin{figure}[t]" + "\n"
        r"\centering" + "\n"
        rf"\includegraphics[width=\columnwidth]{{figures/{name}.png}}" + "\n"
        rf"\caption{{{caption}}}" + "\n"
        rf"\label{{fig:{name}}}" + "\n"
        r"\end{figure}" + "\n"
    )
    with open(fname, "w") as f:
        f.write(body)


def fig_synthetic_fitting(fig_dir, src_dir):
    """copy exp10's plot (with SVD-construction overlay if available)."""
    src = os.path.join(src_dir, "synthetic_fitting.png")
    if not os.path.exists(src):
        log("error", f"missing {src}")
        return
    shutil.copyfile(src, os.path.join(fig_dir, "fig_synthetic_fitting.png"))
    src_pdf = os.path.join(src_dir, "synthetic_fitting.pdf")
    if os.path.exists(src_pdf):
        shutil.copyfile(src_pdf,
                         os.path.join(fig_dir, "fig_synthetic_fitting.pdf"))
    write_tex_stub("fig_synthetic_fitting", fig_dir,
        r"Validation MSE of student SwiGLU and Tucker FFNs fit to a generic "
        r"Tucker teacher with full-rank cores. The matched-coordinates SwiGLU "
        r"shows the predicted knee at $m{=}k^2$ "
        r"(Theorem~\ref{thm:separation}). Stars at $m{=}k^2$ mark the "
        r"\emph{analytic} SVD construction of an aligned-SwiGLU width-$k^2$ "
        r"student via per-gate SVD of $V_j{=}RC^{(j)}$ (no training, no "
        r"optimization), verifying that the upper bound from the proof is "
        r"attained at machine precision ($\lesssim 10^{-12}$).")
    log("done", f"wrote fig_synthetic_fitting.png")


def fig_diagonal_projection(fig_dir, src_dir):
    """2-panel: dose (lambda) + rank truncation (rho).

    drops the trained-vs-projected bar panel — the headline 518x number
    lives in the caption / section prose, not as a single-bar figure.
    """
    dose = os.path.join(src_dir, "diagonal_projection_dose.png")
    rank = os.path.join(src_dir, "rank_truncation_curve.png")

    setup_plot_style()
    panels = [p for p in [dose, rank] if os.path.exists(p)]
    if not panels:
        log("error", f"no exp13 dose/rank outputs in {src_dir}")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 3.5))
    if len(panels) == 1:
        axes = [axes]
    for ax, p in zip(axes, panels):
        img = plt.imread(p)
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    out = os.path.join(fig_dir, "fig_diagonal_projection.png")
    plt.savefig(out, dpi=300)
    plt.close()
    write_tex_stub("fig_diagonal_projection", fig_dir,
        "Diagonal-bottleneck cost on a trained Tucker LM. "
        r"Left: dose-response of validation perplexity over the diagonal "
        r"interpolation $\lambda$ (full Tucker at $\lambda{=}0$, forced "
        r"superdiagonal $C$ at $\lambda{=}1$, with $518\times$ perplexity "
        r"penalty at $\lambda{=}1$). Right: per-gate SVD rank truncation "
        r"$\rho$ of $V_j{=}RC^{(j)}$, tracing the aligned-SwiGLU width "
        r"$m{=}\rho\cdot r$ curve from $\rho{=}1$ (rank-1 ceiling) to "
        r"$\rho{=}r$ (full Tucker).")
    log("done", f"wrote fig_diagonal_projection.png")


def fig_pairing_permutation(fig_dir, src_dir):
    src = os.path.join(src_dir, "pairing_permutation.png")
    if not os.path.exists(src):
        log("error", f"missing {src}")
        return
    dst = os.path.join(fig_dir, "fig_pairing_permutation.png")
    shutil.copyfile(src, dst)
    write_tex_stub("fig_pairing_permutation", fig_dir,
        "Per-layer perplexity under random permutations of one trained "
        r"SwiGLU layer. Joint $\pi_G{=}\pi_U$ breaks the same-index W--G "
        r"coupling; $\pi_U$-only is a control that breaks $U$ pairing alone. "
        "The gap quantifies how binding the same-index coupling is.")
    log("done", f"wrote fig_pairing_permutation.png")


def fig_stable_rank_histogram(fig_dir, src_dir):
    """copy first heatmap and first histogram, for now."""
    setup_plot_style()
    candidates = sorted(pathlib.Path(src_dir).glob("stable_rank_histogram_*.png"))
    if not candidates:
        log("error", f"no stable_rank_histogram_* in {src_dir}")
        return
    src = str(candidates[0])
    dst = os.path.join(fig_dir, "fig_stable_rank_histogram.png")
    shutil.copyfile(src, dst)
    write_tex_stub("fig_stable_rank_histogram", fig_dir,
        r"Distribution of stable rank of $V_j = R C^{(j)}$ across gates "
        "in a Tucker FFN trained from scratch. Stable ranks much greater "
        "than 1 indicate that the model exploits the cross-channel "
        "interactions a SwiGLU cannot represent.")
    log("done", f"wrote fig_stable_rank_histogram.png")


def fig_routing_validation(fig_dir, src_dir):
    """compose exp02 (variance by layer) + exp04 (ablation bar) into 2-panel."""
    setup_plot_style()
    var_path = os.path.join(src_dir, "routing_variance_by_layer.png")
    ab_path = os.path.join(src_dir, "ablation_perplexity.png")
    if not os.path.exists(var_path) or not os.path.exists(ab_path):
        log("error", f"missing exp02/exp04 outputs in {src_dir}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, p in zip(axes, [var_path, ab_path]):
        img = plt.imread(p)
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    out = os.path.join(fig_dir, "fig_routing_validation.png")
    plt.savefig(out, dpi=300)
    plt.close()
    write_tex_stub("fig_routing_validation", fig_dir,
        r"Routed-CP framework validation on Qwen2.5-0.5B. Left: per-layer "
        r"mean routing variance $\mathrm{Var}_x[\alpha_j(x)]$ across depth. "
        r"Right: validation perplexity under three constant-$\alpha$ "
        "ablations of all layers simultaneously.")
    log("done", f"wrote fig_routing_validation.png")


def fig_tucker_teacher_distillation(fig_dir, src_dir):
    src_png = os.path.join(src_dir, "tucker_teacher_distillation.png")
    src_pdf = os.path.join(src_dir, "tucker_teacher_distillation.pdf")
    if not os.path.exists(src_png):
        log("error", f"missing {src_png}")
        return
    shutil.copyfile(src_png,
                     os.path.join(fig_dir, "fig_tucker_teacher_distillation.png"))
    if os.path.exists(src_pdf):
        shutil.copyfile(src_pdf,
                         os.path.join(fig_dir, "fig_tucker_teacher_distillation.pdf"))
    write_tex_stub("fig_tucker_teacher_distillation", fig_dir,
        r"Distillation val MSE for SwiGLU and Tucker FFN students fit to a "
        r"trained Tucker teacher layer ($r{=}s{=}128$, $d{=}512$, layer "
        r"$\ell{=}4$) at matched parameter budgets. Tucker matches the teacher "
        r"more accurately at the larger budgets, where its expressivity "
        r"advantage exceeds the optimization gap.")
    log("done", f"wrote fig_tucker_teacher_distillation.png")


def fig_robustness_panel(fig_dir, results_root, model_tags):
    """3-panel cross-model robustness overlay (exp17).

    overlays Qwen2.5-0.5B (blue, model_tags[0]) and the second model
    family (orange, model_tags[1]) on the same axes. each model's results
    must already exist under results/<tag>/. layer indices on the x-axis
    are the model's own layer indices — depths differ between models, so
    the lines are not paired layer-for-layer; the comparison is in the
    shape and order of magnitude.

    panel (a): per-layer mean routing variance (exp02 routing_stats.npz,
               variances averaged over channels).
    panel (b): bar chart of constant-alpha ablation perplexities, with
               four conditions (baseline, uniform, mean, ones) grouped
               side-by-side per model.
    panel (c): per-layer joint and u_only mean perplexity (lines with
               shaded +/- std bands, log y-axis).
    """
    setup_plot_style()
    if len(model_tags) != 2:
        log("error", f"fig_robustness_panel expects 2 model_tags, got {model_tags}")
        return
    tag_a, tag_b = model_tags
    dir_a = pathlib.Path(results_root) / tag_a
    dir_b = pathlib.Path(results_root) / tag_b

    rs_a = dir_a / "routing_stats.npz"
    rs_b = dir_b / "routing_stats.npz"
    abl_a = dir_a / "ablation_results.json"
    abl_b = dir_b / "ablation_results.json"
    pp_a = dir_a / "pairing_permutation.json"
    pp_b = dir_b / "pairing_permutation.json"
    # exp02 + exp04 are required; exp09 is optional (slow on big models).
    required = [rs_a, rs_b, abl_a, abl_b]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        log("error", f"fig_robustness_panel missing inputs: {missing}")
        return
    have_pp = pp_a.exists() and pp_b.exists()

    color_a = PALETTE["primary"]      # qwen2.5-0.5b — steel blue
    color_b = PALETTE["secondary"]    # second family — dull orange

    # prefer the actual model name from robustness_summary.json (handles the
    # fallback case where the requested model was gated and we landed on
    # something else, so the dir tag is misleading)
    def _label_for(tag, dir_):
        rs = dir_ / "robustness_summary.json"
        if rs.exists():
            try:
                with open(rs) as f:
                    return json.load(f)["model"]
            except Exception:
                pass
        return tag
    label_a = _label_for(tag_a, dir_a)
    label_b = _label_for(tag_b, dir_b)
    # qwen2.5-0.5b results don't have a robustness_summary.json (it's the
    # baseline), so hard-code its label to the canonical model name.
    if label_a == tag_a:
        label_a = "Qwen2.5-0.5B"

    n_panels = 3 if have_pp else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.8))
    if n_panels == 1:
        axes = [axes]

    # ── panel (a): per-layer mean routing variance ─────────────────────
    ax = axes[0]
    for label, path, color, marker in [
        (label_a, rs_a, color_a, "o"),
        (label_b, rs_b, color_b, "s"),
    ]:
        d = np.load(path)
        per_layer_mean = d["variances"].mean(axis=1)
        x = np.arange(len(per_layer_mean))
        ax.plot(x, per_layer_mean, marker=marker, color=color, lw=1.5, ms=4,
                label=label)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"Mean $\mathrm{Var}_x[\alpha_j(x)]$")
    ax.legend(framealpha=0.9, edgecolor="0.8")
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, va="top", ha="left",
            fontsize=11, fontweight="bold")

    # ── panel (b): constant-alpha ablation bars (grouped) ──────────────
    ax = axes[1]
    cond_keys = ["baseline", "uniform", "mean", "ones"]
    cond_labels = ["Baseline", r"$\alpha{=}0.5$", r"$\alpha{=}\bar\alpha$",
                   r"$\alpha{=}1$"]
    width = 0.38
    x = np.arange(len(cond_keys))
    for label, path, color, offset in [
        (label_a, abl_a, color_a, -width / 2),
        (label_b, abl_b, color_b,  width / 2),
    ]:
        with open(path) as f:
            r = json.load(f)
        vals = [float(r[k]) for k in cond_keys]
        ax.bar(x + offset, vals, width=width, color=color, alpha=0.85,
               edgecolor="0.3", linewidth=0.4, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("Perplexity")
    ax.legend(framealpha=0.9, edgecolor="0.8")
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, va="top", ha="left",
            fontsize=11, fontweight="bold")

    # ── panel (c): per-layer joint vs u_only perplexity (optional) ─────
    if have_pp:
        ax = axes[2]
        for label, path, color in [
            (label_a, pp_a, color_a),
            (label_b, pp_b, color_b),
        ]:
            with open(path) as f:
                r = json.load(f)
            floor = float(r["baseline"]) * 0.7
            for cond, ls, marker in [("joint", "-", "o"), ("u_only", "--", "s")]:
                mean = np.array(r[cond]["mean"])
                std = np.array(r[cond]["std"])
                xx = np.arange(len(mean))
                ax.plot(xx, mean, ls=ls, marker=marker, color=color, lw=1.4,
                        ms=3.5, label=f"{label} {cond}")
                ax.fill_between(xx, np.maximum(mean - std, floor), mean + std,
                                color=color, alpha=0.10)
        ax.set_yscale("log")
        ax.set_xlabel("Permuted layer")
        ax.set_ylabel("Perplexity")
        ax.legend(framealpha=0.9, edgecolor="0.8", fontsize=8, ncol=2)
        ax.text(0.02, 0.97, "(c)", transform=ax.transAxes, va="top", ha="left",
                fontsize=11, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(fig_dir, "fig_robustness_panel.png")
    plt.savefig(out, dpi=300)
    plt.close()
    write_tex_stub("fig_robustness_panel", fig_dir,
        r"Cross-model robustness of the routed-CP picture. (a) Per-layer "
        r"mean routing variance $\mathrm{Var}_x[\alpha_j(x)]$. "
        r"(b) Constant-$\alpha$ ablation perplexity (baseline, uniform, "
        r"per-channel mean, $\alpha{=}1$). (c) Joint $\pi_G{=}\pi_U$ vs "
        r"$\pi_U$-only single-layer permutation perplexity (mean $\pm$ std "
        r"over seeds). Both model families show the same qualitative "
        "pattern, ruling out a Qwen-specific artifact.")
    log("done", f"wrote fig_robustness_panel.png")


def fig_lm_loss_curves(fig_dir, src_dir):
    src = os.path.join(src_dir, "loss_curves.png")
    if not os.path.exists(src):
        log("error", f"missing {src}")
        return
    dst = os.path.join(fig_dir, "fig_lm_loss_curves.png")
    shutil.copyfile(src, dst)
    write_tex_stub("fig_lm_loss_curves", fig_dir,
        "Validation loss curves for SwiGLU and Tucker FFN language models "
        "trained from scratch on FineWeb-Edu at matched parameter count "
        r"($d{=}512$, $L{=}8$, $\sim$50M params). Bands are seed std.")
    log("done", f"wrote fig_lm_loss_curves.png")


def write_summary(results_root, fig_dir):
    """write SUMMARY.md indexing every numerical result the paper will cite."""
    out = pathlib.Path(results_root) / "SUMMARY.md"
    lines = []
    lines.append("# Numerical results summary")
    lines.append("")
    lines.append("All numbers cited in the paper, with their source files.")
    lines.append("")

    def add_section(title, path, body_fn):
        if os.path.exists(path):
            lines.append(f"## {title}")
            lines.append(f"_source: `{path}`_")
            lines.append("")
            try:
                body_fn()
            except Exception as e:
                lines.append(f"(error reading: {e})")
            lines.append("")

    # exp04 ablation
    abl = pathlib.Path(results_root) / "qwen25_05b" / "ablation_results.json"
    def _abl():
        with open(abl) as f:
            r = json.load(f)
        lines.append(f"- baseline: {r['baseline']:.2f}")
        lines.append(f"- alpha=0.5 (uniform): {r['uniform']:.2f}")
        lines.append(f"- alpha=mean: {r['mean']:.2f}")
        lines.append(f"- alpha=1 (bilinear): {r['ones']:.2f}")
    add_section("Routing ablation (Qwen2.5-0.5B perplexity)", abl, _abl)

    # exp09 pairing (now supports the 4-condition layout: joint, u_only,
    # g_only, w_only; falls back gracefully on the legacy 2-condition data)
    pp = pathlib.Path(results_root) / "qwen25_05b" / "pairing_permutation.json"
    def _pp():
        with open(pp) as f:
            r = json.load(f)
        lines.append(f"- baseline perplexity: {r['baseline']:.2f}")
        lines.append(f"- n_seeds: {r['n_seeds']}, n_layers: {r['n_layers']}")
        for c in r.get("conditions", ["joint", "u_only"]):
            if c not in r:
                continue
            m = np.array(r[c]["mean"])
            lines.append(f"- {c:8s}: mean={m.mean():.2e}, "
                         f"max={m.max():.2e} (layer {int(m.argmax())}), "
                         f"min={m.min():.2e} (layer {int(m.argmin())})")
        if "geomean_ratio" in r:
            lines.append(f"- geomean(joint/u_only) ratio: "
                         f"{r['geomean_ratio']}")
        # noop control overlay if present
        noop = pathlib.Path(results_root) / "qwen25_05b" / "noop_control.json"
        if noop.exists():
            with open(noop) as f:
                no = json.load(f)
            lines.append(f"- no-op (g, u, w joint perm): "
                         f"max_dev_from_baseline={max(no['noop_per_layer_max_dev']):.2e}")
    add_section("Same-index pairing permutation (exp09 + exp09b)", pp, _pp)

    # exp10 synthetic
    syn = pathlib.Path(results_root) / "exp10" / "synthetic_fitting.npz"
    def _syn():
        d = np.load(syn, allow_pickle=True)
        students = list(d["students"])
        for k in d["k_values"]:
            mvals = d[f"k{k}_m_values"]
            arr = d[f"k{k}_val_mse"]
            lines.append(f"- k = {k}, m_values = {list(mvals)}")
            for si, st in enumerate(students):
                bests = [f"{np.nanmin(arr[si, mi]):.3e}" for mi in range(len(mvals))]
                lines.append(f"  - {st}: {bests}")
    add_section("Synthetic fitting limit (exp10)", syn, _syn)

    # exp11 lm losses (default-init runs and any hill-climb dirs side-by-side)
    lm_root_candidates = sorted(pathlib.Path(results_root).glob("exp11*"))
    if lm_root_candidates:
        lines.append("## LM training (exp11 + hill-climb variants)")
        for lm_dir in lm_root_candidates:
            if not lm_dir.is_dir() or lm_dir.name.endswith("_smoke"):
                continue
            for d in sorted(lm_dir.iterdir()):
                if not d.is_dir():
                    continue
                cfg = d / "config.json"
                ll = d / "loss_log.json"
                if cfg.exists() and ll.exists():
                    with open(cfg) as f:
                        c = json.load(f)
                    with open(ll) as f:
                        log_ = json.load(f)
                    if not log_:
                        continue
                    last = log_[-1]
                    diag_init = c.get("tucker_diagonal_bias_init", False) \
                        if "tucker_diagonal_bias_init" in c else "?"
                    tag = f"{lm_dir.name}/{d.name}"
                    lines.append(
                        f"- `{tag}`: arch={c['arch']} seed={c['seed']} "
                        f"params={c['n_params_total']/1e6:.1f}M | "
                        f"final val_loss={last['val_loss']:.3f} "
                        f"(ppl={np.exp(last['val_loss']):.1f}) "
                        f"after {last['tokens']/1e6:.1f}M tokens"
                    )
        lines.append("")

    # exp12 stable ranks (default-init + hill-climb variants)
    for sr_dir_name in sorted(p.name for p in pathlib.Path(results_root).glob("exp12*")):
        sr = pathlib.Path(results_root) / sr_dir_name / "stable_rank.npz"
        def _sr(_sr=sr, _name=sr_dir_name):
            d = np.load(_sr, allow_pickle=True)
            for key in d.files:
                if not key.endswith("__s_rank"):
                    continue
                tag = key[:-len("__s_rank")]
                arr = d[key]
                lines.append(f"- {_name}/{tag}: mean stable rank = {arr.mean():.2f} | "
                             f"median = {np.median(arr):.2f} | "
                             f"min = {arr.min():.2f} | max = {arr.max():.2f}")
        add_section(f"Stable rank of V_j ({sr_dir_name})", sr, _sr)

    # exp13 diag projection (default-init + hill-climb variants)
    for dp_dir_name in sorted(p.name for p in pathlib.Path(results_root).glob("exp13*")):
        dp = pathlib.Path(results_root) / dp_dir_name / "results.json"
        def _dp(_dp=dp, _name=dp_dir_name):
            with open(_dp) as f:
                r = json.load(f)
            for tag, d_ in r.items():
                full = d_["diagonal_projection"][0]["perplexity"]
                diag = d_["diagonal_projection"][-1]["perplexity"]
                lines.append(f"- {_name}/{tag}: trained ppl = {full:.2f}, "
                             f"diagonal-projected ppl = {diag:.2f}, "
                             f"ratio = {diag/full:.2f}x")
                lines.append(f"  - rank-truncation curve:")
                for r_ in d_.get("rank_truncation", []):
                    lines.append(f"    - rho={r_['rho']}: ppl={r_['perplexity']:.2f}")
        add_section(f"Diagonal projection / rank truncation ({dp_dir_name})", dp, _dp)

    # exp14 distillation (swiglu teacher)
    di = pathlib.Path(results_root) / "exp14_v2" / "distillation.json"
    if not di.exists():
        di = pathlib.Path(results_root) / "exp14" / "distillation.json"
    def _di():
        with open(di) as f:
            r = json.load(f)
        lines.append(f"- teacher: swiglu layer of {r['model']}, "
                     f"layer {r['teacher_layer']}, d={r['d']}, "
                     f"m_teacher={r['m_teacher']}")
        for entry in r["sweep"]:
            sw_means = [x["val_mse"] for x in entry["results"]["swiglu"]]
            tk_means = [x["val_mse"] for x in entry["results"]["tucker"]]
            lines.append(f"- m_swiglu={entry['m_swiglu']}, r=s={entry['r']}: "
                         f"swiglu val_mse = {np.mean(sw_means):.3e} "
                         f"({np.std(sw_means):.1e}), "
                         f"tucker val_mse = {np.mean(tk_means):.3e} "
                         f"({np.std(tk_means):.1e}), "
                         f"ratio = {np.mean(sw_means)/max(np.mean(tk_means), 1e-30):.2f}x")
    add_section("Distillation gap (exp14, swiglu teacher)", di, _di)

    # exp14b distillation (tucker teacher)
    db = pathlib.Path(results_root) / "exp14b" / "tucker_teacher_distillation.json"
    def _db():
        with open(db) as f:
            r = json.load(f)
        lines.append(f"- teacher: trained tucker layer {r['teacher_layer']} "
                     f"of {r['teacher_ckpt']}, d={r['d']}, r_teacher={r['r_teacher']}")
        for entry in r["sweep"]:
            sw_means = [x["val_mse"] for x in entry["results"]["swiglu"]]
            tk_means = [x["val_mse"] for x in entry["results"]["tucker"]]
            lines.append(f"- m_swiglu={entry['m_swiglu']}, r=s={entry['r']}: "
                         f"swiglu val_mse = {np.mean(sw_means):.3e} "
                         f"({np.std(sw_means):.1e}), "
                         f"tucker val_mse = {np.mean(tk_means):.3e} "
                         f"({np.std(tk_means):.1e}), "
                         f"ratio = {np.mean(sw_means)/max(np.mean(tk_means), 1e-30):.2f}x")
    add_section("Distillation gap (exp14b, tucker teacher)", db, _db)

    # paper figures: main text (5) and appendix (3)
    lines.append("## Paper figures (main text)")
    for name in ["fig_routing_validation", "fig_synthetic_fitting",
                  "fig_diagonal_projection", "fig_stable_rank_histogram",
                  "fig_tucker_teacher_distillation"]:
        png = pathlib.Path(fig_dir) / f"{name}.png"
        status = "OK" if png.exists() else "MISSING"
        lines.append(f"- `{name}.png` [{status}], stub `{name}.tex`")
    lines.append("")
    lines.append("## Paper figures (appendix)")
    for name in ["fig_pairing_permutation", "fig_lm_loss_curves",
                  "fig_robustness_panel"]:
        png = pathlib.Path(fig_dir) / f"{name}.png"
        status = "OK" if png.exists() else "MISSING"
        lines.append(f"- `{name}.png` [{status}], stub `{name}.tex`")
    lines.append("")

    # if SUMMARY.md was hand-curated to embed figures inline (look for the
    # paper-flow signature), don't clobber it -- write the auto-generated
    # numbers to SUMMARY_auto.md instead so a human can merge if needed.
    auto_marker = "## §3 Routed-CP framework"
    if out.exists() and auto_marker in out.read_text():
        out_auto = out.with_name("SUMMARY_auto.md")
        out_auto.write_text("\n".join(lines))
        log("info", f"SUMMARY.md is hand-curated (paper-flow); wrote auto "
            f"version to {out_auto}")
    else:
        out.write_text("\n".join(lines))
        log("done", f"wrote SUMMARY.md -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--tucker_variant", type=str, default="hc_v3",
                        help="which trained-tucker variant to source the "
                             "§5.2 figures from. options: '' (=default-init "
                             "exp12/exp13), 'hc' (v1), 'hc_v2', 'hc_v3'. "
                             "the paper narrative uses hc_v3.")
    parser.add_argument("--robustness_tags", type=str,
                        default="qwen25_05b,llama32_1b",
                        help="comma-separated <baseline>,<other> model tags "
                             "for fig_robustness_panel (must match exp17 "
                             "output dir names under results/)")
    args = parser.parse_args()
    setup_plot_style()
    fig_dir = os.path.join(args.results_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # tucker variant for §5.2 figures (diagonal projection + stable rank).
    # the paper narrative is built around hc_v3 (corrected variance-preserving
    # init), so source the trained-tucker figures from exp12_hc_v3 / exp13_hc_v3.
    tucker_variant = args.tucker_variant
    fig_synthetic_fitting(fig_dir, os.path.join(args.results_root, "exp10"))
    fig_diagonal_projection(fig_dir,
                             os.path.join(args.results_root, f"exp13_{tucker_variant}"))
    fig_pairing_permutation(fig_dir, os.path.join(args.results_root, "qwen25_05b"))
    fig_stable_rank_histogram(fig_dir,
                               os.path.join(args.results_root, f"exp12_{tucker_variant}"))
    fig_routing_validation(fig_dir, os.path.join(args.results_root, "qwen25_05b"))
    fig_lm_loss_curves(fig_dir, os.path.join(args.results_root, "exp11"))
    fig_tucker_teacher_distillation(fig_dir,
                                     os.path.join(args.results_root, "exp14b"))

    robustness_tags = [t.strip() for t in args.robustness_tags.split(",")
                        if t.strip()]
    fig_robustness_panel(fig_dir, args.results_root, robustness_tags)

    write_summary(args.results_root, fig_dir)


if __name__ == "__main__":
    main()
