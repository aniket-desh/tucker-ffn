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
    """just copy exp10's plot as the headline."""
    src = os.path.join(src_dir, "synthetic_fitting.png")
    if not os.path.exists(src):
        log("error", f"missing {src}")
        return
    dst = os.path.join(fig_dir, "fig_synthetic_fitting.png")
    shutil.copyfile(src, dst)
    write_tex_stub("fig_synthetic_fitting", fig_dir,
        "Validation MSE of student SwiGLU and Tucker FFNs fit to a generic "
        "Tucker teacher with full-rank cores. The matched-coordinates SwiGLU "
        r"shows the predicted knee at $m=k^2$ (Theorem~\ref{thm:separation}).")
    log("done", f"wrote fig_synthetic_fitting.png")


def fig_diagonal_projection(fig_dir, src_dir):
    bar = os.path.join(src_dir, "diagonal_projection_bar.png")
    dose = os.path.join(src_dir, "diagonal_projection_dose.png")
    rank = os.path.join(src_dir, "rank_truncation_curve.png")

    setup_plot_style()
    n_panels = sum(os.path.exists(p) for p in [bar, dose, rank])
    if n_panels == 0:
        log("error", f"no exp13 outputs in {src_dir}")
        return
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.5))
    if n_panels == 1:
        axes = [axes]
    panel = 0
    for p in [bar, dose, rank]:
        if not os.path.exists(p):
            continue
        img = plt.imread(p)
        axes[panel].imshow(img)
        axes[panel].axis("off")
        panel += 1
    plt.tight_layout()
    out = os.path.join(fig_dir, "fig_diagonal_projection.png")
    plt.savefig(out, dpi=300)
    plt.close()
    write_tex_stub("fig_diagonal_projection", fig_dir,
        "Diagonal-bottleneck cost on a trained Tucker LM. Left: validation "
        r"perplexity at $\lambda{=}0$ (trained) vs $\lambda{=}1$ (forced "
        r"superdiagonal $C$). Center: dose-response over $\lambda$. Right: "
        r"per-gate SVD rank truncation $\rho$ of $V_j{=}RC^{(j)}$.")
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
    src = os.path.join(src_dir, "tucker_teacher_distillation.png")
    if not os.path.exists(src):
        log("error", f"missing {src}")
        return
    dst = os.path.join(fig_dir, "fig_tucker_teacher_distillation.png")
    shutil.copyfile(src, dst)
    write_tex_stub("fig_tucker_teacher_distillation", fig_dir,
        "Distillation val MSE for SwiGLU and Tucker FFN students fit to a "
        r"trained Tucker teacher layer (matched parameter budgets, $r{=}128$). "
        "Tucker matches the teacher more accurately at the larger budgets, "
        "where its expressivity advantage exceeds the optimization gap.")
    log("done", f"wrote fig_tucker_teacher_distillation.png")


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

    # exp12 stable ranks
    sr = pathlib.Path(results_root) / "exp12" / "stable_rank.npz"
    def _sr():
        d = np.load(sr, allow_pickle=True)
        for key in d.files:
            if not key.endswith("__s_rank"):
                continue
            tag = key[:-len("__s_rank")]
            arr = d[key]
            lines.append(f"- {tag}: mean stable rank = {arr.mean():.2f} | "
                         f"median = {np.median(arr):.2f} | "
                         f"min = {arr.min():.2f} | max = {arr.max():.2f}")
    add_section("Stable rank of V_j (exp12)", sr, _sr)

    # exp13 diag projection
    dp = pathlib.Path(results_root) / "exp13" / "results.json"
    def _dp():
        with open(dp) as f:
            r = json.load(f)
        for tag, d_ in r.items():
            full = d_["diagonal_projection"][0]["perplexity"]
            diag = d_["diagonal_projection"][-1]["perplexity"]
            lines.append(f"- {tag}: trained ppl = {full:.2f}, "
                         f"diagonal-projected ppl = {diag:.2f}, "
                         f"ratio = {diag/full:.2f}x")
            lines.append(f"  - rank-truncation curve:")
            for r_ in d_.get("rank_truncation", []):
                lines.append(f"    - rho={r_['rho']}: ppl={r_['perplexity']:.2f}")
    add_section("Diagonal projection / rank truncation (exp13)", dp, _dp)

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

    # paper figures
    lines.append("## Paper figures")
    for name in ["fig_synthetic_fitting", "fig_diagonal_projection",
                  "fig_pairing_permutation", "fig_stable_rank_histogram",
                  "fig_routing_validation", "fig_lm_loss_curves",
                  "fig_tucker_teacher_distillation"]:
        png = pathlib.Path(fig_dir) / f"{name}.png"
        tex = pathlib.Path(fig_dir) / f"{name}.tex"
        status = "OK" if png.exists() else "MISSING"
        lines.append(f"- `{name}.png` [{status}], stub `{name}.tex`")
    lines.append("")

    out.write_text("\n".join(lines))
    log("done", f"wrote SUMMARY.md -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=str, default="results")
    args = parser.parse_args()
    setup_plot_style()
    fig_dir = os.path.join(args.results_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig_synthetic_fitting(fig_dir, os.path.join(args.results_root, "exp10"))
    fig_diagonal_projection(fig_dir, os.path.join(args.results_root, "exp13"))
    fig_pairing_permutation(fig_dir, os.path.join(args.results_root, "qwen25_05b"))
    fig_stable_rank_histogram(fig_dir, os.path.join(args.results_root, "exp12"))
    fig_routing_validation(fig_dir, os.path.join(args.results_root, "qwen25_05b"))
    fig_lm_loss_curves(fig_dir, os.path.join(args.results_root, "exp11"))
    fig_tucker_teacher_distillation(fig_dir,
                                     os.path.join(args.results_root, "exp14b"))

    write_summary(args.results_root, fig_dir)


if __name__ == "__main__":
    main()
