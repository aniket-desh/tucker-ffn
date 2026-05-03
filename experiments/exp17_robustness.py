#!/usr/bin/env python3
"""experiment 17: cross-model robustness panel.

reruns the three core routing/permutation diagnostics on a second model
family so the routed-cp picture (note section on interpretability) is not
dismissable as a Qwen2.5-0.5B artifact. composes existing entry points:

  exp02 — per-channel routing variance s_j = Var_x[alpha_j(x)]
  exp04 — constant-alpha ablation perplexities (note eq 10 with alpha frozen)
  exp09 — same-index pairing permutation (joint pi_G=pi_U vs pi_U-only control)

default model is meta-llama/Llama-3.2-1B; on a 401 / gated-repo failure we
fall back to Qwen/Qwen3-1.7B (also a swiglu architecture). no other model
families are tried — gemma / gpt-2 / pythia / phi-2 do not match the
gate_proj/up_proj/down_proj convention this codebase relies on.

outputs (under --results_dir, default results/<model_tag>/):
  routing_stats.npz                  — from exp02
  alpha_distribution_by_layer.png
  routing_variance_heatmap.png
  routing_variance_by_layer.png
  ablation_results.json              — from exp04
  ablation_perplexity.png
  pairing_permutation.json           — from exp09 (n_seeds=4)
  pairing_permutation.png
  robustness_summary.json            — three headline numbers, with
                                       results/qwen25_05b/ side-by-side
"""

import argparse
import json
import os
import pathlib
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib import add_common_args, log, prepare_run  # noqa: E402

from experiments.exp02_routing_stats import run_routing_stats  # noqa: E402
from experiments.exp04_routing_ablation import run_ablation  # noqa: E402
from experiments.exp09_pairing_permutation import (  # noqa: E402
    run_pairing_permutation,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def slugify_model(model_name):
    """meta-llama/Llama-3.2-1B -> llama32_1b (drop org, dots, dashes -> _).

    rules (matched to existing tags `qwen25_05b`, `llama32_1b`):
      - keep only the trailing path component, lowercased
      - drop dots inside numeric versions: "3.2" -> "32", "2.5" -> "25"
      - drop a single dash directly between letters and the model version
        digits ("llama-32" -> "llama32")
      - any remaining run of non-alphanumerics collapses to a single "_"
    """
    base = model_name.rsplit("/", 1)[-1].lower()
    # collapse dotted versions: "3.2" -> "32", "0.5" -> "05"
    base = re.sub(r"(\d)\.(\d)", r"\1\2", base)
    # drop the dash between a name and its version digits
    base = re.sub(r"([a-z])-(\d)", r"\1\2", base)
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return base


def _safe_prepare_run(args, capture_activations=True):
    """prepare_run with a single fallback to Qwen/Qwen3-1.7B on 401 / gated.

    we catch huggingface_hub access errors and the underlying HTTPError so
    the script keeps running on machines without llama3 access. only one
    fallback model is ever attempted.
    """
    try:
        return prepare_run(args, capture_activations=capture_activations)
    except Exception as e:
        # lazy-import so the script does not require huggingface_hub at top
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
        from requests.exceptions import HTTPError

        is_access = isinstance(e, (GatedRepoError, RepositoryNotFoundError))
        is_401 = (
            isinstance(e, (HfHubHTTPError, HTTPError))
            and getattr(getattr(e, "response", None), "status_code", None) == 401
        )
        # also catch transformers OSError wrapping a 401 ("You are trying
        # to access a gated repo")
        msg = str(e).lower()
        is_msg = ("401" in msg) or ("gated" in msg) or ("access" in msg
                                                        and "repo" in msg)

        if not (is_access or is_401 or is_msg):
            raise

        log("warn", f"model load failed for {args.model} ({type(e).__name__}); "
            f"falling back to Qwen/Qwen3-1.7B")
        args.model = "Qwen/Qwen3-1.7B"
        return prepare_run(args, capture_activations=capture_activations)


def _load_qwen_baseline(qwen_dir):
    """load the three headline numbers for Qwen2.5-0.5B from results/qwen25_05b/.

    returns a dict with the same three keys we report for the new model;
    missing files yield None entries so the panel still writes.
    """
    out = {
        "mean_routing_variance": None,
        "ablation_ones_ratio": None,
        "joint_over_u_only_geomean": None,
    }
    rs_path = os.path.join(qwen_dir, "routing_stats.npz")
    if os.path.exists(rs_path):
        d = np.load(rs_path)
        out["mean_routing_variance"] = float(d["variances"].mean())
    else:
        log("warn", f"missing {rs_path}")

    abl_path = os.path.join(qwen_dir, "ablation_results.json")
    if os.path.exists(abl_path):
        with open(abl_path) as f:
            r = json.load(f)
        out["ablation_ones_ratio"] = float(r["ones"]) / float(r["baseline"])
    else:
        log("warn", f"missing {abl_path}")

    pp_path = os.path.join(qwen_dir, "pairing_permutation.json")
    if os.path.exists(pp_path):
        with open(pp_path) as f:
            r = json.load(f)
        if "geomean_ratio" in r:
            out["joint_over_u_only_geomean"] = float(r["geomean_ratio"])
        else:
            joint = np.array(r["joint"]["mean"])
            uo = np.array(r["u_only"]["mean"])
            ratio = joint / np.maximum(uo, 1e-30)
            out["joint_over_u_only_geomean"] = float(np.exp(np.log(ratio).mean()))
    else:
        log("warn", f"missing {pp_path}")
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B")
    parser.add_argument("--model_tag", type=str, default=None,
                        help="output directory name under results/ "
                             "(default: slugify(--model))")
    parser.add_argument("--qwen_dir", type=str,
                        default="results/qwen25_05b",
                        help="directory holding the Qwen2.5-0.5B baseline "
                             "results to compare against")
    parser.add_argument("--n_seeds", type=int, default=4,
                        help="number of permutation seeds for exp09 "
                             "(default 4; the standalone exp09 default is 8)")
    parser.add_argument("--skip_exp09", action="store_true",
                        help="skip the per-layer pairing-permutation panel. "
                             "exp09 dominates wall-clock on larger models "
                             "(28 layers x 4 conds x n_seeds). exp02+exp04 "
                             "alone still defuse the Qwen-specific concern.")
    args = parser.parse_args()

    t0_total = time.time()

    # remember the requested model tag separately from args.model in case
    # the load falls back to Qwen3-1.7B — the user almost always wants the
    # output dir named after what they asked for.
    requested_model = args.model
    requested_tag = args.model_tag or slugify_model(requested_model)
    args.results_dir = os.path.join(args.results_dir, requested_tag)
    os.makedirs(args.results_dir, exist_ok=True)

    log("info", "experiment 17: cross-model robustness panel")
    log("info", f"requested_model={requested_model} | model_tag={requested_tag}")
    print()

    ctx = _safe_prepare_run(args, capture_activations=True)
    actual_model = args.model  # may have changed in the fallback path
    if actual_model != requested_model:
        log("info", f"actual_model={actual_model} (fell back from "
            f"{requested_model})")
        print()

    layers_info = ctx["layers_info"]
    mlp_inputs = ctx["mlp_inputs"]
    eval_ids = ctx["eval_ids"]
    device = ctx["device"]
    model = ctx["model"]
    results_dir = args.results_dir

    # ── exp02: routing variance ──────────────────────────────────────────
    t0 = time.time()
    all_vars, _ = run_routing_stats(layers_info, mlp_inputs, results_dir)
    mean_routing_variance = float(np.concatenate(all_vars).mean())
    log("robust", f"exp02 mean_routing_variance={mean_routing_variance:.3e} "
        f"(across all layers, all channels) | time={time.time() - t0:.1f}s")
    print()

    # ── exp04: constant-alpha ablation ───────────────────────────────────
    t0 = time.time()
    abl_results = run_ablation(
        model, layers_info, mlp_inputs, eval_ids, device, results_dir,
    )
    ablation_ones_ratio = float(abl_results["ones"]) / float(abl_results["baseline"])
    log("robust", f"exp04 ones_over_baseline_ppl_ratio={ablation_ones_ratio:.2e} "
        f"(alpha=1 / baseline) | time={time.time() - t0:.1f}s")
    print()

    # ── exp09: same-index pairing permutation ────────────────────────────
    if args.skip_exp09:
        log("info", "skipping exp09 (--skip_exp09 set)")
        joint_over_u_only_geomean = None
    else:
        t0 = time.time()
        pp_results = run_pairing_permutation(
            model, layers_info, eval_ids, device, results_dir,
            n_seeds=args.n_seeds, base_seed=args.seed,
        )
        joint_over_u_only_geomean = float(pp_results["geomean_ratio"])
        log("robust", f"exp09 joint_over_u_only_geomean={joint_over_u_only_geomean:.2e} "
            f"| time={time.time() - t0:.1f}s")
        print()

    # ── compose summary side-by-side with Qwen2.5-0.5B ───────────────────
    qwen_baseline = _load_qwen_baseline(args.qwen_dir)
    summary = {
        "model": actual_model,
        "model_tag": requested_tag,
        "this_model": {
            "mean_routing_variance": mean_routing_variance,
            "ablation_ones_ratio": ablation_ones_ratio,
            "joint_over_u_only_geomean": joint_over_u_only_geomean,
        },
        "qwen25_05b": qwen_baseline,
    }
    out_path = os.path.join(results_dir, "robustness_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log("done", f"saved robustness_summary.json -> {results_dir}/")

    def _fmt(v, spec):
        return format(v, spec) if v is not None else "n/a"
    log("summary",
        f"this_model: var={_fmt(summary['this_model']['mean_routing_variance'], '.3e')} | "
        f"ones_ratio={_fmt(summary['this_model']['ablation_ones_ratio'], '.2e')} | "
        f"joint/u_only={_fmt(summary['this_model']['joint_over_u_only_geomean'], '.2e')}")
    if any(v is not None for v in qwen_baseline.values()):
        log("summary",
            f"qwen25_05b: var={_fmt(qwen_baseline['mean_routing_variance'], '.3e')} | "
            f"ones_ratio={_fmt(qwen_baseline['ablation_ones_ratio'], '.2e')} | "
            f"joint/u_only={_fmt(qwen_baseline['joint_over_u_only_geomean'], '.2e')}")

    log("done", f"experiment 17 complete | "
        f"total_time={(time.time() - t0_total) / 60:.1f}min")


if __name__ == "__main__":
    main()
