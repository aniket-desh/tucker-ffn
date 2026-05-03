#!/usr/bin/env python3
"""Rerun §5.1 ablation and Fig 2 baseline preservation on a 32K-token
WikiText-2 test split (vs the cached 4096-token chunk). Reports deltas;
regenerates Table 1 inputs only if the headline numbers shift materially."""
import json, sys, pathlib, torch
sys.path.insert(0, ".")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from lib import (ablated_routing, compute_channel_quantities,  # noqa: E402
                 compute_perplexity, log)
from lib.model_utils import get_swiglu_layers
from lib.activations import capture_mlp_io


def chunked_perplexity(model, ids, device, chunk=4096):
    """compute perplexity over a long sequence by chunking into non-overlapping
    windows of length `chunk`. averages cross-entropy uniformly across tokens
    (treats each chunk as independent context, like the cached 4K eval did)."""
    import math
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, ids.shape[1], chunk):
            seg = ids[:, start:start + chunk].to(device)
            if seg.shape[1] < 2:
                continue
            out = model(seg, labels=seg)
            n = seg.shape[1] - 1
            total_nll += float(out.loss.item()) * n
            total_tokens += n
    return float(math.exp(total_nll / total_tokens))


MODEL = "Qwen/Qwen2.5-0.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"
log("info", f"loading {MODEL} on {device}")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32
).to(device).eval()

# fixed 32K-token WikiText-2 test chunk (cached so future reruns are bit-exact)
out_dir = pathlib.Path("results/eval_32k")
out_dir.mkdir(parents=True, exist_ok=True)
ids_cache = out_dir / "wikitext2_test_32k.pt"
if ids_cache.exists():
    ids = torch.load(ids_cache).to(device)
else:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids[:, :32768].to(device)
    torch.save(ids.cpu(), ids_cache)
log("info", f"eval_ids shape = {tuple(ids.shape)}  ({ids.numel()} tokens)")

# baseline (chunk 4K windows to fit memory; matches cached eval's chunk size)
ppl_base = chunked_perplexity(model, ids.cpu(), device, chunk=4096)
log("eval", f"baseline (32K, chunked 4K) ppl = {ppl_base:.4f}")

# calibrate alpha mean (4K-token slice -- §5.1 calibration size)
ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
calib_text = "\n\n".join(t for t in ds_train["text"] if t.strip())
calib_ids = tok(calib_text, return_tensors="pt").input_ids[:, :4096].to(device)

layers_info = get_swiglu_layers(model)
mlp_inputs, _ = capture_mlp_io(model, calib_ids, layers_info, device)
mean_alphas = {}
for info in layers_info:
    _, _, alpha, _ = compute_channel_quantities(
        mlp_inputs[info["layer_idx"]], info)
    mean_alphas[info["layer_idx"]] = alpha.mean(dim=0)

# constant-alpha sweep on 32K eval set
results = {"baseline": ppl_base, "n_eval_tokens": int(ids.numel())}
for mode in ("uniform", "mean", "ones"):
    with ablated_routing(model, layers_info, mode, mean_alphas):
        p = chunked_perplexity(model, ids.cpu(), device, chunk=4096)
    results[mode] = p
    log("eval", f"{mode:8s}  ppl = {p:.2f}")

with open(out_dir / "ablation_32k.json", "w") as f:
    json.dump(results, f, indent=2)

# delta vs cached 4K
cached = json.load(open("results/qwen25_05b/ablation_results.json"))
log("info", "comparison (4K cached vs 32K rerun):")
for k in ("baseline", "uniform", "mean", "ones"):
    rel = (results[k] - cached[k]) / max(cached[k], 1e-9) * 100
    log("info", f"  {k:8s}: 4K={cached[k]:.2f}  32K={results[k]:.2f}  Δ={rel:+.1f}%")
