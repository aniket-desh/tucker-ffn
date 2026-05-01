"""device detection, model loading, swiglu layer discovery, text data loading."""

import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .log_utils import log


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
