"""shared cli setup for standalone experiment scripts.

each experiments/expNN_*.py uses add_common_args() + prepare_run() so that
running a single experiment performs the same model load / activation
capture as the full orchestrator in run_experiments.py.
"""

import os
import time

import numpy as np
import torch

from .activations import capture_mlp_io
from .log_utils import log
from .model_utils import (
    detect_device,
    get_swiglu_layers,
    load_model_and_tokenizer,
    load_text_data,
)
from .plot_style import setup_plot_style


def add_common_args(parser):
    """attach the standard --model / --max_tokens / --results_dir / etc. flags."""
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B",
        help="huggingface model name (must use swiglu architecture)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096,
        help="max tokens per data chunk (analysis and eval)",
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
    return parser


def prepare_run(args, capture_activations=True):
    """seed, set up plot style, load model + data, optionally capture activations.

    returns a dict with everything an experiment needs:
      model, tokenizer, layers_info, analysis_ids, eval_ids, device,
      mlp_inputs, mlp_outputs (the latter two empty if capture_activations=False)
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    setup_plot_style()

    device = args.device or detect_device()
    log("info", f"model={args.model} | max_tokens={args.max_tokens} | device={device}")
    log("info", f"results_dir={args.results_dir}")
    print()

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    layers_info = get_swiglu_layers(model)
    analysis_ids, eval_ids = load_text_data(tokenizer, args.max_tokens)
    print()

    mlp_inputs, mlp_outputs = {}, {}
    if capture_activations:
        log("info", "running forward pass to capture mlp activations")
        t0 = time.time()
        mlp_inputs, mlp_outputs = capture_mlp_io(
            model, analysis_ids, layers_info, device,
        )
        log("info", f"activations captured | n_layers={len(mlp_inputs)} | "
            f"time={time.time() - t0:.1f}s")
        print()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "layers_info": layers_info,
        "analysis_ids": analysis_ids,
        "eval_ids": eval_ids,
        "device": device,
        "mlp_inputs": mlp_inputs,
        "mlp_outputs": mlp_outputs,
    }
