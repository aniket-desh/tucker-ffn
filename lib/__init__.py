"""shared utilities for swiglu tensor decomposition experiments."""

from .activations import (
    capture_mlp_io,
    compute_channel_quantities,
    get_weight_and_bias,
)
from .eval_utils import compute_perplexity
from .log_utils import log
from .model_utils import (
    detect_device,
    get_swiglu_layers,
    load_model_and_tokenizer,
    load_text_data,
)
from .permutation import permuted_layer
from .plot_style import (
    COLOR_CYCLE,
    LAYER_CMAP,
    LS_CYCLE,
    MARKER_CYCLE,
    PALETTE,
    setup_plot_style,
)
from .routing import (
    ChannelSubsetRouting,
    ConstantRouting,
    InterpolatedRouting,
    ablated_routing,
    ablated_single_layer,
    channel_subset_ablation,
    interpolated_routing,
)
from .runner import add_common_args, prepare_run
from .tucker_ffn import (
    SwiGLUFFN,
    SwiGLUFFNAligned,
    TuckerFFN,
    swiglu_params,
    swiglu_width_for_params,
    tucker_params,
)

__all__ = [
    "COLOR_CYCLE",
    "ChannelSubsetRouting",
    "ConstantRouting",
    "InterpolatedRouting",
    "LAYER_CMAP",
    "LS_CYCLE",
    "MARKER_CYCLE",
    "PALETTE",
    "SwiGLUFFN",
    "SwiGLUFFNAligned",
    "TuckerFFN",
    "ablated_routing",
    "ablated_single_layer",
    "add_common_args",
    "capture_mlp_io",
    "channel_subset_ablation",
    "compute_channel_quantities",
    "compute_perplexity",
    "detect_device",
    "get_swiglu_layers",
    "get_weight_and_bias",
    "interpolated_routing",
    "load_model_and_tokenizer",
    "load_text_data",
    "log",
    "permuted_layer",
    "prepare_run",
    "setup_plot_style",
    "swiglu_params",
    "swiglu_width_for_params",
    "tucker_params",
]
