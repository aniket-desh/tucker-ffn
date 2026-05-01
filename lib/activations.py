"""mlp activation capture and per-channel cp-decomposition quantities."""

import torch


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
