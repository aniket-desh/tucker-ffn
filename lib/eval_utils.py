"""perplexity evaluation."""

import torch


def compute_perplexity(model, input_ids, device):
    """compute perplexity via teacher-forced cross-entropy loss."""
    with torch.no_grad():
        out = model(input_ids.to(device), labels=input_ids.to(device))
    return torch.exp(out.loss).item()
