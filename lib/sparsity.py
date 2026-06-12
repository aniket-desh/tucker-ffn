"""Trained sparsity penalties for CP/LL1 FFNs (sprint 2, Exp C).

SparsityManager(model, mode, eps) registers forward hooks on every FFN in a
lib.lm.LM and accumulates a differentiable penalty per forward pass:

  mode="route_l1"    mean |SiLU(gate_pre)| over tokens and routes
                     (SwiGLU: per-atom gates; LL1: per-block gates)
  mode="contrib_l1"  mean |h| over tokens and hidden units (SwiGLU h_j; for
                     LL1 the blockwise main*gate activations)
  mode="group_lasso" mean over tokens of sum_b ||block hidden vector||_2
                     (LL1 blocks; for SwiGLU each atom is its own group, which
                     reduces to contrib_l1)

Usage in a train step:
    mgr.reset()
    _, ce = model(x, targets=y)
    loss = ce + lam * mgr.penalty()

mgr.realized_sparsity() reports the fraction of route activations below eps on
the last forward (logging only, not differentiable).
"""

import torch
import torch.nn.functional as F

from .ll1_ffn import LL1FFN
from .tucker_ffn import SwiGLUFFN


class SparsityManager:
    def __init__(self, model, mode="route_l1", eps=1e-3):
        self.mode = mode
        self.eps = eps
        self._terms = []
        self._below = []
        self._handles = []
        for blk in model.blocks:
            ffn = blk.ffn
            if isinstance(ffn, (SwiGLUFFN, LL1FFN)):
                self._handles.append(
                    ffn.gate_proj.register_forward_hook(self._gate_hook(ffn)))
                if mode in ("contrib_l1", "group_lasso"):
                    self._handles.append(
                        ffn.up_proj.register_forward_hook(self._up_hook(ffn)))
        self._gate_cache = {}

    def _gate_hook(self, ffn):
        def hook(mod, inp, out):
            s = F.silu(out)
            if self.mode == "route_l1":
                self._terms.append(s.abs().mean())
            self._gate_cache.setdefault(id(ffn), {})["gate"] = s
            with torch.no_grad():
                self._below.append((s.abs() < self.eps).float().mean().item())
            self._maybe_emit(ffn)
        return hook

    def _up_hook(self, ffn):
        def hook(mod, inp, out):
            self._gate_cache.setdefault(id(ffn), {})["up"] = out
            self._maybe_emit(ffn)
        return hook

    def _maybe_emit(self, ffn):
        if self.mode == "route_l1":
            return
        cache = self._gate_cache.get(id(ffn), {})
        if "gate" not in cache or "up" not in cache:
            return
        s, up = cache.pop("gate"), cache.pop("up")
        if isinstance(ffn, LL1FFN):
            shape = up.shape[:-1]
            h = up.view(*shape, ffn.n_blocks, ffn.block_rank) * s.unsqueeze(-1)
            if self.mode == "group_lasso":
                self._terms.append(h.norm(dim=-1).mean())
            else:
                self._terms.append(h.abs().mean())
        else:  # SwiGLU: h = up * silu(gate)
            self._terms.append((up * s).abs().mean())

    def reset(self):
        self._terms = []
        self._below = []
        self._gate_cache = {}

    def penalty(self):
        if not self._terms:
            return torch.tensor(0.0)
        return torch.stack(self._terms).mean()

    def realized_sparsity(self):
        return float(sum(self._below) / max(1, len(self._below)))

    def remove(self):
        for h in self._handles:
            h.remove()


# ── unit test ───────────────────────────────────────────────────────────────

def _test():
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from lib.lm import make_lm
    torch.manual_seed(0)
    for kind, kw, mode in [
        ("swiglu", dict(m=32), "route_l1"),
        ("swiglu", dict(m=32), "contrib_l1"),
        ("ll1", dict(n_blocks=8, block_rank=4), "route_l1"),
        ("ll1", dict(n_blocks=8, block_rank=4), "group_lasso"),
    ]:
        model = make_lm(kind, d=32, n_heads=2, n_layers=2, vocab_size=100,
                        max_seq_len=16, **kw)
        mgr = SparsityManager(model, mode=mode)
        x = torch.randint(0, 100, (2, 8))
        mgr.reset()
        _, loss = model(x, targets=x)
        pen = mgr.penalty()
        assert pen.requires_grad and torch.isfinite(pen), (kind, mode)
        (loss + 0.1 * pen).backward()
        gnorm = sum(p.grad.abs().sum().item() for p in model.parameters()
                    if p.grad is not None)
        assert gnorm > 0
        rs = mgr.realized_sparsity()
        assert 0.0 <= rs <= 1.0
        print(f"[done] {kind:7s} {mode:12s} penalty={pen.item():.4f} "
              f"realized_sparsity={rs:.3f}")
        mgr.remove()


if __name__ == "__main__":
    _test()
