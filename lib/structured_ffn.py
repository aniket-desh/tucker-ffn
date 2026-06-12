"""FFN blocks with structured factor matrices (sprint 2, Exp B).

StructuredSwiGLU(d, m, kind, **kw)
    SwiGLU with gate/up (d->m) and down (m->d) projections replaced by a
    structured linear map of the given kind ({lowrank, blockdiag, monarch,
    butterfly}; "dense" recovers plain SwiGLU).

StructuredLL1(d, n_blocks, block_rank, kind, **kw)
    LL1 with the stacked main factor A (d->BL) and output factor U (BL->d)
    structured; the gate matrix G (d->B) stays dense (it is small and is the
    routing object under study elsewhere).

Width helpers find the widest m (or B) whose parameter count fits a budget —
at matched parameters, structured maps buy WIDER ffns; FLOPs == params for
every kind here, so parameter-matching is FLOP-matching.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .structured_linear import make_structured


def _np(mod):
    return mod.num_params() if hasattr(mod, "num_params") \
        else sum(p.numel() for p in mod.parameters())


class StructuredSwiGLU(nn.Module):
    def __init__(self, d, m, kind="monarch", **kw):
        super().__init__()
        self.d, self.m, self.kind = d, m, kind
        self.gate_proj = make_structured(kind, d, m, **kw)
        self.up_proj = make_structured(kind, d, m, **kw)
        self.down_proj = make_structured(kind, m, d, **kw)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def num_params(self):
        return _np(self.gate_proj) + _np(self.up_proj) + _np(self.down_proj)


class StructuredLL1(nn.Module):
    def __init__(self, d, routes, block_rank, kind="monarch", **kw):
        super().__init__()
        self.d, self.n_blocks, self.block_rank = d, routes, block_rank
        m = routes * block_rank
        self.m = m
        self.up_proj = make_structured(kind, d, m, **kw)
        self.gate_proj = nn.Linear(d, routes, bias=False)
        nn.init.normal_(self.gate_proj.weight, std=1.0 / math.sqrt(d))
        self.down_proj = make_structured(kind, m, d, **kw)

    def forward(self, x):
        main = self.up_proj(x)
        gate = F.silu(self.gate_proj(x))
        shape = main.shape[:-1]
        h = main.view(*shape, self.n_blocks, self.block_rank) * gate.unsqueeze(-1)
        return self.down_proj(h.view(*shape, self.m))

    def num_params(self):
        return _np(self.up_proj) + self.gate_proj.weight.numel() + _np(self.down_proj)


def _swiglu_struct_params(d, m, kind, **kw):
    tmp = StructuredSwiGLU(d, m, kind, **kw)
    n = tmp.num_params()
    del tmp
    return n


def swiglu_struct_width_for_params(d, target, kind, step=None, **kw):
    """largest m (multiple of `step`) with params <= target."""
    if step is None:
        step = kw.get("n_blocks", 4)
        if kind == "butterfly":
            step = None  # m must be a power of two (down_proj input)
    if kind == "butterfly":
        best = None
        m = 2
        while True:
            try:
                n = _swiglu_struct_params(d, m, kind, **kw)
            except AssertionError:
                break
            if n <= target:
                best = m
                m *= 2
            else:
                break
        return best
    lo, hi = step, step
    while _swiglu_struct_params(d, hi, kind, **kw) <= target:
        hi *= 2
    while hi - lo > step:
        mid = ((lo + hi) // 2 // step) * step
        if mid == lo:
            break
        if _swiglu_struct_params(d, mid, kind, **kw) <= target:
            lo = mid
        else:
            hi = mid
    return lo


def ll1_struct_blocks_for_params(d, block_rank, target, kind, **kw):
    """largest B (multiple of n_blocks) whose StructuredLL1 fits the budget."""
    step = kw.get("n_blocks", 4)
    B = step
    best = step
    while True:
        try:
            tmp = StructuredLL1(d, routes=B, block_rank=block_rank, kind=kind, **kw)
        except AssertionError:
            B += step
            if B > 100000:
                break
            continue
        n = tmp.num_params()
        del tmp
        if n <= target:
            best = B
            B += max(step, B // 8 // step * step)
        else:
            break
    return best


# ── unit tests ──────────────────────────────────────────────────────────────

def _tests():
    torch.manual_seed(0)
    d = 64
    out = []
    for kind, kw in [("monarch", {"n_blocks": 4}), ("blockdiag", {"n_blocks": 4}),
                     ("lowrank", {"rank": 16}), ("butterfly", {"n_blocks": 4})]:
        m = 128 if kind != "butterfly" else 128
        ffn = StructuredSwiGLU(d, m, kind, **kw)
        x = torch.randn(5, d)
        y = ffn(x)
        assert y.shape == (5, d)
        y.sum().backward()
        out.append((f"swiglu_{kind}", ffn.num_params(),
                    sum(p.numel() for p in ffn.parameters())))
        if kind != "lowrank":
            ll1 = StructuredLL1(d, routes=16, block_rank=8, kind=kind, **kw)
            y = ll1(x)
            assert y.shape == (5, d)
            out.append((f"ll1_{kind}", ll1.num_params(),
                        sum(p.numel() for p in ll1.parameters())))
    for name, claimed, actual in out:
        assert claimed == actual, (name, claimed, actual)
    # width search sanity
    m = swiglu_struct_width_for_params(512, 2_293_248, "monarch", n_blocks=4)
    tmp = StructuredSwiGLU(512, m, "monarch", n_blocks=4)
    assert tmp.num_params() <= 2_293_248
    return out, m


if __name__ == "__main__":
    res, m = _tests()
    for name, claimed, actual in res:
        print(f"[done] {name:18s} params={actual}")
    print(f"[done] monarch swiglu width at 2.293M budget (d=512, nb=4): m={m}")
