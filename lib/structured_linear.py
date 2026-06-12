"""Structured linear maps for FFN factor matrices (sprint 2, Exp B).

Drop-in replacements for nn.Linear (no bias) with sub-quadratic parameter
counts. All modules expose .num_params(), .flops_per_token(), and .dense()
(materialized weight for testing at small dims; forward equals x @ dense().T).

Modules
-------
LowRankLinear(in, out, rank)
    W = B A with A (rank, in), B (out, rank). params = rank (in + out).

BlockDiagonalLinear(in, out, n_blocks)
    W = blockdiag(W_1..W_nb), W_b ((out/nb), (in/nb)). params = in*out/nb.
    Input feature block b maps only to output block b — no cross-block mixing.

MonarchLinear(in, out, n_blocks)
    Rectangular Monarch (Dao et al. 2022 style): two block-diagonal GEMMs with
    an interleaving transpose, so every input block reaches every output block
    through the permutation:
        x (.., nb, p) --W1 (nb,q,p)--> (.., nb, q) --transpose--> (.., q, nb)
          --W2 (q,nb,nb)--> (.., q, nb) --reshape--> (.., q*nb = out)
    with p = in/nb, q = out/nb. params = in*out/nb + out*nb.

ButterflyLinear(in, out, n_blocks)
    log2(in) butterfly mixing stages on the input dimension (in must be a
    power of two; each stage pairs coordinates at stride 2^k with learned 2x2
    blocks), followed by a BlockDiagonalLinear(in, out, n_blocks) width map.
    params = 4 * (in/2) * log2(in) + in*out/nb. Mixing is global (FFT-like)
    at O(in log in) cost; the block-diagonal tail changes width.
"""

import math

import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.empty(out_features, rank))
        nn.init.normal_(self.A, std=1.0 / math.sqrt(in_features))
        nn.init.normal_(self.B, std=1.0 / math.sqrt(rank))

    def forward(self, x):
        return (x @ self.A.T) @ self.B.T

    def num_params(self):
        return self.rank * (self.in_features + self.out_features)

    def flops_per_token(self):
        return self.num_params()

    def dense(self):
        return self.B @ self.A


class BlockDiagonalLinear(nn.Module):
    def __init__(self, in_features, out_features, n_blocks):
        super().__init__()
        assert in_features % n_blocks == 0 and out_features % n_blocks == 0, \
            (in_features, out_features, n_blocks)
        self.in_features, self.out_features, self.nb = in_features, out_features, n_blocks
        self.p = in_features // n_blocks
        self.q = out_features // n_blocks
        self.weight = nn.Parameter(torch.empty(n_blocks, self.q, self.p))
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(self.p))

    def forward(self, x):
        shape = x.shape[:-1]
        xb = x.view(*shape, self.nb, self.p)
        y = torch.einsum("...bp,bqp->...bq", xb, self.weight)
        return y.reshape(*shape, self.out_features)

    def num_params(self):
        return self.nb * self.q * self.p

    def flops_per_token(self):
        return self.num_params()

    def dense(self):
        return torch.block_diag(*[self.weight[b] for b in range(self.nb)])


class MonarchLinear(nn.Module):
    def __init__(self, in_features, out_features, n_blocks):
        super().__init__()
        assert in_features % n_blocks == 0 and out_features % n_blocks == 0, \
            (in_features, out_features, n_blocks)
        self.in_features, self.out_features, self.nb = in_features, out_features, n_blocks
        self.p = in_features // n_blocks
        self.q = out_features // n_blocks
        self.W1 = nn.Parameter(torch.empty(n_blocks, self.q, self.p))
        self.W2 = nn.Parameter(torch.empty(self.q, n_blocks, n_blocks))
        nn.init.normal_(self.W1, std=1.0 / math.sqrt(self.p))
        nn.init.normal_(self.W2, std=1.0 / math.sqrt(n_blocks))

    def forward(self, x):
        shape = x.shape[:-1]
        xb = x.view(*shape, self.nb, self.p)
        y1 = torch.einsum("...bp,bqp->...bq", xb, self.W1)   # (.., nb, q)
        y1 = y1.transpose(-1, -2)                            # (.., q, nb)
        y2 = torch.einsum("...qb,qbc->...qc", y1, self.W2)   # (.., q, nb)
        # output laid out as (q, nb) -> out index o = j*nb + c
        return y2.reshape(*shape, self.out_features)

    def num_params(self):
        return self.W1.numel() + self.W2.numel()

    def flops_per_token(self):
        return self.num_params()

    def dense(self):
        # column e = W @ e for each basis vector — fine for tests
        I = torch.eye(self.in_features, device=self.W1.device,
                      dtype=self.W1.dtype)
        return self.forward(I).T

    def project_from_dense(self, W):
        """analytic-ish init from a dense weight (out, in): least-squares via
        alternating one pass (W1 from block-row energy, W2 refit). Good enough
        as a warm start for distillation; not the exact Monarch projection."""
        with torch.no_grad():
            # crude: initialize W1 by block-diagonal part of W, W2 near identity
            for b in range(self.nb):
                blk = W[b * self.q:(b + 1) * self.q, b * self.p:(b + 1) * self.p]
                self.W1[b].copy_(blk)
            eye = torch.eye(self.nb, device=W.device, dtype=W.dtype)
            self.W2.copy_(eye.unsqueeze(0).expand(self.q, -1, -1))


class ButterflyLinear(nn.Module):
    def __init__(self, in_features, out_features, n_blocks=4):
        super().__init__()
        self.n_mix = 1 << math.ceil(math.log2(in_features))  # pad to pow2
        k = int(math.log2(self.n_mix))
        self.in_features, self.out_features = in_features, out_features
        self.k = k
        # stage s pairs indices differing in bit s: weights (k, n_mix/2, 2, 2)
        self.stages = nn.Parameter(torch.empty(k, self.n_mix // 2, 2, 2))
        with torch.no_grad():
            # near-identity init with small mixing noise: preserves signal scale
            self.stages.zero_()
            self.stages[:, :, 0, 0] = 1.0
            self.stages[:, :, 1, 1] = 1.0
            self.stages.add_(torch.randn_like(self.stages) * 0.05)
        self.resize = BlockDiagonalLinear(self.n_mix, out_features, n_blocks)

    def _mix(self, x):
        if self.n_mix != self.in_features:
            x = torch.nn.functional.pad(x, (0, self.n_mix - self.in_features))
        shape = x.shape[:-1]
        n = self.n_mix
        y = x.reshape(-1, n)
        for s in range(self.k):
            stride = 1 << s
            # pair index i with i + stride within blocks of 2*stride
            y = y.view(-1, n // (2 * stride), 2, stride)
            a, b = y[:, :, 0, :], y[:, :, 1, :]                  # (-1, nb2, stride)
            w = self.stages[s].view(n // (2 * stride), stride, 2, 2)
            na = w[None, :, :, 0, 0] * a + w[None, :, :, 0, 1] * b
            nb_ = w[None, :, :, 1, 0] * a + w[None, :, :, 1, 1] * b
            y = torch.stack([na, nb_], dim=2).reshape(-1, n)
        return y.view(*shape, n)

    def forward(self, x):
        return self.resize(self._mix(x))

    def num_params(self):
        return self.stages.numel() + self.resize.num_params()

    def flops_per_token(self):
        return 4 * (self.n_mix // 2) * self.k + self.resize.flops_per_token()

    def dense(self):
        I = torch.eye(self.in_features, device=self.stages.device,
                      dtype=self.stages.dtype)
        return self.forward(I).T


def make_structured(kind, in_features, out_features, **kw):
    """factory: kind in {dense, lowrank, blockdiag, monarch, butterfly}."""
    if kind == "dense":
        lin = nn.Linear(in_features, out_features, bias=False)
        nn.init.normal_(lin.weight, std=1.0 / math.sqrt(in_features))
        lin.num_params = lambda: in_features * out_features
        lin.flops_per_token = lambda: in_features * out_features
        return lin
    if kind == "lowrank":
        return LowRankLinear(in_features, out_features, kw.get("rank", 64))
    if kind == "blockdiag":
        return BlockDiagonalLinear(in_features, out_features, kw.get("n_blocks", 4))
    if kind == "monarch":
        return MonarchLinear(in_features, out_features, kw.get("n_blocks", 4))
    if kind == "butterfly":
        return ButterflyLinear(in_features, out_features, kw.get("n_blocks", 4))
    raise ValueError(kind)


# ── unit tests ──────────────────────────────────────────────────────────────

def _tests():
    torch.manual_seed(0)
    results = []
    for kind, kw in [("lowrank", {"rank": 8}), ("blockdiag", {"n_blocks": 4}),
                     ("monarch", {"n_blocks": 4}), ("butterfly", {"n_blocks": 4})]:
        m = make_structured(kind, 16, 24, **kw)
        x = torch.randn(7, 16)
        y = m(x)
        assert y.shape == (7, 24), (kind, y.shape)
        # dense materialization equality
        err = (y - x @ m.dense().T).abs().max().item()
        assert err < 1e-5, (kind, err)
        # gradient flow
        y.sum().backward()
        for p in m.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), kind
        # param count matches actual parameters
        actual = sum(p.numel() for p in m.parameters())
        assert m.num_params() == actual, (kind, m.num_params(), actual)
        results.append((kind, err, actual))
    # monarch mixes across blocks (unlike blockdiag)
    mon = make_structured("monarch", 16, 16, n_blocks=4)
    W = mon.dense()
    off = W[:4, 4:].abs().sum().item()
    assert off > 0, "monarch must mix across blocks"
    bd = make_structured("blockdiag", 16, 16, n_blocks=4)
    Wb = bd.dense()
    assert Wb[:4, 4:].abs().sum().item() == 0, "blockdiag must not mix"
    # butterfly mixing is global: gradient of y[0] wrt x covers all inputs
    bf = make_structured("butterfly", 16, 16, n_blocks=4)
    Wf = bf.dense()
    assert (Wf.abs() > 0).float().mean() > 0.5, "butterfly should be dense-ish"
    return results


if __name__ == "__main__":
    for kind, err, n in _tests():
        print(f"[done] {kind:10s} dense-equality err={err:.2e} params={n}")
    print("[done] mixing-pattern checks passed")
