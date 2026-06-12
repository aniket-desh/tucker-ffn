"""LL1 / block-CP ffn (sprint theory_notes section 4).

LL1FFN(d, n_blocks, block_rank) computes, per block b:

    r_b(x) = A_b^T x            in R^L      (block main response)
    s_b(x) = SiLU(g_b^T x)      scalar      (block route)
    y(x)   = sum_b U_b r_b(x) s_b(x)

i.e. grouped CP: L rank-one atoms u_{b,l} (a_{b,l}^T x) SiLU(g_b^T x) share a
single gate direction g_b. The interaction tensor is a routed sum of
multilinear rank-(L,L,1) terms

    A(x) = sum_b sigma(g_b^T x) (U_b A_b^T) (x) g_b,

the LL1 / block-term decomposition with the gate mode as the rank-1 mode.
Per-gate output-by-main matrix V_b = U_b A_b^T has rank <= L by construction,
so the aligned-width theorem's control variable (per-gate rank) is the
hyperparameter L.

Nesting:
  L=1, n_blocks=m   -> SwiGLU(d, m) exactly (test below).
  block-sparse Tucker -> LL1(B, L) equals TuckerFFN(d, r=BL, s=BL) whose core
                          is block-superdiagonal (test below).

Implementation is three dense GEMMs (same shape regime as SwiGLU; A and U are
stored stacked as (d, B*L)), plus a broadcasted multiply — no batched-small-GEMM
core contraction, so throughput should track SwiGLU, unlike TuckerFFN.

Parameter count: d*B*(2L+1)  (= 3dm at L=1,B=m).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LL1FFN(nn.Module):
    """LL1 / block-CP GLU ffn.

    weights follow HF Linear convention:
      up_proj.weight    (B*L, d)  rows are a_{b,l}^T, block-major order
      gate_proj.weight  (B, d)    rows are g_b^T
      down_proj.weight  (d, B*L)  columns are u_{b,l}
    """

    def __init__(self, d, n_blocks, block_rank, bias=False):
        super().__init__()
        self.d = d
        self.n_blocks = n_blocks
        self.block_rank = block_rank
        m = n_blocks * block_rank
        self.m = m
        self.up_proj = nn.Linear(d, m, bias=bias)
        self.gate_proj = nn.Linear(d, n_blocks, bias=bias)
        self.down_proj = nn.Linear(m, d, bias=bias)
        nn.init.normal_(self.up_proj.weight,   std=1.0 / math.sqrt(d))
        nn.init.normal_(self.gate_proj.weight, std=1.0 / math.sqrt(d))
        nn.init.normal_(self.down_proj.weight, std=1.0 / math.sqrt(m))

    def forward(self, x):
        main = self.up_proj(x)                          # (..., B*L)
        gate = F.silu(self.gate_proj(x))                # (..., B)
        shape = main.shape[:-1]
        h = main.view(*shape, self.n_blocks, self.block_rank) \
            * gate.unsqueeze(-1)                        # (..., B, L)
        return self.down_proj(h.view(*shape, self.m))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    # ── analysis helpers ────────────────────────────────────────────────────

    def block_factors(self):
        """return (A, G, U) with A (B, d, L), G (d, B), U (B, L, d)."""
        B, L, d = self.n_blocks, self.block_rank, self.d
        A = self.up_proj.weight.view(B, L, d).transpose(1, 2)   # (B, d, L)
        G = self.gate_proj.weight.T                              # (d, B)
        U = self.down_proj.weight.T.view(B, L, d)                # (B, L, d)
        return A, G, U

    def per_gate_matrices(self):
        """V_b = U_b A_b^T in (B, d, d) — rank <= L by construction."""
        A, _, U = self.block_factors()
        return torch.einsum("bdl,ble->bde", A, U).transpose(1, 2)  # (B, d, d)


def ll1_params(d, n_blocks, block_rank):
    """parameter count of LL1FFN, no bias."""
    return d * n_blocks * (2 * block_rank + 1)


def ll1_blocks_for_params(d, block_rank, target_params):
    """largest B with d*B*(2L+1) <= target (rounded)."""
    return max(1, int(round(target_params / (d * (2 * block_rank + 1)))))


# ── unit tests ──────────────────────────────────────────────────────────────

def _test_l1_equals_swiglu():
    """LL1 with L=1, B=m must equal SwiGLU(d, m) with matched weights."""
    from .tucker_ffn import SwiGLUFFN
    torch.manual_seed(0)
    d, m = 16, 12
    ll1 = LL1FFN(d, n_blocks=m, block_rank=1)
    sw = SwiGLUFFN(d, m)
    with torch.no_grad():
        sw.up_proj.weight.copy_(ll1.up_proj.weight)
        sw.gate_proj.weight.copy_(ll1.gate_proj.weight)
        sw.down_proj.weight.copy_(ll1.down_proj.weight)
    x = torch.randn(5, d)
    err = (ll1(x) - sw(x)).abs().max().item()
    assert err < 1e-6, f"L=1 vs swiglu mismatch: {err}"
    return err


def _test_equals_blocksparse_tucker():
    """LL1(B, L) must equal a TuckerFFN(r=s=BL) with block-superdiagonal core.

    Tucker with P = A_stack, Q[:, j-th block column] = g_b broadcast, R = U_stack
    and core C[o,i,j] nonzero only for o == i and i in block(j). We build the
    core explicitly and compare forwards. The Tucker gate index runs over BL
    columns; we replicate each g_b into the L columns of its block and put the
    core mass on the first gate column of each block.
    """
    from .tucker_ffn import TuckerFFN
    torch.manual_seed(1)
    d, B, L = 10, 3, 2
    BL = B * L
    ll1 = LL1FFN(d, n_blocks=B, block_rank=L)
    A, G, U = ll1.block_factors()      # (B,d,L), (d,B), (B,L,d)

    tk = TuckerFFN(d, r=BL, s=BL, diagonal_only=False)
    with torch.no_grad():
        # P columns: stacked a_{b,l}; Q columns: g_b replicated per block slot;
        # R columns: stacked u_{b,l}
        P = torch.cat([A[b] for b in range(B)], dim=1)            # (d, BL)
        Q = torch.cat([G[:, b:b+1].expand(d, L) for b in range(B)], dim=1)
        R = torch.cat([U[b].T for b in range(B)], dim=1)          # (d, BL)
        tk.P.copy_(P); tk.Q.copy_(Q); tk.R.copy_(R)
        C = torch.zeros(BL, BL, BL)
        for b in range(B):
            for l in range(L):
                i = b * L + l
                C[i, i, b * L] = 1.0   # gate via first slot of block b
        tk.C.copy_(C)
    x = torch.randn(4, d)
    err = (ll1(x) - tk(x)).abs().max().item()
    assert err < 1e-5, f"LL1 vs block-sparse tucker mismatch: {err}"
    return err


def _test_per_gate_rank():
    """per-gate matrices must have rank <= L."""
    torch.manual_seed(2)
    d, B, L = 24, 4, 3
    ll1 = LL1FFN(d, n_blocks=B, block_rank=L)
    V = ll1.per_gate_matrices()
    for b in range(B):
        r = torch.linalg.matrix_rank(V[b]).item()
        assert r <= L, f"block {b}: rank {r} > L={L}"
    return max(torch.linalg.matrix_rank(V[b]).item() for b in range(B))


def _test_param_count():
    d, B, L = 32, 7, 5
    ll1 = LL1FFN(d, n_blocks=B, block_rank=L)
    assert ll1.num_params() == ll1_params(d, B, L), \
        (ll1.num_params(), ll1_params(d, B, L))
    return ll1.num_params()


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    # re-import as package member so relative imports in tests work
    from lib.ll1_ffn import (_test_l1_equals_swiglu as t1,
                             _test_equals_blocksparse_tucker as t2,
                             _test_per_gate_rank as t3,
                             _test_param_count as t4)
    print(f"[done] ll1_l1_equals_swiglu        | err={t1():.2e}")
    print(f"[done] ll1_equals_blocksparse_tucker | err={t2():.2e}")
    print(f"[done] ll1_per_gate_rank_leq_L     | maxrank={t3()}")
    print(f"[done] ll1_param_count             | n={t4()}")
