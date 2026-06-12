"""Improved interpretability metrics (sprint 2, confound D).

Sprint 1 used global statistics (effective active count, mass90, whole-corpus
single-unit ablation, greedy atom-cosine matching). These can miss rare
context-specific mechanisms and rotation-equivalent blocks. This module adds:

  mine_top_contexts          per-unit top-activating token positions (+windows)
  local_vs_global_ablation   ablate a unit on its top contexts vs everywhere;
                             report local and global delta-loss
  signed_logit_contribution  direct-path contribution of a unit's output to the
                             correct-token logit (tied-embedding direct path;
                             ignores final-RMSNorm rescaling and downstream
                             layers — documented approximation)
  svd_canonicalize_block     canonical (U, S, V) for an LL1 block V_b = U_b A_b^T
  principal_angle_overlap    mean squared cosine of principal angles
  linear_cka                 CKA between two sets of unit vectors
  procrustes_similarity      orthogonal-Procrustes alignment residual
  pruning_curve              global unit removal (by importance order) vs loss

Unit contribution conventions follow exp19_interp_proxies.unit_contributions.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ── context mining ──────────────────────────────────────────────────────────

@torch.no_grad()
def mine_top_contexts(contrib, token_ids, k=20, window=8):
    """contrib: (n_positions, n_units) unit contributions for a flat stream of
    token positions; token_ids: (n_positions,) the token at each position.
    Returns {unit: [(pos, value, window_ids)]} for the top-k positions/unit.
    Positions are indices into the flat stream (sequence boundaries are the
    caller's concern; windows may cross them — fine for mining)."""
    out = {}
    n, U = contrib.shape
    for u in range(U):
        vals, idx = contrib[:, u].topk(min(k, n))
        rows = []
        for v, i in zip(vals.tolist(), idx.tolist()):
            lo = max(0, i - window)
            rows.append((i, v, token_ids[lo:i + 1].tolist()))
        out[u] = rows
    return out


def context_overlap(ctxA, ctxB, k=20):
    """Jaccard overlap of the top-k *positions* of two units (same stream)."""
    a = {p for p, _, _ in ctxA[:k]}
    b = {p for p, _, _ in ctxB[:k]}
    return len(a & b) / max(1, len(a | b))


# ── local vs global causal effect ───────────────────────────────────────────

@torch.no_grad()
def per_position_loss(model, val_inp, val_tgt, batch_size=8):
    """(n_seqs*seq_len,) per-position CE loss."""
    losses = []
    for i in range(0, val_inp.size(0), batch_size):
        ib, tb = val_inp[i:i + batch_size], val_tgt[i:i + batch_size]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(ib)
        ll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                             tb.reshape(-1), reduction="none")
        losses.append(ll.float())
    return torch.cat(losses)


class UnitAblator:
    """context manager that zeroes one routed unit in one layer's FFN."""

    def __init__(self, ffn, unit):
        from .ll1_ffn import LL1FFN
        from .tucker_ffn import SwiGLUFFN, TuckerFFN
        self.ffn, self.unit = ffn, unit
        self.handle = None
        self._tucker_col = None
        if isinstance(ffn, SwiGLUFFN):
            self.handle = ffn.up_proj.register_forward_hook(self._zero_hook)
        elif isinstance(ffn, LL1FFN):
            self.handle = ffn.gate_proj.register_forward_hook(self._zero_hook)
        elif isinstance(ffn, TuckerFFN):
            self._tucker_col = ffn.Q[:, unit].clone()
            with torch.no_grad():
                ffn.Q[:, unit] = 0  # silu(0)=0 removes the gate slice exactly

    def _zero_hook(self, mod, inp, out):
        out = out.clone()
        out[..., self.unit] = 0
        return out

    def close(self):
        if self.handle is not None:
            self.handle.remove()
        if self._tucker_col is not None:
            with torch.no_grad():
                self.ffn.Q[:, self.unit] = self._tucker_col

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@torch.no_grad()
def local_global_delta(model, ffn, unit, val_inp, val_tgt, top_positions,
                       base_pos_loss, batch_size=8):
    """delta per-position loss from ablating `unit`, summarized locally (on
    top_positions) and globally."""
    with UnitAblator(ffn, unit):
        abl = per_position_loss(model, val_inp, val_tgt, batch_size)
    delta = (abl - base_pos_loss)
    top = torch.as_tensor(sorted(top_positions), device=delta.device)
    return {
        "local_delta": delta[top].mean().item() if len(top) else float("nan"),
        "global_delta": delta.mean().item(),
        "local_max": delta[top].max().item() if len(top) else float("nan"),
    }


# ── signed logit contribution (direct path) ─────────────────────────────────

@torch.no_grad()
def signed_logit_contribution(unit_out_vec, target_ids, embed_weight):
    """direct-path contribution of a unit's output vector to the correct-token
    logits: (unit_out_vec) . E[y]. unit_out_vec: (n, d) the unit's additive
    residual contribution at each position; target_ids: (n,). Approximation:
    ignores final-norm rescaling and all downstream mixing."""
    E = embed_weight[target_ids]                      # (n, d)
    return (unit_out_vec * E).sum(-1)


# ── block canonicalization and subspace matching ────────────────────────────

def svd_canonicalize_block(U_b, A_b):
    """V_b = U_b A_b^T -> SVD V_b = P S Q^T. Returns (P*sqrt(S), Q*sqrt(S), S):
    canonical output/main factors invariant to within-block rotations."""
    V = U_b @ A_b.T if A_b.shape[0] != U_b.shape[0] else U_b @ A_b.T
    P, S, Qt = torch.linalg.svd(V, full_matrices=False)
    L = min(U_b.shape[1], A_b.shape[1])
    s = S[:L].sqrt()
    return P[:, :L] * s, Qt[:L].T * s, S[:L]


def principal_angle_overlap(A, B):
    """mean squared cosine of principal angles between span(A), span(B)."""
    Qa, _ = torch.linalg.qr(A)
    Qb, _ = torch.linalg.qr(B)
    s = torch.linalg.svdvals(Qa.T @ Qb)
    return (s ** 2).mean().item()


def linear_cka(X, Y):
    """linear CKA between feature matrices X (n, p), Y (n, q) over the same
    n examples (or unit-vector matrices over the same coordinate space)."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    num = (X.T @ Y).norm() ** 2
    den = (X.T @ X).norm() * (Y.T @ Y).norm()
    return (num / (den + 1e-12)).item()


def procrustes_similarity(A, B):
    """max_R ||A R - B||_F minimized over orthogonal R; returns
    1 - residual^2 / ||B||^2 (1 = perfectly alignable)."""
    M = A.T @ B
    U, S, Vt = torch.linalg.svd(M)
    R = U @ Vt
    res = (A @ R - B).norm() ** 2
    return (1 - res / (B.norm() ** 2 + 1e-12)).item()


# ── pruning curve ───────────────────────────────────────────────────────────

@torch.no_grad()
def pruning_curve(model, ffn, importance, val_inp, val_tgt, fracs,
                  batch_size=8):
    """globally remove the lowest-importance units (all at once per fraction)
    and measure val loss. importance: (n_units,) tensor. Returns
    [(frac_removed, loss)]. Restores the model afterwards."""
    from .ll1_ffn import LL1FFN
    from .tucker_ffn import SwiGLUFFN, TuckerFFN
    order = importance.argsort()              # ascending: prune least important
    n = len(order)
    out = []
    for frac in fracs:
        k = int(n * frac)
        units = order[:k].tolist()
        ablators = [UnitAblator(ffn, u) for u in units] if k else []
        # for swiglu/ll1 hooks stack fine; for tucker each ablator zeroes a col
        losses = []
        for i in range(0, val_inp.size(0), batch_size):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(val_inp[i:i + batch_size],
                                targets=val_tgt[i:i + batch_size])
            losses.append(loss.item())
        for a in ablators:
            a.close()
        out.append((frac, float(np.mean(losses))))
    return out


# ── unit tests ──────────────────────────────────────────────────────────────

def _tests():
    torch.manual_seed(0)
    # canonicalization invariance: rotate within-block factors, canonical forms agree
    d, L = 16, 4
    U = torch.randn(d, L)
    A = torch.randn(d, L)
    R = torch.linalg.qr(torch.randn(L, L))[0]
    U2, A2 = U @ R, A @ R          # V unchanged: U2 A2^T = U R R^T A^T = U A^T
    P1, Q1, S1 = svd_canonicalize_block(U, A)
    P2, Q2, S2 = svd_canonicalize_block(U2, A2)
    assert torch.allclose(S1, S2, atol=1e-5)
    ov = principal_angle_overlap(P1, P2)
    assert ov > 0.999, ov
    # procrustes: rotated copies align perfectly
    X = torch.randn(32, 8)
    ps = procrustes_similarity(X @ torch.linalg.qr(torch.randn(8, 8))[0], X)
    assert ps > 0.999, ps
    # cka: identical = 1, independent ~ small
    assert linear_cka(X, X) > 0.999
    assert linear_cka(X, torch.randn(32, 8)) < 0.5
    # principal angles: random subspaces in high dim have low overlap
    lo = principal_angle_overlap(torch.randn(256, 4), torch.randn(256, 4))
    assert lo < 0.1, lo
    print("[done] canonicalization invariant under within-block rotation "
          f"(overlap={ov:.4f}); procrustes={ps:.4f}; null subspace overlap={lo:.4f}")


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    _tests()
