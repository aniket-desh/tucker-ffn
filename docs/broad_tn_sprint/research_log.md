# Research log — broad TN sprint (sprint 2)

Sprint start (UTC): **2026-06-12T16:29Z**. Ceiling 24h → stop by 2026-06-13T16:29Z.
Sprint 1 ended ~11:00Z same day; GPUs idle since; all sprint-1 checkpoints intact in
results/sprint_lm.

## T+0:00 — Orientation (required first entry)

**1. What the previous sprint established.**
(a) Per-route rank $L$ is a bidirectionally binding capacity dial (synthetic recovery
to $10^{-13}$ exactly at matched structure). (b) Real Qwen FFN maps are better
compressed by LL1 $L\approx4$–$16$ than rank-one atoms (all 9 cells, small but
$\sim30\times$ noise); dense Tucker far worse. (c) LM scale (52.5M/100M tok, 14
runs): LL1 $L\in\{2,4,8\}$ ties SwiGLU (every point nominally lower, $|t|\le1.4$),
beats Tucker ($t=3.8$); LL1 5–7% faster than SwiGLU, $2.06\times$ faster than
Tucker. (d) Per-route stable rank $\approx4$ triangulated three ways. (e)
Interpretability proxies negative: no sparse routing, negligible single-unit
ablations, LL1 blocks at chance cross-seed.

**2. What it did NOT establish.**
- The Monarch/butterfly/low-rank/block-diagonal factor-matrix axis (never run).
- Trained sparsity (only post-hoc top-$k$).
- Whether LL1's effect is anything beyond the route/atom trade (LL1 ≡ tied-gate
  SwiGLU; never compared against atom-matched/route-matched SwiGLU controls).
- Whether rank$\approx$4 is init/metric-induced (only stable rank, only diagonal
  warm-start Tucker).
- Whether Tucker loses for representational vs optimization/budget reasons.
- Whether the interp negatives are real or artifacts of global-average metrics and
  rotation-naive matching.
- Any FFN-loaded mechanistic task (induction is attention-dominated).

**3. Most serious confounds.**
(A) Route/atom: LL1's nominal LM edge may be purely "more atoms at same params."
(B) Stable-rank≈4 may be diag-warm-start + stable-rank-insensitivity artifact.
(C) Tucker's losses may be budget accounting ($sr^2$ eats 91%) + no per-arch tuning.
(D) Interp metrics: global averages and greedy cosine matching can miss rare
context-specific mechanisms and rotation-equivalent blocks.

**4. First architecture families to test and why.**
Order of information value per GPU-hour: (i) Exp-A LM controls (atom-matched
$m=1992$ / route-matched $m=498$ SwiGLU) — two cheap runs that can *rename the
headline finding*; (ii) Tucker fairness runs (random init + tuned core lr; low-rank
core) — can reverse a major negative; (iii) structured factor matrices in
distillation+throughput (Thomas's explicit ask, cheap on the distillation harness);
(iv) trained-sparsity LM runs; (v) superposition-recovery task (ground-truth atoms =
the only setting where "interpretability" has an objective answer); (vi) Qwen
top-context analysis with the new metrics.

**5. What would make me pivot.**
If atom-matched SwiGLU (more params) clearly beats LL1 and route-matched SwiGLU
(fewer params) matches it, the whole "structure" framing collapses to "atom count at
any cost" and the sprint pivots to the route/atom scaling law. If Monarch/butterfly
matches dense at half params AND real throughput parity, the efficiency axis becomes
the headline. If SVD-canonicalized matching flips the LL1 stability result, sprint-1's
interp negative was a measurement artifact and the interp story reopens.
