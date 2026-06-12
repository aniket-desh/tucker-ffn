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

## T+0:55 — Modules built, first structured-factor result

- lib/structured_linear.py (LowRank/BlockDiag/Monarch/Butterfly; dense-materialization
  equality tests pass to 1e-6, mixing-pattern checks pass), lib/structured_ffn.py,
  lib/sparsity.py (route_l1/contrib_l1/group_lasso with realized-sparsity logging),
  lib/interp_metrics.py (SVD canonicalization verified rotation-invariant; CKA;
  Procrustes; principal angles; local-vs-global ablation; pruning curves).
- **Confound A algebraic half settled: LL1(B,L) == tied-gate SwiGLU(BL), 0 error**, and
  LL1 is that model minus $d\,B(L-1)$ redundant gate params. The empirical half
  (atom-matched m=1992 / route-matched m=498) is training on GPU0.
- **First exp23 rows (Qwen layer 4, 0.6M budget): Monarch-SwiGLU 0.598 < dense
  SwiGLU 0.625 < blockdiag 0.648** (2 seeds, ±0.0006). Monarch's 3.9× width at
  matched params beats dense; pure block-diagonal (no mixing) is worse than dense —
  mixing is load-bearing, exactly the Monarch-vs-blockdiag prediction.
- Tucker fairness probes (30M tok): core-lr×2 = 5.483 (worse than the 5.28-ish
  warm-start trajectory at 30M in sprint 1) — boosting core lr hurts. clr×0.5 next.
- exp24 (superposition recovery) and exp25 (spectra) running; exp26 (Qwen contexts)
  written.

## T+1:50 — Three confound verdicts land

**Confound B (rank≈4) — sprint-1 claim CORRECTED.** Full spectra (exp25): trained
Tucker per-route matrices have stable rank 3.97 but numerical rank ~56 (τ=0.1) /
~116 (τ=0.01), spectral-entropy rank ~33, top-4 energy 39%. The unconstrained model
keeps a long substantial tail; "the trained dense core IS an L≈4 LL1 model" was a
stable-rank artifact. Surviving form of the claim: the top-4 directions dominate the
energy *head*, and LL1's loss-tie at L=4 shows the tail is cheap to discard at this
scale — not that it doesn't exist. (Init anchors: diagWS init stable rank=1.00,
random init 27.5; LL1 caps verified exactly: nr0.1 = L for L∈{1,2,4}; L=8 uses 8.0,
L=16 uses 16.0 — interesting: numerical rank saturates caps even when stable rank
under-saturates.)

**Confound C (Tucker fairness) — deficit not a core-lr artifact.** At 30M tokens:
default core-lr 5.36 (sprint-1 trajectory) < clr×2 5.483 < noWS 5.495 < clr×0.5
5.545. Both lr adjustments and random init hurt; the warm start remains load-bearing.

**Confound A (route/atom) — first empirical half.** atom-matched SwiGLU (m=1992
atoms+routes, +33% params): 4.716±0.018 (n=2) — beats LL1(L=4) 4.747 at equal atoms
by spending 33% more params. LL1-at-budget ≈ SwiGLU-at-budget remains a tie. So gate
tying is how you buy atoms without params; atoms-with-params is simply better
(capacity). route-matched (m=498, −67% params) running now.

**exp24 (superposition) — striking interp negative pending swap-aware rerun:** even
with NO superposition (K=48<d), task solved to 98% variance, students' atoms show
~zero alignment with the generating rank-one atoms; route-L1 does not change it.
Swap-aware scoring rerun in flight to rule out the (w↔g) metric artifact.

**exp23 (structured factors):** monarch < dense < blockdiag confirmed on layer 4 at
0.6M; ll1/butterfly rows + layer 12 pending.

## T+2:50 — Route/atom factorial complete; exp24 + exp26 verdicts

**Exp A complete (the renaming).** 100M-token runs, val loss:
| routes | atoms | FFN params | loss |
| 1493 | 1493 | 2.29M | 4.751±0.016 (swiglu, n=5) |
| 498 | 1992 | 2.29M | 4.747±0.005 (LL1 L=4, n=3) |
| 1992 | 1992 | 3.06M | 4.716±0.018 (atom-matched, n=2) |
| 498 | 498 | 0.77M | 4.887±0.001 (route-matched, n=2) |
At fixed params the route/atom split is loss-flat across a 3× route range; param count
dominates both directions. **Sprint-1's "LL1 ties SwiGLU" = "gate sharing is free";
success mode 1 (LL1 reduced to tied-gate CP) confirmed.** LL1's real deliverables:
the freedom to pick the split + the throughput edge.

**Exp D (exp24, swap-aware, 3 seeds) — ground-truth interp negative.** Recovery rate
0.00 for every architecture in every condition, including K=48 separable hub where
the task is solved to 98% variance and the generating model IS a routed-CP/LL1
tensor. Route-L1 changes nothing (λ=3e-3). Fit ordering still tracks structure (hub:
ll1_l4 0.592 best in superposition; random: swiglu 0.606 best, ll1_l4 0.681 worst;
tucker worst everywhere) — *capacity allocation* follows structure, *solution basis*
does not. High-sparsity control (p=0.08) launched.

**Exp E (exp26) — confound D vindicated on the real model.** Qwen layer-12 atoms are
token-selective above chance (mean 0.19 vs null 0.093; p90 0.35; one atom at 1.0).
Local-vs-global causal asymmetry is real: the most selective atom costs 0.102 logprob
on its top contexts vs −0.002 globally (~60×) — sprint-1's global ablation metric
would have scored it "negligible". Distilled students are MORE token-selective than
the teacher (0.28-0.31 vs 0.19); sparse-CP ≈ plain CP; LL1 slightly less selective.

Queued on GPU0: sparse_swiglu (route_l1 1e-3), sparse_ll1 (group_lasso 1e-3),
struct_monarch4 — 100M tokens each.
