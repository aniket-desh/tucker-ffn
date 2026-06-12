# Broad tensor-network FFN sweep + interpretability diagnosis: sprint 2 summary

> STATUS: in progress — sparse LM runs, monarch LM run, exp23b structured
> distillation completing.

## 1. Executive summary
[TO FILL LAST]

## 2. What the previous sprint found

LL1/block-CP ties SwiGLU on LM loss at matched budget while running 5–7% faster;
real FFN maps prefer $L\approx4$–$16$ blocks in compression; dense Tucker dominated;
"per-route rank ≈ 4" triangulated three ways; interpretability proxies negative.

## 3. What this sprint corrected

1. **"LL1" is not a new tensor architecture; it is gate sharing.** LL1$(B,L)$ equals
   a width-$BL$ SwiGLU with gates tied in groups of $L$ (verified, 0 error), minus
   the $dB(L-1)$ redundant gate parameters. The route/atom factorial (§6) shows loss
   is *flat* in the route/atom split at fixed parameters — sprint 1's "tie" was the
   sound of a free parameter being moved along a flat direction.
2. **"Per-route rank ≈ 4" was metric- and init-conditioned.** Full spectra: trained
   warm-start Tucker has stable rank 3.97 but numerical rank ≈ 56 ($\tau{=}0.1$) /
   ≈ 116 ($\tau{=}0.01$), spectral-entropy rank ≈ 33, top-4 energy 39%; without the
   diagonal warm start even the stable rank is 15.5. What survives: the top-4
   directions dominate the energy head, and discarding the tail is *cheap*
   (LL1 $L{=}4$ ties on loss) — not that the tail is absent.
3. **Tucker's deficit is not a core-lr artifact.** Core-lr $\times2$ (5.483),
   $\times0.5$ (5.545), and random init (5.495) all do worse at 30M tokens than the
   default warm-start trajectory (~5.36). The warm start is load-bearing; per-arch lr
   tuning does not rescue the dense core.
4. **Sprint-1's interpretability metrics were too blunt — and fixing them split the
   verdict in two** (§8, §9): context-specific causal structure EXISTS in real models
   (and global ablations wash it out), but architectural/sparsity choices do NOT make
   training find generative atoms even when ground truth is available.

## 4. Architecture lattice tested

See architecture_lattice.md. Cells run: CP/SwiGLU, sparse-CP (route-L1, contrib-L1,
group lasso), tied-gate CP (≡ LL1, verified), LL1, dense Tucker (+3 fairness
variants), low-rank/block-diagonal/Monarch/butterfly factor matrices in SwiGLU and
LL1 (distillation + LM for monarch), atom-/route-matched SwiGLU controls.
Deliberately skipped, with reasons pre-registered in theory_notes.md: BTD with
gate-mode rank > 1, TT/MPO, softmax/entmax routing.

## 5. Theory: the route/atom trade and structured factor axes

At budget $N$: SwiGLU buys $N/3d$ atoms with private routes; LL1($L$) buys
$\tfrac{3L}{2L+1}\times$ atoms at $\tfrac{3}{2L+1}\times$ routes; structured factor
matrices buy *width*: Monarch ($nb{=}4$) affords $3.9\times$ the hidden width of
dense at equal params/FLOPs. Params = FLOPs for every family tested, so parameter
matching is FLOP matching throughout; only wall-clock distinguishes hardware reality.

## 6. Experiment A: route/atom confound — RESOLVED

100M FineWeb-Edu tokens, $d=512$, identical hyperparameters:

| config | routes | atoms | FFN params/layer | val loss |
|---|---|---|---|---|
| SwiGLU $m{=}1493$ ($n{=}5$) | 1493 | 1493 | 2.29M | $4.751 \pm 0.016$ |
| LL1 $L{=}4$ ($n{=}3$) | 498 | 1992 | 2.29M | $4.747 \pm 0.005$ |
| atom-matched SwiGLU $m{=}1992$ ($n{=}2$) | 1992 | 1992 | 3.06M | $4.716 \pm 0.018$ |
| route-matched SwiGLU $m{=}498$ ($n{=}2$) | 498 | 498 | 0.77M | $4.887 \pm 0.001$ |

**Verdict:** at fixed parameters, loss is flat in the route/atom split across a
3× route range; parameter count dominates in both directions. Tied-gate CP *is* LL1
(algebraic identity). The correct name for sprint-1's finding is: **gate sharing is
loss-free at this scale, and it buys throughput** (LL1 75–76K tok/s vs SwiGLU 71K at
matched FLOPs). There is no LL1-specific representational advantage on language loss.

## 7. Experiment B: structured factor matrices

Qwen2.5-0.5B FFN distillation (rel. val MSE, 2 seeds, spread ±0.001), layer 4 at
0.6M params: **Monarch 0.598 < dense 0.625 < block-diagonal 0.648**; low-rank
[0.754 at smoke scale — final numbers pending exp23b]; butterfly: too slow in our
PyTorch implementation to run at full steps (caveated short-run number pending).
[FULL exp23b TABLE + LM run for monarch PENDING]

Early reading: width-via-structure beats dense at matched budget when the structure
mixes globally (Monarch); block-diagonal (no mixing) is worse than dense — mixing,
not just width, is load-bearing. This is the first positive evidence for Thomas's
efficiency axis in this project.

## 8. Experiment C: trained sparsity — [LM RUNS IN FLIGHT]

Infrastructure: route-L1 / contribution-L1 / group-lasso penalties with
realized-sparsity logging. In the ground-truth setting (exp24), route-L1 at
$\lambda \in \{3 \times 10^{-3}\}$ changed neither fit nor recovery. LM-scale runs
(swiglu+route_l1, ll1+group_lasso at $\lambda{=}10^{-3}$) pending: the question is
where the loss-vs-active-units frontier moves.

## 9. Experiments D/E: where the interpretability question actually got answered

**D (superposition recovery, ground truth known).** Generative model: sparse latents,
target = sum of 32 rank-one bilinear atoms; topologies matching CP (random pairs)
and LL1 (hubs); three regimes (superposition $K{=}96{>}d$; separable $K{=}48$;
separable + high sparsity $p{=}0.08$); 5 architectures × 3 seeds; swap-aware,
SVD-canonicalized recovery scoring. Results: fit quality tracks structure (hub →
LL1 best; random → CP best; Tucker worst everywhere — replicating exp18's
matched-structure law on a third task family), but **ground-truth atom recovery is
0.00 for every architecture in every regime** (mean matched score ≤ 0.32), including
when the task is solved to 98% variance and the generating model is exactly a
routed-CP tensor the student class contains. Route-L1 changes nothing.
*Architecture choice allocates capacity along the right structure; it does not make
SGD select the generative basis.*

**E (real pretrained layer, context-specific metrics).** Qwen2.5-0.5B layer 12,
200K tokens: atom token-selectivity is real (top-20-context modal-token fraction
0.19 vs 0.093 shuffled null; p90 = 0.35; max = 1.0). The most selective atom loses
0.102 logprob on its top contexts when ablated vs −0.002 globally (~60× local/global
asymmetry): **sprint-1's global single-unit ablations (max Δ 0.0016) were measuring
the wrong thing — rare context-specific causal atoms exist.** Distilled students
(1.2M params) have *higher* token selectivity than the teacher (0.28–0.31 vs 0.19);
sparse-CP ≈ plain CP; LL1 blocks slightly less selective per-unit.

## 10. What failed

- Butterfly: our stage-loop PyTorch implementation is latency-bound (~2.2 s/step at
  $d{=}896$) — unusable without a fused kernel; only a short-run caveated number.
- Ground-truth atom recovery: failed for every architecture and penalty (the finding).
- Tucker rescue attempts (core-lr, random init): all worse.
- [exp23b ll1-structured rows pending]

## 11. What changed our mind

1. We went in believing the route/atom trade might be a real architectural effect;
   it is a flat direction — capacity dominates.
2. We believed (sprint 1) per-route rank ≈ 4 was a robust fact; it is a
   head-of-spectrum statement conditioned on the warm start.
3. We believed global interp metrics showed "no structure"; context-specific metrics
   show selective, locally-causal atoms in the real model — the metrics, not only
   the architectures, were the problem.
4. We assumed identifiability (CP uniqueness) would help training find true atoms in
   at least the easiest setting; it does not.

## 12. Limitations

- Same single scale/dataset/tokenizer as sprint 1 for LM runs; $n \le 3$ seeds.
- Sparse LM runs cover one $\lambda$ per penalty (frontier not swept).
- Butterfly untested at full training (implementation, not concept).
- exp24's generative family is one task; recovery failure may be regime-specific
  (though three regimes × two topologies all agree).
- Token-selectivity is a crude semantic proxy (modal current token); no human or
  LLM-judge evaluation of contexts.
- Monarch LM evidence is distillation + [1 LM run pending]; no scaling claim.

## 13. Recommended next direction

The interpretability question has moved: not "which architecture is interpretable"
but **"under what training conditions does the learned basis align with a generative
basis, and can context-specific causal atoms be reliably surfaced?"** Concretely:
(1) dictionary-learning-style post-hoc rotation of trained FFN atoms toward sparsity
(the SAE/transcoder move) measured against exp24's ground truth — does post-hoc
rotation succeed where architectural priors failed?; (2) Monarch-SwiGLU at 10×
scale with measured wall-clock (the one axis where structure delivered unambiguous
wins); (3) local-ablation-based atom auditing as a standard metric replacing global
ablation.

## 14. Research map

- lib/structured_linear.py, lib/structured_ffn.py — structured factor modules+FFNs
- lib/sparsity.py — trained sparsity penalties; lib/interp_metrics.py — improved
  metrics (SVD canonicalization, principal angles, CKA, Procrustes, local ablation)
- experiments/exp23_structured_distill.py → results/exp23, exp23b
- experiments/exp24_superposition.py → results/exp24{,_easy,_sparse}
- experiments/exp25_spectra.py → results/exp25{,b}
- experiments/exp26_qwen_contexts.py → results/exp26 (incl. contexts_teacher.txt)
- LM runs → results/s2_lm/{atom_matched,route_matched,tucker_clr2,tucker_clr05,
  tucker_noWS,sparse_swiglu,sparse_ll1,monarch}
- docs/broad_tn_sprint/{plan,theory_notes,architecture_lattice,research_log}.md

## 15. Red-team checklist — [TO ANSWER AT END]
