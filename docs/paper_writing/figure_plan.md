# Figure plan (drafted before results; captions to be finalized from data)

Color semantics everywhere: SwiGLU/CP **blue**, LL1/block-CP **green**, dense Tucker
**red**, attn-only/baselines **gray**, structured-Tucker/other **orange**.

## Fig 1 — Architecture ladder (diagram, TikZ or matplotlib)
A) SwiGLU as routed CP: x → (w_j, g_j) projections → rank-1 atom $u_j \otimes w_j \otimes g_j$ routed by
   $\alpha_j(x)=\sigma(g_j^\top x)$. One gate : one atom.
B) LL1/block-CP: one gate g_b routes a rank-$L$ block $U_b A_b^\top$. One gate : L atoms.
C) Dense Tucker: all-to-all core C couples every p_i to every q_j. r gates : full-rank V_j,
   gauge-entangled.
D) The dial: per-gate rank L, with param-matched atom/gate counts annotated
   (m gates·m atoms ↔ B gates·BL atoms ↔ r gates·r·r atoms).
Caption defines atom, route, block rank.

## Fig 2 — Synthetic teacher-student (exp18)
Panels per teacher (CP / LL1 L*=4 / dense Tucker): rel-MSE vs student block rank L at
matched 9216-param budget; SwiGLU and Tucker students as horizontal reference lines;
vertical line at L*=4 in the LL1 panel. Expected story: matching structure wins; the
L-axis is a real dial (CP teacher: monotone up; LL1 teacher: knee at L*; Tucker teacher:
monotone down).

## Fig 3 — LM training (exp11/sprint_lm)
A) Final val loss vs L (LL1 sweep, seed 0) with SwiGLU and Tucker bands (3-seed mean±std)
   at matched 52.5M params, 100M tokens.
B) Val-loss curves vs tokens for swiglu/ll1_l4/tucker (3 seeds, mean±std).
C) Throughput (train tokens/sec) per arch (bar), idle-GPU benchmark.

## Fig 4 — Interpretability proxies (exp19)
A) Top-k decomposability: val loss vs k units/token (per arch; x as fraction of units
   and absolute k — choose clearer).
B) Effective active units (fraction) by layer, per arch.
C) Per-gate stable rank distributions: Tucker (learned) vs LL1 (=L by construction) vs
   SwiGLU (=1); Tucker core diagnostics inset (superdiag energy, band energy).
D) Ablation locality: sorted Δloss for single-unit ablations.

## Fig 5 — Induction pilot (exp20)
A) Second-half accuracy vs step (mean±std over 3 seeds, per arch incl. attn-only).
B) Induction score vs step.
C) If the Tucker low-induction-score effect replicates: induction score final vs arch,
   highlighting alternative-circuit finding.

Tables:
- T1: param counts/configs per arch (B, L, atoms, routes) at the matched budget.
- T2: final LM val loss/ppl per arch×seed + throughput + FLOPs.
- T3 (appendix): synthetic full grid.
