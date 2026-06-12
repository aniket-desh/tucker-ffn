# Claim registry (live — update as results land)

## Claim 1: SwiGLU is exactly a routed CP tensor field; LL1 is the rank-controlled family between CP and Tucker
Status: **established (algebraic)**
Evidence: derivation (theory_notes §1–4); lib/ll1_ffn.py exactness tests (L=1≡SwiGLU
0 err; LL1≡block-superdiagonal Tucker 1e-6; rank(V_b)≤L).
Alternative explanations: none (exact algebra).
Wording: strong. "LL1 exactly recovers SwiGLU as the block-rank-one case."

## Claim 2: Per-route rank L is a real, bidirectionally-binding control variable (synthetic)
Status: **established (synthetic, 3 seeds/cell)**
Evidence: exp18. LL1(L*=4) teacher: student error minimum exactly at L=4 (2e-13
machine precision, 2/3 seeds) with failures both below (L<4: too few atoms, relMSE
0.04–0.17) and above (L=8: too few routes, 0.36). CP teacher: monotone degradation
with L (0.0001 → 0.67). [Tucker teacher: pending.]
Alternative explanations: optimization noise — addressed by 3 seeds + machine-precision
floor; param mismatch — budgets matched within 8% (report exact).
Wording: medium-strong. "In teacher-student recovery, error is minimized exactly at the
teacher's per-route rank, with degradation in both directions at matched budget."

## Claim 3: At matched params at LM scale, [LL1 vs SwiGLU vs Tucker ordering]
Status: **pending** (exp11/sprint_lm; 3 seeds swiglu/tucker/ll1_l4; L-sweep 1 seed)
Evidence so far: swiglu seed0 ~57M tokens val 4.94 (on track to ≈4.75 like prior work).
Risks: single dataset, 100M tokens, hyperparams tuned on swiglu defaults.

## Claim 4: Real pretrained FFN maps are best compressed by [CP/LL1/Tucker?]
Status: **pending** (exp21 Qwen2.5-0.5B distillation)

## Claim 5: Interpretability proxies order as SwiGLU ≤ LL1 < Tucker (diffuseness)
Status: **pending** (exp19, exp22)
Sub-claims to check: Tucker learned per-gate stable rank ≈4 replicates across 3 seeds;
top-k decomposability curves; ablation locality; factor stability vs null.

## Claim 6: FFN tensor structure does not affect induction emergence at pilot scale
Status: **likely (negative result)** — 13/18 runs in: steps-to-90% identical
(140–180) across swiglu/ll1_l1/ll1_l4/ll1_l16; attn-only faster (80). Tucker pending;
earlier toy-scale run showed tucker solving the task without the canonical
induction-attention pattern (ind score 0.045) — check replication at scale.
Wording if confirmed: "We find no evidence that FFN tensor structure changes
induction-circuit emergence at this scale; the pilot's value is methodological."

## Claim 7 (efficiency): LL1 throughput ≈ SwiGLU > Tucker at matched params
Status: **pending** (scripts/sprint_throughput.py on idle GPU; training step_dt
already suggests tucker ~2.1x slower per step than swiglu: 950ms vs 450ms).
