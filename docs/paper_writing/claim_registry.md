# Claim registry (final)

## Claim 1: SwiGLU is exactly a routed CP tensor field; LL1 is the rank-controlled family between CP and Tucker
Status: **established (algebraic)**
Evidence: derivation (theory_notes §1–4); lib/ll1_ffn.py exactness tests ($L=1$ ≡
SwiGLU, 0 err; LL1 ≡ block-superdiagonal Tucker, $10^{-6}$;
$\operatorname{rank}(V_b) \le L$).
Alternative explanations: none (exact algebra).
Wording: strong. "LL1 exactly recovers SwiGLU as the block-rank-one case."

## Claim 2: Per-route rank $L$ is a real, bidirectionally-binding control variable (synthetic)
Status: **established (synthetic, 3 seeds/cell)**
Evidence: exp18. LL1($L^*=4$) teacher: student error minimum exactly at $L=4$
($2\times 10^{-13}$ machine precision, 2/3 seeds) with failures both below ($L<4$:
too few atoms, relMSE 0.04–0.17) and above ($L=8$: too few routes, 0.36). CP teacher:
monotone degradation with $L$ ($10^{-4} \to 0.67$). Tucker teacher: only the Tucker
student recovers ($1.1\times 10^{-10}$).
Alternative explanations: optimization noise — addressed by 3 seeds +
machine-precision floor; param mismatch — budgets matched within 8% (reported).
Wording: medium-strong. "In teacher-student recovery, error is minimized exactly at
the teacher's per-route rank, with degradation in both directions at matched budget."

## Claim 3: At matched params at LM scale, LL1/SwiGLU/Tucker ordering
Status: **established as a tie (LL1 vs SwiGLU); LL1 > Tucker beyond noise**
Evidence: 14 runs. SwiGLU $4.7509 \pm 0.0157$ ($n=5$); LL1 $4.7409/4.7472/4.7427$ at
$L=2/4/8$ ($n=3$ each); Tucker $4.7626 \pm 0.0087$ ($n=3$). Welch $|t| \le 1.4$ for
all LL1-vs-SwiGLU; $t = 3.8$ for LL1($L=8$) vs Tucker. Throughput: LL1 75–76K >
SwiGLU 71K > Tucker 36.5K tok/s at matched FLOPs.
Risks: single dataset, 100M tokens, hyperparams tuned on swiglu defaults (disclosed).

## Claim 4: Real pretrained FFN maps are best compressed by LL1 with $L \approx 4$–$16$
Status: **established (modest effect, robust)**
Evidence: exp21, all 9 layer×budget cells, 2 seeds (spread $\pm 0.001$), gains 2–4%
relMSE over CP; Tucker worse by 35–130%.

## Claim 5: Interpretability proxies order as SwiGLU ≤ LL1 < Tucker (diffuseness)
Status: **partially established; the stronger LL1-interpretability hypothesis failed**
Evidence: eff-active fraction 0.49 / 0.63 / 0.64 (SwiGLU / LL1$_{L=4}$ / Tucker),
monotone in $L$; Tucker learned stable rank $\approx 3.97$ replicated on 3 seeds;
factor stability: SwiGLU atoms $3\times$ null, Tucker $\mathrm{vec}(V)$ $1.6\times$
null, LL1 blocks at chance.

## Claim 6: FFN tensor structure does not affect induction emergence at pilot scale
Status: **established (negative result)** — steps-to-90% identical (140–180) across
all FFN archs × 3 seeds; attn-only faster (80). One Tucker seed solves the task
without the canonical induction attention pattern (verified via offset profiles +
head ablation; $n=1$, reported as an observation).
Wording: "We find no evidence that FFN tensor structure changes induction-circuit
emergence at this scale."

## Claim 7 (efficiency): LL1 throughput ≈ SwiGLU > Tucker at matched params
Status: **established, stronger than hypothesized** — LL1 is 5–7% *faster* than
SwiGLU (76.2K vs 71.4K train tok/s) and $2.06\times$ faster than Tucker (36.5K),
idle-A40 benchmark at matched FLOPs ($4.59\times 10^6$ MACs/token/layer $\pm 0.2\%$).
