# Red-team pass (paper/main.tex @ final draft)

## Strongest defensible claim
"Per-route interaction rank is a measurable property of FFN computation, ≈4 at this
scale (triangulated 3 ways), and a block-CP parameterization delivers it at zero LM
loss cost with a throughput gain." — defensible; every number is multi-seed or
~30× noise.

## What a reviewer attacks first, and our answer
1. "LL1 doesn't beat SwiGLU, so what's the point?" → We never claim a loss win; the
   claims are (a) the dial exists and binds, (b) real maps prefer L>1, (c) the dense
   core is dominated, (d) loss-neutrality + speed at LM scale. The abstract/intro/
   discussion all state the tie explicitly.
2. "Scale is small." → Stated in every caveat + limitations; framed as pilot;
   concrete next step given.
3. "Distillation gains are 2-4% — is that meaningful?" → Reported as modest;
   significance via seed spread (±0.001); all 9 cells agree; we don't oversell.
4. "Hyperparameters tuned for SwiGLU." → Disclosed twice (experiments + limitations);
   notes the asymmetries *favoring* Tucker too (init, wd=0 core).
5. "Interpretability proxies are not interpretability." → We agree; the section's
   claims are mostly negative; abstract says "deliberate negative."
6. "exp20 anomaly is n=1." → Explicitly labeled single-seed observation.
7. "Stable rank ≈4 may be init-biased (diagonal warm start)." → TRUE RISK. Prior
   work shows random-init Tucker reaches stable rank ≈27.6 — so the rank-4 value is
   conditional on the warm start (the only init that matches SwiGLU loss). Mitigation:
   the distillation knee and LL1 cap-saturation are init-independent corroborations.
   ACTION TAKEN: added a sentence to §4.4 noting the init-conditionality.
8. "Throughput depends on implementation quality." → All archs use plain PyTorch
   GEMM paths, no custom kernels for anyone; stated.

## Checks performed
- All numbers in the paper traced to results JSON/logs (lm_final.md, exp18/21/19/22
  JSONs). ✓
- Gauge-freedom transpose fixed (C^(j) ↦ C^(j)M). ✓
- "Identifiable parameterization" claim softened (cross-ref to negative result). ✓
- Captions self-contained with seed counts. ✓
- Tie language used consistently (abstract, §4.3, discussion). ✓
- L=1 ≡ SwiGLU control reported in 3 places (unit test, distillation ±0.001, LM
  4.765 in swiglu range). ✓
- Figure colors consistent (CP blue / LL1 green / Tucker red). ✓ (fig1 panel D,
  exp18, exp21, lm figures, interp figures)
- No "novel framework"/"significantly improves" filler. ✓

## Remaining weaknesses accepted
- L-sweep single-seed off L=4 (disclosed).
- exp19 ablation on one layer (3) only; defensible as proxy, disclosed in appendix.
- No trained-sparsity variant; listed as next step.
- Tucker n=3 in LM but n=2 in some interp metrics (seed2 analyzed in exp19b and
  merged numbers reported as three seeds where available).
