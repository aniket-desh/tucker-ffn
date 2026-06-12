# Structured tensor-network FFNs: sprint summary

> STATUS: skeleton — numbers and verdicts filled as experiments complete. Do not cite.

## 1. Executive summary (≤600 words)
[TO FILL after results: 2–5 findings, each with one figure/table, evidence polarity.]

## 2. Research question
What tensor-network structure gives the best tradeoff between interpretability,
expressivity, parameter efficiency, and trainability in transformer FFNs? Concretely:
SwiGLU is exactly a routed CP tensor model (one gate routes one rank-one interaction
atom). Dense Tucker removes the same-index restriction but is gauge-entangled and
parameter-hungry. Is the middle ground — LL1/block-CP, where one gate routes a rank-L
block — a better point on the frontier?

## 3. Background: SwiGLU as routed CP, and why dense Tucker is suspicious
[Condensed from theory_notes §1–2: exact decomposition; prior evidence (routing
load-bearing, k² aligned separation, trained-Tucker stable rank ≈4, end-to-end tie).]

## 4. Thomas's critique and the structured-decomposition reframing
[CPD sparse core for free; identifiability; LL1 as principled relaxation; the per-gate
rank as the control variable; LL1 trades gate diversity for atoms: 3L/(2L+1)×.]

## 5. LL1/block-CP FFN: derivation
[theory_notes §4; exactness tests.]

## 6. Architecture families tested
[architecture_spec.md summary + table of matched configs.]

## 7. Experiments run
- exp18 synthetic teacher-student (CP/LL1/Tucker teachers × L-sweep students)
- exp11/sprint_lm from-scratch LMs, 52.5M params, 100M FineWeb-Edu tokens
- exp19 interpretability proxies + top-k decomposability + ablation locality
- exp20 induction pilot
- throughput benchmark

## 8. Results
[TO FILL]

## 9. Interpretability vs expressivity tradeoff
[TO FILL]

## 10. Circuit-level pilot
[TO FILL]

## 11. What changed our mind
[TO FILL]

## 12. What failed or was inconclusive
[TO FILL]

## 13. Limitations
[TO FILL — scale, single dataset, proxies-not-semantics, init asymmetries, wd asymmetry
for Tucker core, LL1 hyperparams not tuned separately.]

## 14. Next steps
[TO FILL]

## 15. Research map
[code/scripts/figures/tables index.]

## Red-team checklist (TASK §12)
[Answer all 14 explicitly at the end.]
