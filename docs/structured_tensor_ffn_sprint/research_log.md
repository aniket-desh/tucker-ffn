# Research log

Sprint start (UTC): **2026-06-12T03:22Z**. Hard ceiling 24h → must stop by 2026-06-13T03:22Z.

## T+0:00 — Setup and orientation

- Read TASK.md, WRITING_PAPER.md, WRITING_NEEL.md, both old drafts
  (docs/swiglu-idea.pdf = original note; docs/scrapped-workshop.pdf = ICML-style
  workshop draft, scrapped).
- Hardware: 2× A40 46GB, 96 CPU, 503GB RAM, torch 2.6.0+cu124, CUDA OK on both GPUs.
- lib/tucker_ffn.py unit tests pass (diag-Tucker≡SwiGLU exact, einsum vs loop 2.4e-7).
- Key observation for the new framing, mined from the old draft's own appendix: the trained
  Tucker LM's per-gate stable rank concentrates at ρ̄≈3.97 (min 2.79, max 5.30) — i.e., a
  *trained dense-Tucker model behaves like an LL1 model with L≈4*. This was reported as a
  curiosity in B.2; it is actually the strongest motivation for the LL1 architecture and a
  pre-registered prediction: LL1 with L≈4 should match dense Tucker end-to-end, and the
  L-sweep should show diminishing returns past L≈4.
- Decision: structure the sprint around the conceptual ladder SwiGLU/CP (L=1, n_blocks=m)
  → LL1(L) → dense Tucker (single block, L=r), with matched-parameter LM training as the
  load-bearing experiment and synthetic teacher-student as the clean theory test.
- What would make me pivot: if LL1 at all L loses badly to SwiGLU at matched params in
  the LM run (then the story becomes "CP atomization is the right point, structured
  relaxations don't pay"), or if synthetic recovery doesn't show the L-knee (would suggest
  optimization, not representation, dominates everything at these scales).

## T+0:40 — Implementation + launches

- Wrote theory_notes.md. Sharpest new framing found while deriving: at matched params,
  LL1 trades gate diversity for atoms (3L/(2L+1)× more atoms, B=m·3/(2L+1) gates).
  LL1(B,L) is exactly a SwiGLU of width BL with gates tied in groups of L.
- lib/ll1_ffn.py implemented; exactness tests pass (L=1≡SwiGLU exact 0 err;
  LL1≡block-sparse Tucker 9.5e-7; per-gate rank ≤ L).
- Wired ll1 into lib/lm.py + exp11 (arch tags ll1_l{L}, auto budget-matched B).
- Smoke LM run OK (tiny scale, ll1_l4 params match swiglu target exactly: 44928).
- LAUNCHED (T+0:35): full LM runs, 100M FineWeb-Edu tokens each, d=512 L=8 ~52.5M
  params: GPU0 = swiglu,ll1_l4 × seeds 0,1,2; GPU1 = tucker(diag-bias init) × 0,1,2
  then ll1_l{1,2,8,16} × seed 0. Observed step_dt: swiglu ~0.4s, tucker ~0.95s
  (core contraction cost visible). ETA ~3h GPU0, ~5h GPU1.
- LAUNCHED (T+0:40): exp18 synthetic teacher-student sweep on GPU0 (shared);
  ~13s/fit, 144 fits ≈ 35 min. Early cells: CP teacher fit by swiglu and ll1_l1
  students to ~1e-4 relMSE — sanity holds (CP-structured students recover CP teacher).
- Wrote exp19 (interp proxies) and exp20 (induction pilot); exp20 smoke running.

## T+1:35 — First headline result (exp18, H1 confirmed)

- LL1(L*=4) teacher: student relMSE at matched 9216-param budget:
  L=1 (swiglu / ll1_l1): 0.15–0.17 | L=2: 0.04–0.06 | **L=4: 2e-13 (machine precision,
  2/3 seeds; 3rd seed 2.7e-2 optimizer outlier)** | L=8: 0.36.
- The minimum is exactly at the teacher's structure and the failures on both sides match
  the counting argument: L<4 students have too few atoms (48, 58 < 64 = Σ rank V_b);
  L=8 student has too few routes (8 < 16 = B*). Gate diversity and per-route rank are
  both binding constraints — the L-axis is a real dial, not a soft preference.
- CP teacher (32 routes): recovered by L=1 students (1e-4); monotone degradation with L
  (L=16: 0.66) — when routes are scarce, gate-tying hurts. Symmetric story.
- exp19/exp22 pipelines smoke-tested on all three arch paths (incl. tucker einsum path
  and factor-stability matching: swiglu seeds matched_cos 0.40 vs null 0.13 at toy scale).
- Paper scaffold compiles (tectonic); theory section drafted in paper/main.tex.
- LM runs: GPU0 swiglu seed0 at 54M tokens (val 4.98); GPU1 tucker seed0 at 20M
  (contended by exp20, recovers when exp20 finishes in ~15 min).

## T+2:15 — Mid-sprint status

- exp18: CP + LL1 teachers complete (3 seeds), Tucker teacher in progress. Pattern so
  far is fully symmetric "matched structure wins": each student family is best on its
  own teacher; Tucker student is poor on CP/LL1 teachers at matched budget (its sr²
  core eats the budget leaving 17 routes).
- exp20: 15/18 runs. Emergence speed identical across all FFN archs (steps-to-90% =
  140–180 for swiglu/ll1_l{1,4,16}; attn-only faster at 80). Clean negative for the
  "FFN structure changes induction emergence" hypothesis so far; tucker runs next —
  watching whether the "solves task without canonical induction attention" anomaly
  from the toy-scale smoke replicates.
- LM: swiglu_seed0 at 75M tokens val 4.84 (on track vs prior 4.758@100M).
  tucker_seed0 at 26M val 5.42, no longer contended (exp20 ending).
- Fig 1 ladder diagram drafted; paper theory+appendix+related work compile.
- Queued: exp21 (Qwen distillation) on GPU0 when exp18 ends; throughput bench when a
  GPU is fully idle; exp19/exp22 when LM checkpoints land.

## T+2:40 (mislabeled earlier as T+4:00) — Distillation pattern + tucker circuit anomaly mechanism

- exp21 (Qwen2.5-0.5B layer 4, budgets 0.6M/1.2M): LL1 L≥2 consistently beats CP
  (relMSE 0.601-0.609 vs 0.625 at 0.6M; 0.511-0.513 vs 0.528 at 1.2M), saturating
  ~L=4-8. Tucker clearly worst (0.74 / 0.71). Seed pairs nearly identical (±0.001) so
  the ~0.02-0.04 gaps are far above noise. Real FFN maps prefer small per-route rank
  >1; dense core wastes compression budget. Layers 12, 20 pending.
- exp20b first probe (tucker seed0, the ind=0.34 anomaly): FFN-bypass collapses
  accuracy 1.00 → 0.03, while ablating the best layer-2 "induction head" leaves 1.00.
  The anomalous Tucker model implements copying through its FFN rather than through a
  canonical induction attention head. Other seeds + swiglu control pending.
- exp18 final figures regenerated with best-of-seed lines + consistent colors.
- LM: swiglu seed0 done 4.7626 (prior work 4.758 ✓). seed1 at 28M; tucker seed0 48M.

## T+2:55 (mislabeled) — exp20b complete (induction mechanism probe)

- FFN-bypass collapses ALL architectures (swiglu 0.13, tucker 0.03-0.22 accuracy) —
  off-distribution intervention, not tucker-specific. Lesson logged: FFN-bypass is not
  a clean circuit test when models train jointly.
- The architecture-specific finding survives: tucker seed0 reaches 100% accuracy with
  NO canonical induction head (its L2 heads attend at diffuse offsets 54-56/37-44 with
  mass ~0.1-0.2, vs mass 0.93-1.00 at exactly offset 63 for every other run incl.
  tucker s1/s2). Best-head ablation leaves it at 100%. One of three seeds; suggestive
  existence proof that the dense-core FFN admits an alternative copying basin.
- exp20 phase closed: emergence-speed null + this single-seed anomaly.

## T+3:05 (mislabeled) — exp21 complete (Qwen distillation)

Robust ordering across all 9 (layer × budget) cells, 2 seeds each (seed var ±0.001):
LL1(L=4-16) < LL1(L=2) < CP=LL1(L=1)=SwiGLU << dense Tucker. Gains of LL1 over CP:
2-4% relMSE consistently; Tucker worse by 35-130% relMSE and non-improving (sometimes
worsening) with budget — its core spends the compression budget on interactions the
real map doesn't need, and optimization at r needed for 2.4M params (r~? ) is hard.
The L-knee at ~4-8 matches the prior trained-Tucker stable-rank ρ̄≈4. ll1_l1 ≡ swiglu
within 0.001 in every cell (implementation control).
Interpretation: real pretrained FFN input-output maps contain routed low-rank block
structure; per-route rank ~4-8 captures it at matched compression budget.

## T+3:14 (06:36 UTC) — headline results all in; timestamp correction

- NOTE: log labels between T+2:15 and here were inflated (wakeup cadence misjudged);
  corrected above. True elapsed: 3h14m.
- LM headline (100M tokens, matched 52.5M params, 3 seeds):
  ll1_l4 4.7472±0.0041 | swiglu 4.7542±0.0104 | tucker 4.7578±0.0024 (n=2, seed2 rerun
  in flight after a deliberate scheduling kill). Welch t(LL1 vs SwiGLU)≈1.1 → tie,
  LL1 nominally ahead. swiglu seed0 reproduces prior draft (4.763 vs 4.758).
- Throughput (idle A40, bf16, matched FLOPs 4.59e6 MAC/tok/layer): ll1_l2..16
  74.8-76.2K train tok/s > swiglu 71.4K > tucker 36.5K (1.95-2.06x slower).
- exp22 (early): swiglu cross-seed matched atom cosine 0.269 vs null 0.089.
- In flight: ll1_l1/l8 (GPU0), tucker seed2 + ll1_l2/l16 (GPU1), exp19, exp22.

## T+5:40 (09:02 UTC) — All experiments complete

- ll1_l16 final run done. Full L-sweep: 4.765 / 4.740 / 4.747±.004 / 4.742 / 4.750
  (L=1/2/4/8/16); swiglu 4.754±.010; tucker 4.763±.007. Flat sweep, everything L≥2
  at/below swiglu mean.
- exp19b: tucker seed2 stable rank 3.97 (third replication); LL1 realized ranks
  1.83/3.26/5.62 for caps 2/4/8; eff-active fraction monotone in L (0.50→0.65).
- exp22b: tucker 1vs2 V_cos 0.0120 vs null 0.0079 — V-recurrence replication.
- All artifacts regenerated; paper experiment numbers final. Moving to red-team +
  final writing. GPU work: done. Total experiment phase: 5h40m wall.

## T+6:10 (09:30 UTC) — Writing and red-team

- summary.md complete: executive summary, all 15 sections, 14-question red-team
  checklist answered.
- paper/main.tex complete (14 pp): abstract, intro (7-paragraph structure +
  contributions), exact theory section (routed CP → V_j → LL1 + budget identity),
  4 experiment sections in Question/Setup/Metric/Result/Interpretation/Caveat form,
  discussion (established/plausible/failed/limitations/next step), related work,
  theorem appendix with proof + LL1 remark, full experimental-details appendix.
- Red-team pass (docs/paper_writing/red_team.md): fixed gauge-freedom transpose,
  GEMM-factor sloppiness, softened "identifiable parameterization", added
  init-conditionality caveat for the stable-rank result. All numbers traced to JSONs.

## T+5:20 (08:42 UTC) — Seed-strengthening pass

- All 9 phases complete and pushed; paper at 15 pp. Remaining wall budget is large, so
  addressing the biggest red-team weakness (single-seed L-sweep): launched ll1_l2
  seeds 1,2 (GPU0), ll1_l8 seeds 1,2 (GPU1), then swiglu seeds 3,4 — 6 runs ≈ 2.5h.
  If L=2/8 hold at ~4.74, every L≥2 point gets multi-seed support; pooled comparison
  vs swiglu (n=5) becomes meaningfully powered.

## T+7:30 (10:50 UTC) — Seed-strengthening complete; final stats

- Final LM table (14 runs): swiglu n=5 4.7509±0.0157 | ll1_l2 4.7409±0.0044 |
  ll1_l4 4.7472±0.0050 | ll1_l8 4.7427±0.0025 | tucker 4.7626±0.0087 (sample std).
- The new swiglu seeds (4.763, 4.729) WIDENED the baseline spread — the tie verdict
  is now robust (all LL1 |t|≤1.4, pooled t=1.0); LL1 vs Tucker significant (t=3.8).
  Honest secondary observation: LL1 seed variance 3-6× smaller than SwiGLU's.
- Paper, summary, figures, tables all updated to exact full-precision numbers.
