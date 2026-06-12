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
