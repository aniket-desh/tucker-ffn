# Sprint plan: structured tensor-network FFNs between routed CP and dense Tucker

Sprint start (UTC): 2026-06-12T03:22:06Z. Hard wall-clock ceiling: 24h (target ~16-20h
including writing). Hardware: 2× NVIDIA A40 (46GB), 96 CPU, 503GB RAM.

## Central question

What tensor-network structure gives the best tradeoff between interpretability,
expressivity, parameter efficiency, and trainability in transformer FFNs?

## What the repo already establishes (from the scrapped workshop draft)

1. **Exact routed-CP view.** SwiGLU $= \sum_j \alpha_j(x)\, u_j \otimes w_j \otimes g_j$
   with $\alpha_j(x) = \sigma(g_j^\top x)$. Algebraically exact. Verified empirically
   (routing is input-dependent and load-bearing: constant-$\alpha$ ablations cost 4–6
   orders of magnitude in perplexity on Qwen2.5-0.5B and Qwen3-1.7B).
2. **Aligned-width theorem.** With fixed latent dictionaries $P, Q$, an aligned SwiGLU
   needs $m_{\min} = \sum_j \operatorname{rank}(V_j)$ units to exactly reproduce a
   Tucker block ($V_j = R\,C^{(j)}$). Generic core ⇒
   $m_{\min} = r \min(r,s) = k^2$ balanced. Verified at machine precision (exp10).
3. **Trained Tucker LMs use off-diagonal core structure.** Diagonal projection costs
   $518\times$ perplexity; per-gate stable rank $\approx 4$ (not 1, not dense $\sim r$).
4. **End-to-end at 50M params / 100M tokens, Tucker ≈ SwiGLU** (within seed noise,
   with variance-preserving init). No throughput win (75K vs 112K tok/s on A100).

## What changed (Thomas Dooms's critique)

Dense Tucker is the wrong proposal: empirically weaker than CPD in deep learning, and
less interpretable (dense core, gauge freedom). CP's superdiagonal core is a *feature*
(sparse core for free). The principled question: which structured decomposition
between CP and dense Tucker is right? His pointer: LL1 (rank-$(L,L,1)$ block-term
decomposition) = sparsify the Tucker core via block structure; Monarch/butterfly for
the efficiency axis.

Key reframing the old draft missed: **the trained-Tucker stable-rank result
($\bar\rho \approx 4$) is itself evidence for LL1** — the model uses per-gate rank
~4, far below dense (128) but above 1 (SwiGLU). The natural architecture makes this
rank a hyperparameter.

## Dmitry Manning-Coe's suggestion

After the architecture sweep, test on a known circuit (induction heads): does FFN
tensor structure change emergence/robustness/interpretability of a recognizable
mechanism?

## Hypotheses going in (each falsifiable)

H1. LL1 students recover LL1 teachers with a knee at
$L_{\text{student}} = L_{\text{teacher}}$ (clean theory test).

H2. At matched parameters, LL1 with small $L$ (2–8) matches or beats SwiGLU and dense
Tucker on LM validation loss (because trained Tucker is effectively low per-gate rank
anyway).

H3. LL1's interpretability proxies (active blocks per token, ablation locality) sit
between SwiGLU and dense Tucker, closer to SwiGLU.

H4. Dense Tucker's learned core is diffuse (high core entropy / no block structure) —
its expressivity is not used in structured form.

What would kill the story: LL1 loses to SwiGLU at matched params at every $L$; or
LL1's proxies are no better than dense Tucker's. Either is a reportable negative.

## Phases

1. Setup + reproduction smoke tests (done: lib unit tests pass).
2. Literature notes (Dooms/Pearce bilinear MLPs, LL1/BTD, Monarch) + theory_notes.md.
3. Implement LL1GLUFFN + tests ($L=1$ ≡ grouped CP; shapes; param counts).
4. Synthetic teacher-student: CP / LL1($L=2,4$) / Tucker teachers × CP / LL1
   ($L$ sweep) / Tucker students, matched params, ≥3 seeds at key points.
5. From-scratch LM training on FineWeb-Edu: swiglu / tucker / ll1 ($L$ sweep) at
   matched ~50M params; 3 seeds for the headline comparison; throughput accounting.
6. Interpretability proxies on trained models: per-token active atoms/blocks,
   contribution entropy, ablation locality, per-gate stable rank, factor stability
   across seeds.
7. Induction/copying pilot (2-layer attn+FFN models on synthetic repeated-pair task).
8. Sprint summary + red-team (≥90 min).
9. Paper writing per WRITING_PAPER.md (inventory → claims → figures → draft →
   red-team), iterating back into experiments if a claim is under-supported.

## GPU budget sketch

- A40 ≈ half an A100 for this workload ⇒ ~45 min per 100M-token 50M-param run.
- Phase 5 needs ~9–12 runs (3 archs × 3 seeds + $L$ sweep) ⇒ parallelize across 2
  GPUs, ~5–7h wall. Start these as early as possible; run synthetic sweeps on the
  other GPU.
- Smoke-test everything at 1/100 scale first.
