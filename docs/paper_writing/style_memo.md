# Style memo (target papers studied 2026-06-12)

Sources: Pearce/Dooms et al. 2410.08417 ("Bilinear MLPs enable weight-based mechanistic
interpretability", v2 HTML), Manning-Coe et al. 2502.01739 ("Grokking vs. Learning"),
plus the structural guidance in WRITING_PAPER.md and Nanda's advice.

## Dooms/Pearce structural moves to imitate

1. **Abstract shape**: (i) field-level true statement of an obstacle ("mechanistic
   understanding of MLPs remains elusive"), (ii) the specific technical blocker
   ("element-wise nonlinearities introduce higher-order interactions"), (iii) the object
   they introduce + its exact property ("fully expressed ... using a third-order tensor"),
   (iv) what the analysis reveals ("interpretable low-rank structure across ..."),
   (v) demonstrations, (vi) takeaway ("drop-in replacement", "weight-based interpretability
   is viable"). One claim per sentence. No "novel framework" fluff.
2. **Exactness as the selling point**: "can be *fully* expressed", "the decomposition is
   exact". Our analogue: SwiGLU *is* a routed CP tensor field, exactly, pointwise in x.
3. **Method menu sections**: they enumerate analysis routes (direct interactions /
   eigendecomposition / HOSVD) and say the choice depends on what's known. Our analogue:
   the ladder CP → LL1(L) → Tucker indexed by per-gate rank.
4. **Hedging style**: "there are no guarantees the eigenvectors will be monosemantic";
   "we expect ... may limit". Empirical observations marked as such ("surprisingly
   low-rank"). Never claim interpretability without a metric.
5. **Captions**: panel-by-panel, defining symbols inline, self-contained ("Multiplying
   the bilinear tensor by output direction u produces an interaction matrix Q that can
   be decomposed...").

## Manning-Coe moves to imitate

1. Sharp opening question; controlled comparisons designed around it.
2. Transitions: "Having established X, we now ask Y. To make this comparison clean, we
   choose Z." Use these verbatim patterns between theory → synthetic → LM → proxies.
3. Hand-built task-specific measures (their compressibility; our per-gate stable rank,
   effective active blocks, recovery knee).
4. Honest deflation when an object is not yet practical.

## House rules for our draft

- Claims-first: 2-3 claims max, each tied to one figure.
- Tense: present for math, past for experiments run.
- "principled" only with the principle named; "interpretable" only with the proxy named.
- \citep for background, \citet for actors. No "[29] shows".
- Colors: SwiGLU/CP blue, sparse-CP teal, LL1 green, structured-Tucker orange, dense
  Tucker red, baselines gray. Same across all figures.
- Every experiment subsection: Question / Setup / Metric / Result / Interpretation / Caveat.
