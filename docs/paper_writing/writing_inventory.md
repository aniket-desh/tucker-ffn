# Writing inventory (live)

## Experiments actually run (sprint)
1. lib unit tests: diag-Tucker≡SwiGLU; LL1 L=1≡SwiGLU (0 err); LL1≡block-sparse Tucker
   (1e-6); $\operatorname{rank}(V_b)\le L$; param counts exact.
2. exp18: synthetic teacher-student. 3 teachers (CP 32 atoms / LL1 16×L*=4 / dense
   Tucker r=16, all d=64, unit-variance outputs) × 7 lsweep students at 9216-param
   budget × 3 seeds + budget sweep {0.25,0.5,2}× for swiglu/ll1_l4/tucker. COMPLETE
   except tucker-teacher tail.
3. exp11/sprint_lm: 52.5M-param LMs, 100M FineWeb-Edu tokens. swiglu×3 seeds,
   tucker(diag-bias init)×3, ll1_l4×3, ll1_l{1,2,8,16}×1. IN PROGRESS.
4. exp19: interp proxies on trained LMs (eff-active, mass90, gate stats, stable rank,
   core diagnostics, single-unit ablation locality, all-layer top-k decomposability).
   Smoke-tested all arch paths. PENDING checkpoints.
5. exp20: induction pilot, 6 archs × 3 seeds. COMPLETE except final run.
6. exp20b: tucker mechanism probe (offset profiles, FFN bypass, head ablation). RUNNING.
7. exp21: Qwen2.5-0.5B FFN distillation, layers {4,12,20}, budgets {0.6M,1.2M,2.4M},
   7 archs × 2 seeds. RUNNING.
8. exp22: factor stability across seeds. Smoke-tested. PENDING checkpoints.
9. throughput bench. PENDING idle GPU.

## Figures available / planned
- fig1_ladder (diagram, drafted, needs polish)
- exp18_lsweep.png, exp18_budget.png (auto-generated; regenerate when complete)
- lm_lsweep, lm_curves (script ready)
- exp20_curves.png (done when exp20 ends), exp20b_offsets.png
- exp21_distill.png
- exp19 figures: to write plotting (top-k curve, eff-active by layer, stable-rank)

## Claims the data supports so far → see claim_registry.md

## Theory results that are exact
- routed-CP form; superdiagonal-core equivalence; aligned-width theorem + LL1 reading;
  LL1 nesting (SwiGLU at L=1, block-sparse Tucker generally); param/atom/route algebra
  ($3L/(2L+1)$ atom multiplier).

## Empirical results that are preliminary
- everything at 50M/100M-token scale (single dataset, default hyperparams, ≤3 seeds);
- induction pilot at d=128 2-layer scale;
- distillation budgets cover only the compression regime (students ≪ teacher size).

## Baselines and fairness caveats
- Tucker trained with its best-known init (diagonal warm start); its core gets wd=0
  while swiglu/ll1 weights get wd=0.1 (inherited from prior work; asymmetry noted).
- LR/schedule tuned for swiglu in prior work; reused for all archs (LL1 not separately
  tuned — conservative for LL1).
- LL1 budgets matched within ±0.2%.

## Missing experiments for a stronger paper
- multi-seed LL1 L-sweep at LM scale; >100M tokens; second dataset; trained sparse-CP
  (L1) variants; Monarch-factor axis; transfer of interp proxies to downstream tasks;
  inducing-head probe with path patching instead of ablation.
