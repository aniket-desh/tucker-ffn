# Structured tensor-network FFNs: sprint summary

## 1. Executive summary

This sprint asked which tensor-network structure a transformer FFN should have, using
the exact observation that SwiGLU is a routed CP tensor model (rank-one interaction
atoms gated by sigmoid routes) and Thomas Dooms's critique that dense Tucker — the
prior draft's proposal — is unprincipled, while LL1/block-CP (one gate routes a
rank-$L$ block) is the structured relaxation worth testing. We implemented the LL1 FFN
(exactly nesting SwiGLU at $L=1$ and block-sparse Tucker generally), and compared
SwiGLU / LL1 ($L\in\{1,2,4,8,16\}$) / dense Tucker at parameter- and FLOP-matched
budgets ($\pm 0.2\%$) across four experiment families. Five findings:

**1. Per-route rank is a real, bidirectionally binding capacity dial.** (Fig
exp18_lsweep) In synthetic teacher-student recovery at fixed budget, students recover
a routed tensor map to machine precision exactly when their block rank and route
count both suffice; the error minimum sits exactly at the teacher's rank, and the
"most expressive" dense core loses every cross-class comparison because its core
consumes the budget that buys routes. Evidence: positive, strong (3 seeds, $10^{-13}$
recovery floors).

**2. Real pretrained FFN functions contain routed low-rank block structure with
rank $\approx 4$–$8$.** (Fig exp21_distill) Compressing Qwen2.5-0.5B FFN layers,
LL1 ($L=4$–$16$) beats rank-one atoms in all nine layer×budget cells ($\sim 30\times$
seed noise; 2–4% relMSE) and dense Tucker loses by 35–130%. Evidence: positive,
robust, modest effect size.

**3. At LM scale the route/atom trade is loss-neutral — and LL1 is faster.** (Table
lm_final, Fig lm_curves) From-scratch 52.5M-param/100M-token training: every LL1
$L\in\{2,4,8\}$ ($n=3$ each: 4.741/4.747/4.743) sits nominally below SwiGLU ($n=5$,
$4.751\pm0.016$) but no comparison is significant (Welch $|t|\le 1.4$; pooled
$t=1.0$) — a robust tie; Tucker ($4.763\pm0.009$) is behind LL1 beyond noise
($t=3.8$ vs $L=8$). Throughput at matched FLOPs: LL1 75–76K tok/s > SwiGLU 71K >
Tucker 36.5K ($2.06\times$ slower). Evidence: tie is solid (14 runs); throughput
differences are large and real.

**4. Three independent measurements converge on per-route rank $\approx 4$.** The
unconstrained Tucker core, free to use rank 128, learns per-gate stable rank
3.97/3.99/3.97 (three fresh seeds, replicating the prior draft); LL1 capped at $L=4$
saturates to 3.26; the distillation optimum is $L\approx 4$–$8$. This is the sprint's
cleanest structural fact: dense Tucker parameterizes $\sim$91% of its budget for
interactions it does not use. Evidence: positive, triangulated.

**5. The interpretability hypothesis failed.** (Fig interp_topk) No architecture
routes sparsely (49–64% of units effectively active per token); single-unit ablations
are negligible everywhere; and under a gauge-invariant cross-seed matching test,
SwiGLU atoms ($3.0\times$ null) and even Tucker slices ($1.6\times$) recur weakly
across seeds while LL1 blocks are at chance. Structured $\ne$ stable $\ne$ legible.
The induction-circuit pilot adds a clean null (emergence speed identical across all
FFNs) plus a one-seed existence proof of an alternative dense-core copying circuit
(diffuse attention + FFN, no canonical induction head). Evidence: negative results,
honestly reported.

**Verdict on the working hypothesis:** Success mode is closest to "LL1 is a real
middle ground" on the *efficiency/structure* axes — it matches SwiGLU end-to-end
while running faster and hard-coding the rank the computation actually uses — but
Thomas's CPD-first instinct survives on the *interpretability* axis: atomized CP
remains the most seed-stable, most truncatable decomposition, and nothing we measured
says block structure makes mechanisms more accessible. Dense Tucker is dominated on
every axis and should be retired as a proposal.

## 2. Research question

What tensor-network structure gives the best tradeoff between interpretability,
expressivity, parameter efficiency, and trainability in transformer FFNs?

SwiGLU is exactly a routed CP tensor model: each hidden channel contributes one
rank-one interaction atom $u_j \otimes w_j \otimes g_j$, routed by
$\alpha_j(x) = \sigma(g_j^\top x)$. The superdiagonal latent core is simultaneously
what makes SwiGLU atomized (interpretable-by-structure) and what restricts each route
to a single rank-one interaction. Dense Tucker removes the restriction but is
gauge-entangled and spends $\sim$91% of its budget on an all-to-all core. The sprint
question, following Thomas Dooms's critique: is the structured middle — LL1/block-CP,
one gate routing a rank-$L$ block — a better point on the frontier?

## 3. Background (from the prior drafts and repo)

Established before this sprint: (i) the routed-CP decomposition is exact and
load-bearing (constant-$\alpha$ ablations cost 4–6 orders of magnitude in perplexity
on Qwen2.5-0.5B and Qwen3-1.7B); (ii) the aligned-width theorem: with fixed latent
dictionaries, an aligned SwiGLU needs $m = \sum_j \operatorname{rank}(V_j)$ units to
represent a Tucker block (generically $k^2$ balanced) — verified at machine
precision; (iii) a dense-Tucker LM trained from scratch (52.5M params, 100M
FineWeb-Edu tokens) matches SwiGLU only with a diagonal warm-start init, and its
trained per-gate stable rank concentrates at $\bar\rho \approx 4$; (iv) post-hoc
diagonalization of that model costs $518\times$ perplexity.

## 4. The reframing

Thomas Dooms: Tucker "performs much worse than CPD in a deep learning setting and is
much less interpretable. CPD gives a sparse core for free... You can make a structured
tucker by sparsifying the weights of the factor matrices using CPD (LL1)." The sprint
adopts per-gate rank $L$ as the design variable (it is exactly what the aligned-width
theorem prices), and reads the prior $\bar\rho \approx 4$ observation as a
pre-registered prediction that $L \approx 4$ should suffice at LM scale.

Budget algebra that frames everything: at matched parameters $N$, SwiGLU buys $N/3d$
atoms each with a private gate; LL1($L$) buys $\tfrac{3L}{2L+1}\times$ more atoms but
only $\tfrac{3}{2L+1}\times$ as many gates. **LL1 trades routing diversity for
interaction atoms.** (Table: tables/matched_configs.md — all configs matched within
$\pm 0.2\%$.)

## 5. LL1/block-CP FFN

$$y(x) \;=\; \sum_{b=1}^{B} U_b\,(A_b^\top x)\,\cdot\,\mathrm{SiLU}(g_b^\top x)$$

$B$ blocks, each a multilinear rank-$(L,L,1)$ term of the interaction tensor with the
gate mode rank-one. $V_b = U_b A_b^\top$ has $\operatorname{rank} \le L$ by
construction. Exactness tests pass: $L=1$ ≡ SwiGLU (0 error); LL1 ≡
block-superdiagonal Tucker ($10^{-6}$); implementation is three dense GEMMs
(SwiGLU-shaped — no batched-small-GEMM core contraction). lib/ll1_ffn.py.

## 6. Architectures tested

SwiGLU ($m=1493$) | LL1 ($L\in\{1,2,4,8,16\}$, $B$ matched) | dense Tucker
($r=s=128$, diagonal warm-start, core wd=0) — all at 2.293M $\pm 0.2\%$ FFN
params/layer, $d=512$, 8 layers, 52.5M total params. See architecture_spec.md.

## 7. Experiments run

| exp | what | status |
|---|---|---|
| exp18 | synthetic teacher-student, 3 teachers × 7 students × 3 seeds + budget sweep | done |
| exp11/sprint_lm | 52.5M-param LMs, 100M FineWeb-Edu tokens, 3–5 seeds headline + $L$-sweep | done |
| exp19 | interp proxies: eff-active, mass90, stable rank, top-$k$ decomposability, ablation locality | done |
| exp20/20b | induction pilot (6 archs × 3 seeds) + tucker mechanism probe | done |
| exp21 | Qwen2.5-0.5B FFN distillation, 3 layers × 3 budgets × 7 archs × 2 seeds | done |
| exp22 | factor stability across seeds | done |
| throughput | tokens/sec fwd + train, idle GPU | done |

## 8. Results

### 8.1 Synthetic recovery (exp18): matched structure wins, in both directions

At a fixed 9216-parameter student budget ($d=64$), each teacher is recovered
efficiently only by students whose per-route rank matches its structure:

| student | CP teacher (32 routes) | LL1 teacher ($16 \times L^*{=}4$) | Tucker teacher ($r=16$) |
|---|---|---|---|
| SwiGLU ($m=48$) | $\mathbf{1.2\times 10^{-4}}$ | $1.5\times 10^{-1}$ | $2.3\times 10^{-1}$ |
| LL1 $L=1$ ($B=48$) | $\mathbf{8.3\times 10^{-5}}$ | $1.5\times 10^{-1}$ | $2.3\times 10^{-1}$ |
| LL1 $L=2$ ($B=29$) | $4.8\times 10^{-2}$ | $4.0\times 10^{-2}$ | $2.1\times 10^{-1}$ |
| LL1 $L=4$ ($B=16$) | $2.9\times 10^{-1}$ | $\mathbf{2.1\times 10^{-13}}$ | $2.2\times 10^{-1}$ |
| LL1 $L=8$ ($B=8$) | $5.1\times 10^{-1}$ | $3.6\times 10^{-1}$ | $3.2\times 10^{-1}$ |
| LL1 $L=16$ ($B=4$) | $6.6\times 10^{-1}$ | $5.7\times 10^{-1}$ | $5.2\times 10^{-1}$ |
| Tucker ($r=17$) | $3.7\times 10^{-1}$ | $4.7\times 10^{-1}$ | $\mathbf{1.1\times 10^{-10}}$ |

(rel. val MSE, best of 3 seeds; full grid incl. means in data/exp18_results.json.)

Three observations. (1) The LL1 $L$-sweep has its minimum exactly at the teacher's
block rank, with failure on both sides: $L < L^*$ students lack atoms
($\sum_b \operatorname{rank} V_b$), $L > L^*$ students lack routes. Both resources
bind. (2) Tucker students fail on CP/LL1 teachers at matched budget — the $sr^2$ core
leaves only $r=17$ routes. Expressivity-as-superset does not survive a parameter
budget. (3) Each family recovers its own teacher to ~machine precision, so these are
optimization-reachable representations, not just existence claims. Caveat: $d=64$
gaussians, structure-matched teachers; says nothing yet about real LM computation.

### 8.2 LM training (exp11): three-way statistical tie; LL1 nominally ahead and faster

52.5M-param LMs, 100M FineWeb-Edu tokens, matched FFN budgets, identical
hyperparameters:

| arch | final val loss (mean ± sample std) | ppl | train tok/s (A40, idle) |
|---|---|---|---|
| LL1 $L=2$ ($n=3$) | $4.7409 \pm 0.0044$ | 114.5 | 76,162 |
| LL1 $L=4$ ($n=3$) | $4.7472 \pm 0.0050$ | 115.3 | 75,251 |
| LL1 $L=8$ ($n=3$) | $\mathbf{4.7427 \pm 0.0025}$ | 114.7 | 74,757 |
| SwiGLU ($n=5$) | $4.7509 \pm 0.0157$ | 115.7 | 71,373 |
| Tucker ($n=3$, diag-init) | $4.7626 \pm 0.0087$ | 117.1 | 36,545 |

Welch $t$ vs SwiGLU: $L=2$: $-1.34$, $L=4$: $-0.49$, $L=8$: $-1.15$, pooled
$L\in\{2,4,8\}$: $-1.02$ — **a statistical tie** (every $L \ge 2$ nominally below the
SwiGLU mean; LL1's seed variance is 3–6$\times$ smaller, an observation we do not
over-read at $n=3$). LL1 ($L=8$) vs Tucker: $t = -3.8$ — LL1 is ahead of dense Tucker
beyond seed noise. What is decisively not a tie: throughput. At matched parameters
and matched FLOPs ($4.59\times 10^6$ MACs/token/layer $\pm 0.2\%$), LL1 trains 5–7%
faster than SwiGLU (smaller gate GEMM) and $2.06\times$ faster than Tucker, whose
core contraction is GEMM-unfriendly. SwiGLU seed0 reproduces the prior draft's result
(4.763 vs 4.758).

Reading: gate-diversity loss ($3\times$ fewer routes at $L=4$) and atom gain (+33%)
cancel at this scale on language modeling loss; the LL1 parameterization costs
nothing end-to-end while buying throughput and bounded per-route structure.

### 8.3 Real-layer distillation (exp21): real FFN maps prefer small per-route rank > 1

Distilling Qwen2.5-0.5B FFN layers $\{4, 12, 20\}$ (13.07M params each) into students
at $\{0.6, 1.2, 2.4\}$M-param compression budgets (rel. val MSE, mean of 2 seeds,
seed spread $\pm 0.001$):

| layer, budget | SwiGLU | LL1 $L=2$ | LL1 $L=4$ | LL1 $L=8$ | LL1 $L=16$ | Tucker |
|---|---|---|---|---|---|---|
| L4, 0.6M | .6247 | .6088 | .6026 | **.6008** | .6010 | .7372 |
| L4, 2.4M | .4292 | .4135 | **.4077** | .4083 | .4145 | .7247 |
| L12, 0.6M | .5296 | .5121 | .5075 | .5051 | **.5044** | .6706 |
| L12, 2.4M | .3611 | .3496 | **.3475** | .3474 | .3497 | .6418 |
| L20, 0.6M | .4411 | .4300 | .4278 | .4262 | **.4253** | .6403 |
| L20, 2.4M | .3172 | .3120 | .3074 | **.3043** | .3046 | .7179 |

(full 9-cell grid in data/exp21_results.json; ll1_l1 ≡ swiglu within $\pm 0.001$ in
every cell — implementation control.)

The ordering LL1 ($L\approx 4$–$16$) < LL1 ($L=2$) < CP $\ll$ dense Tucker holds in
all nine cells. The LL1-over-CP gain is modest (2–4% relMSE) but $\sim 30\times$ the
seed noise and monotone in the expected direction, with a shallow optimum at
$L \approx 4$–$8$ — matching the per-gate stable rank $\bar\rho \approx 4$ that a
from-scratch dense-Tucker LM converged to in prior work. Dense Tucker is worse by
35–130% relMSE and does not improve (layer 20: worsens) with budget: at these budgets
its core forces $r$ so low (and its optimization is hard enough) that
expressivity-as-superset never materializes. Caveat: this is the compression regime
(students 4.6–18.4% of teacher size); it measures the structure of the teacher's
function, not end-to-end trainability.

### 8.4 Interpretability proxies (exp19/exp22)

**Per-route rank converges to $\approx 4$ regardless of parameterization.** The
trained dense Tucker LMs have per-gate stable rank 3.97/3.99/3.97 (mean over gates
and layers, three seeds; max 5.3) — replicating the prior draft's
$\bar\rho \approx 3.97$ on fresh seeds — despite being free to use rank 128. LL1
capped at $L=4$ saturates its cap (realized stable rank $3.26 \pm 0.01$). Together
with the distillation knee at $L \approx 4$–$8$ (§8.3), three independent
measurements point to a natural per-route interaction rank of $\sim 4$ for LM FFN
computation at this scale.

**Routing is not sparse for any architecture.** Effective active routed units per
token (exp-entropy of contribution distribution, mean over layers/seeds): SwiGLU
733/1493 (49%), LL1 312/498 (63%), Tucker 82/128 (64%). SwiGLU is the most selective
*relative to capacity*; Tucker the most diffuse; in absolute object count the
ordering reverses ($82 < 312 < 733$). Top-$k$ decomposability tells the same story
from the loss side: to stay within $\sim 0.03$ nats of base loss, SwiGLU needs
$\sim 512$ of 1493 atoms per token (34%), LL1 $\sim 256$ of 498 blocks (51%), Tucker
$\sim 128$ of 128 gates (100%).

**Single-unit ablations are uniformly negligible** (max $\Delta$loss 0.0003–0.0016
nats at layer 3 over top+random units): at 52.5M params no single atom/block/gate is
load-bearing; effects scale with unit size (Tucker's 128-gate units have
$\sim 10\times$ the median effect of SwiGLU's atoms), not with structure.

**Factor stability across seeds is weak everywhere, and LL1 is the one at chance.**
Cross-seed greedy matching: SwiGLU joint atoms $[w; g; u]$ match at cosine 0.268 vs
null 0.089 ($3.0\times$); gate directions alone are at chance for ALL architectures
(swiglu 0.154 vs null 0.155; ll1 0.135 vs 0.134; tucker 0.116 vs 0.115);
gauge-invariant per-route interaction matrices $\mathrm{vec}(V)$: swiglu 0.0100 vs
null 0.0060 ($1.7\times$), tucker 0.0118 vs null 0.0073 ($1.6\times$), ll1 0.0065 vs
null 0.0077 (at chance). LL1's theoretical identifiability advantages do not
translate into cross-seed recurrence of its learned blocks at this scale — an honest
negative for the "structured ⇒ stable factors" hypothesis (we suspect the block
partition itself, $B=498$ groups of 4 atoms, adds combinatorial assignment freedom
that atom-level CP does not have).

Tucker core diagnostics: 26.8% of core energy remains on the superdiagonal
(warm-start legacy), effective entry fraction 8% — the core is neither dense-uniform
nor block-structured.

### 8.5 LM L-sweep (complete)

| $L$ | 1 ($n=1$) | 2 ($n=3$) | 4 ($n=3$) | 8 ($n=3$) | 16 ($n=1$) | — | SwiGLU ($n=5$) | Tucker ($n=3$) |
|---|---|---|---|---|---|---|---|---|
| val loss | 4.765 | $4.741\pm0.004$ | $4.747\pm0.005$ | $4.743\pm0.003$ | 4.750 | | $4.751\pm0.016$ | $4.763\pm0.009$ |

The sweep is flat: every $L \ge 2$ sits below the SwiGLU mean (within its band);
$L=1$ reproduces SwiGLU (control); no sharp optimum — a wide loss-neutral plateau in
the route/atom trade. ($L\in\{2,4,8\}$ strengthened to 3 seeds and SwiGLU to 5 in the
seed-strengthening pass.) Realized per-route stable ranks under-saturate caps as $L$
grows ($L=2 \to 1.83$, $L=4 \to 3.26$, $L=8 \to 5.62$), and routing diffuseness rises
monotonically with $L$ (eff-active fraction $0.50 \to 0.58 \to 0.63 \to 0.65$),
interpolating from SwiGLU toward Tucker (0.65).

### 8.6 Induction pilot (exp20/exp20b)

Emergence: all FFN architectures learn the repeated-sequence induction task at
statistically identical speed (steps-to-90% accuracy: 140–180 across
swiglu/ll1_l1/ll1_l4/ll1_l16/tucker × 3 seeds); the attention-only control is
*faster* (80 steps) — at this scale the FFN is overhead for a pure-attention task,
and FFN tensor structure does not change circuit emergence speed. **Negative result
for the hypothesis that FFN structure modulates induction emergence.**

Mechanism: every swiglu/ll1 run converges to the canonical induction attention
pattern (score $\approx 1.0$ at offset $T/2 - 1$). One of three Tucker seeds reaches
100% accuracy with no canonical induction head: its layer-2 heads attend at diffuse
offsets (mass $\le 0.19$ spread over offsets 37–56 vs 0.93–1.00 at exactly offset 63
for every other run), and ablating its best head leaves accuracy at 100% (exp20b
offset profiles). A methodological caution from exp20b: zeroing all FFNs collapses
*every* architecture (SwiGLU included, to 13%), so FFN-bypass is not a valid
circuit-attribution test for jointly trained models.

## 9. Interpretability vs expressivity tradeoff

The sprint's central quantitative statement: the per-route interaction rank $L$ is a
real, measurable property of FFN computation, and at LM scale it is $\approx 4$.

- Expressivity side: dense Tucker's nominal superset capacity never pays — it loses
  every matched-budget comparison (synthetic non-Tucker teachers, real-map
  compression, throughput) and merely ties on LM loss while learning rank-4 slices.
- Interpretability side: the proxies do not support a strong "LL1 is more
  interpretable" claim. LL1 reduces the number of routed objects $3\times$ at equal
  loss and bounds each object's rank by construction (no post-hoc rank discovery
  needed), but its routing is no sparser relative to capacity, and its blocks recur
  across seeds no better than chance. SwiGLU's atoms remain the most seed-stable
  objects.
- Net: LL1's wins are concrete but engineering-flavored (throughput, compression,
  bounded structure, fewer objects); the mechanistic-interpretability advantage is
  not demonstrated by our proxies.

## 10. Circuit-level pilot

See §8.6. Null on emergence speed; single-seed existence proof that the dense-core
FFN admits an alternative copying implementation (diffuse attention + FFN). A proper
follow-up would train many seeds and characterize basin frequency vs architecture.

## 11. What changed our mind

1. Going in, dense Tucker was the prior draft's proposal and LL1 the hypothesized
   sweet spot on *interpretability*. The data moved the LL1 case from interpretability
   to *efficiency + structural honesty*: it matches the rank the computation actually
   uses, at SwiGLU speed, with no loss penalty — but the interp proxies are a wash.
2. The aligned-width theorem's role flipped: from "Tucker is $k\times$ cheaper than
   SwiGLU" to "per-route rank is the priced quantity, and its empirical value is
   small."
3. exp20b taught us that FFN-bypass is not a valid circuit test (all archs collapse);
   only the attention-offset profile cleanly separated mechanisms.
4. Factor-stability: we expected identifiability to help LL1; measured the opposite.

## 12. What failed or was inconclusive

- LL1 loss advantage over SwiGLU at LM scale: not established (tie, $|t| \le 1.4$).
- "Structured ⇒ stable factors": failed under our matching metrics.
- Sparse-CP trained variants (L1 on routes): not run (time); only post-hoc top-$k$.
- Monarch/butterfly factor axis: not run (deprioritized per plan).
- Induction emergence effects: clean null.

## 13. Limitations

- Scale: 52.5M params / 100M tokens / one dataset (FineWeb-Edu) / one tokenizer.
- Hyperparameters inherited from SwiGLU tuning (conservative for LL1 and Tucker, but
  Tucker also gets its best-known init and wd=0 core — a fairness asymmetry in
  Tucker's favor).
- $L$-sweep endpoints ($L \in \{1, 16\}$) are single-seed; $L \in \{2,4,8\}$ have 3
  seeds each.
- Interpretability proxies measure statistics of routing/structure, not semantic
  legibility; no human/auto-interp evaluation of atoms or blocks was done.
- Distillation tested the compression regime only (students $\ll$ teacher).
- exp18/exp21 students trained with one optimizer recipe; representational
  conclusions rest on best-of-seeds reaching machine precision where expected.

## 14. Next steps

1. Scale the LL1 $L$-sweep ($\ge$300M params, $\ge$1B tokens, per-arch lr tuning,
   $\ge$3 seeds) to test whether the rank-4 sweet spot and the throughput edge
   persist.
2. Trained sparsity: L1 on block routes (sparse-LL1) targeting the eff-active gap.
3. Auto-interp evaluation of blocks vs atoms (are rank-4 blocks human-describable?).
4. Monarch-factorized $A_b$/$U_b$ for the efficiency axis.
5. Inducing-basin study for the Tucker alternative-circuit observation.

## 15. Research map

- lib/ll1_ffn.py — LL1FFN + exactness tests. lib/lm.py — ll1 wiring.
- experiments/exp18_ll1_synthetic.py → results/exp18, figures exp18_*.png,
  data/exp18_results.json
- experiments/exp11_train_lm.py (extended) → results/sprint_lm, figures lm_*.png,
  tables/lm_final.md, tables/matched_configs.md
- experiments/exp19_interp_proxies.py → data/exp19_results.json, figures interp_*.png,
  tables/interp_summary.md
- experiments/exp20_induction.py, exp20b_tucker_mechanism.py → data/exp20*.json,
  figures exp20_curves.png, exp20b_offsets.png
- experiments/exp21_qwen_distill.py → data/exp21_results.json, figures exp21_distill.png
- experiments/exp22_factor_stability.py → data/exp22_results.json
- scripts/sprint_throughput.py → results/sprint_throughput.json
- docs/structured_tensor_ffn_sprint/{plan,theory_notes,architecture_spec,research_log,
  scripts_used}.md; paper draft in paper/main.tex.

## Red-team checklist (TASK §12)

1. **Did LL1 beat CP/SwiGLU at matched parameter count?** On LM loss: no — tie
   ($4.741$–$4.747$ at $L\in\{2,4,8\}$, $n=3$ each, vs $4.751\pm0.016$, $n=5$;
   $|t|\le1.4$). On real-map compression: yes, modestly but robustly (all 9 cells,
   $\sim 30\times$ seed noise). On synthetic LL1-class targets: yes, decisively. On
   throughput: yes (+5–7%).
2. **Matched FLOPs or only params?** Both — FLOP counts match within $\pm 0.2\%$ by
   construction (table in architecture_spec); throughput measured separately.
3. **Did LL1 beat dense Tucker?** Yes on every axis except LM loss, where it leads
   beyond seed noise ($t = 3.8$ at $L=8$) — and Tucker is $2.06\times$ slower.
4. **Did dense Tucker actually use dense core structure?** No: per-gate stable rank
   $\approx 4$ (replicated, 3 seeds), 27% of core energy still superdiagonal, 8%
   effective entries. This is the strongest evidence the dense core is the wrong
   parameterization.
5. **Did sparse CP improve proxies without destroying performance?** Only post-hoc
   top-$k$ was tested: all archs tolerate moderate per-token truncation; none is
   sparse. Trained-sparse variants untested.
6. **Are gains larger than seed noise?** Compression gains: yes ($\sim 30\times$
   noise). LM "gain": no — explicitly reported as a tie. Throughput: yes
   (measurement noise ~1–2%).
7. **Explained by hidden width or param count?** Param/FLOP counts matched to
   $\pm 0.2\%$; LL1's atom-count advantage at matched params is the *mechanism under
   study*, not a confound — we state it as the route/atom trade.
8. **Throughput honest?** Idle-GPU benchmark, bf16, fwd and fwd+bwd+opt, 20-iter
   timed after warmup; training step_dt logs corroborate (contended numbers higher
   for all).
9. **Interpretability measured or asserted?** Measured; results reported are mostly
   *negative* for our own hypothesis H3.
10. **Baselines tuned fairly?** Shared hyperparams tuned historically for SwiGLU;
    Tucker gets its best-known init + wd=0 core. LL1 received no dedicated tuning —
    if anything the deck was stacked against it.
11. **Did synthetic transfer to real?** Yes: the $L$-knee (synthetic, exact)
    reappears as a soft knee in real-map compression and as the learned stable rank
    of an unconstrained model.
12. **Did the induction pilot measure a circuit or only behavior?** Both: behavior
    (accuracy/emergence: null) and mechanism (attention-offset profiles: one
    alternative-circuit seed; head/FFN ablations — with the FFN-bypass caveat).
13. **Did any experiment falsify the initial story?** Yes: H3 (LL1 interpretability
    advantage) is unsupported; factor-stability went the wrong way for LL1.
14. **Simplest explanation of the results?** LM FFN computation at this scale needs
    many moderately-diffuse routed interactions of low per-route rank ($\sim 4$). Any
    parameterization that can express that cheaply reaches the same loss;
    parameterizations that cannot allocate routes (dense Tucker) waste budget.
    Nothing here requires appeal to interpretability or identifiability.
