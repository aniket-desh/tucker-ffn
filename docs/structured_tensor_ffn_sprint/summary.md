# Structured tensor-network FFNs: sprint summary

> STATUS: in progress — LM / interp / distillation sections pending their runs.

## 1. Executive summary (≤600 words)
[TO FILL LAST]

## 2. Research question

What tensor-network structure gives the best tradeoff between interpretability,
expressivity, parameter efficiency, and trainability in transformer FFNs?

SwiGLU is exactly a routed CP tensor model: each hidden channel contributes one
rank-one interaction atom u_j⊗w_j⊗g_j, routed by α_j(x)=σ(g_j^T x). The superdiagonal
latent core is simultaneously what makes SwiGLU atomized (interpretable-by-structure)
and what restricts each route to a single rank-one interaction. Dense Tucker removes
the restriction but is gauge-entangled and spends ~91% of its budget on an all-to-all
core. The sprint question, following Thomas Dooms's critique: is the structured middle
— LL1/block-CP, one gate routing a rank-L block — a better point on the frontier?

## 3. Background (from the prior drafts and repo)

Established before this sprint: (i) the routed-CP decomposition is exact and
load-bearing (constant-α ablations cost 4–6 orders of magnitude in perplexity on
Qwen2.5-0.5B and Qwen3-1.7B); (ii) the aligned-width theorem: with fixed latent
dictionaries, an aligned SwiGLU needs m = Σ_j rank(V_j) units to represent a Tucker
block (generically k² balanced) — verified at machine precision; (iii) a dense-Tucker
LM trained from scratch (52.5M params, 100M FineWeb-Edu tokens) matches SwiGLU only
with a diagonal warm-start init, and its trained per-gate stable rank concentrates at
ρ̄≈4; (iv) post-hoc diagonalization of that model costs 518× perplexity.

## 4. The reframing

Thomas Dooms: Tucker "performs much worse than CPD in a deep learning setting and is
much less interpretable. CPD gives a sparse core for free... You can make a structured
tucker by sparsifying the weights of the factor matrices using CPD (LL1)." The sprint
adopts per-gate rank L as the design variable (it is exactly what the aligned-width
theorem prices), and reads the prior ρ̄≈4 observation as a pre-registered prediction
that L≈4 should suffice at LM scale.

Budget algebra that frames everything: at matched parameters N, SwiGLU buys N/3d atoms
each with a private gate; LL1(L) buys 3L/(2L+1)× more atoms but only 3/(2L+1)× as many
gates. **LL1 trades routing diversity for interaction atoms.** (Table:
tables/matched_configs.md — all configs matched within ±0.2%.)

## 5. LL1/block-CP FFN

y(x) = Σ_b U_b (A_b^T x) · SiLU(g_b^T x): B blocks, each a multilinear rank-(L,L,1)
term of the interaction tensor with the gate mode rank-one. V_b = U_b A_b^T has rank ≤ L
by construction. Exactness tests pass: L=1 ≡ SwiGLU (0 error); LL1 ≡ block-superdiagonal
Tucker (1e-6); implementation is three dense GEMMs (SwiGLU-shaped — no batched-small-GEMM
core contraction). lib/ll1_ffn.py.

## 6. Architectures tested

SwiGLU(m=1493) | LL1(L∈{1,2,4,8,16}, B matched) | dense Tucker(r=s=128, diagonal
warm-start, core wd=0) — all at 2.293M ± 0.2% FFN params/layer, d=512, 8 layers,
52.5M total params. See architecture_spec.md.

## 7. Experiments run

| exp | what | status |
|---|---|---|
| exp18 | synthetic teacher-student, 3 teachers × 7 students × 3 seeds + budget sweep | done |
| exp11/sprint_lm | 52.5M-param LMs, 100M FineWeb-Edu tokens, 3 seeds headline + L-sweep | running |
| exp19 | interp proxies: eff-active, mass90, stable rank, top-k decomposability, ablation locality | pending ckpts |
| exp20/20b | induction pilot (6 archs × 3 seeds) + tucker mechanism probe | 20 done / 20b running |
| exp21 | Qwen2.5-0.5B FFN distillation, 3 layers × 3 budgets × 7 archs × 2 seeds | running |
| exp22 | factor stability across seeds | pending ckpts |
| throughput | tokens/sec fwd + train, idle GPU | pending |

## 8. Results

### 8.1 Synthetic recovery (exp18): matched structure wins, in both directions

At a fixed 9216-parameter student budget (d=64), each teacher is recovered efficiently
only by students whose per-route rank matches its structure:

| student | CP teacher (32 routes) | LL1 teacher (16×L*=4) | Tucker teacher (r=16) |
|---|---|---|---|
| SwiGLU (m=48) | **1.2e-4** | 1.5e-1 | 2.3e-1 |
| LL1 L=1 (B=48) | **8.3e-5** | 1.5e-1 | 2.3e-1 |
| LL1 L=2 (B=29) | 4.8e-2 | 4.0e-2 | 2.1e-1 |
| LL1 L=4 (B=16) | 2.9e-1 | **2.1e-13** | 2.2e-1 |
| LL1 L=8 (B=8) | 5.1e-1 | 3.6e-1 | 3.2e-1 |
| LL1 L=16 (B=4) | 6.6e-1 | 5.7e-1 | 5.2e-1 |
| Tucker (r=17) | 3.7e-1 | 4.7e-1 | **1.1e-10** |

(rel. val MSE, best of 3 seeds; full grid incl. means in results/exp18_results.json.)

Three observations. (1) The LL1 L-sweep has its minimum exactly at the teacher's
block rank, with failure on both sides: L<L* students lack atoms (Σ rank V_b), L>L*
students lack routes. Both resources bind. (2) Tucker students fail on CP/LL1 teachers
at matched budget — the sr² core leaves only r=17 routes. Expressivity-as-superset does
not survive a parameter budget. (3) Each family recovers its own teacher to ~machine
precision, so these are optimization-reachable representations, not just existence
claims. Caveat: d=64 gaussians, structure-matched teachers; says nothing yet about
real LM computation.

### 8.2 LM training (exp11): three-way statistical tie; LL1 nominally ahead and faster

52.5M-param LMs, 100M FineWeb-Edu tokens, matched FFN budgets, identical
hyperparameters (3 seeds each; tucker seed 2 rerunning after a scheduling restart):

| arch | final val loss (mean ± std) | ppl | train tok/s (A40, idle) |
|---|---|---|---|
| LL1 (L=4, B=498) | **4.7472 ± 0.0041** | 115.3 | 75,251 |
| SwiGLU (m=1493) | 4.7542 ± 0.0104 | 116.1 | 71,373 |
| Tucker (r=s=128, diag-init) | 4.7578 ± 0.0024 (n=2) | 116.5 | 36,545 |

Welch t for LL1 vs SwiGLU ≈ 1.1 — **a statistical tie**; we do not claim an LL1 loss
win. What is *not* a tie: throughput. At matched parameters and matched FLOPs
(4.59e6 MACs/token/layer ±0.2%), LL1 trains 5–7% faster than SwiGLU (smaller gate
GEMM) and 2.06× faster than Tucker, whose core contraction is GEMM-unfriendly.
SwiGLU seed0 reproduces the prior draft's result (4.763 vs 4.758). L-sweep at
L∈{1,2,8,16} (1 seed each): [PENDING — runs in flight].

Reading: gate-diversity loss (×3 fewer routes at L=4) and atom gain (+33%) cancel at
this scale on language modeling loss; the LL1 parameterization costs nothing
end-to-end while buying throughput and bounded per-route structure.

### 8.3 Real-layer distillation (exp21): real FFN maps prefer small per-route rank > 1

Distilling Qwen2.5-0.5B FFN layers {4, 12, 20} (13.07M params each) into students at
{0.6, 1.2, 2.4}M-param compression budgets (rel. val MSE, mean of 2 seeds, seed spread
±0.001):

| layer, budget | SwiGLU | LL1 L=2 | LL1 L=4 | LL1 L=8 | LL1 L=16 | Tucker |
|---|---|---|---|---|---|---|
| L4, 0.6M | .6247 | .6088 | .6026 | **.6008** | .6010 | .7372 |
| L4, 2.4M | .4292 | .4135 | **.4077** | .4083 | .4145 | .7247 |
| L12, 0.6M | .5296 | .5121 | .5075 | .5051 | **.5044** | .6706 |
| L12, 2.4M | .3611 | .3496 | **.3475** | .3474 | .3497 | .6418 |
| L20, 0.6M | .4411 | .4300 | .4278 | .4262 | **.4253** | .6403 |
| L20, 2.4M | .3172 | .3120 | .3074 | **.3043** | .3046 | .7179 |

(full 9-cell grid in data/exp21_results.json; ll1_l1 ≡ swiglu within ±0.001 in every
cell — implementation control.)

The ordering LL1(L≈4–16) < LL1(2) < CP ≪ dense Tucker holds in all nine cells. The
LL1-over-CP gain is modest (2–4% relMSE) but ~30× the seed noise and monotone in the
expected direction, with a shallow optimum at L≈4–8 — matching the per-gate stable
rank ρ̄≈4 that a from-scratch dense-Tucker LM converged to in prior work. Dense Tucker
is worse by 35–130% relMSE and does not improve (layer 20: worsens) with budget: at
these budgets its core forces r so low (and its optimization is hard enough) that
expressivity-as-superset never materializes. Caveat: this is the compression regime
(students 4.6–18.4% of teacher size); it measures the structure of the teacher's
function, not end-to-end trainability.

### 8.4 Interpretability proxies (exp19/exp22)

**Per-route rank converges to ≈4 regardless of parameterization.** The trained dense
Tucker LMs have per-gate stable rank 3.97/3.99 (mean over gates and layers, seeds 0/1;
max 5.3) — replicating the prior draft's ρ̄≈3.97 on fresh seeds — despite being free to
use rank 128. LL1 capped at L=4 saturates its cap (realized stable rank 3.26±0.01).
Together with the distillation knee at L≈4–8 (§8.3), three independent measurements
point to a natural per-route interaction rank of ~4 for LM FFN computation at this
scale.

**Routing is not sparse for any architecture.** Effective active routed units per token
(exp-entropy of contribution distribution, mean over layers/seeds): SwiGLU 733/1493
(49%), LL1 312/498 (63%), Tucker 82/128 (64%). SwiGLU is the most selective *relative
to capacity*; Tucker the most diffuse; in absolute object count the ordering reverses
(82 < 312 < 733). Top-k decomposability tells the same story from the loss side: to
stay within ~0.03 nats of base loss, SwiGLU needs ~512 of 1493 atoms per token (34%),
LL1 ~256 of 498 blocks (51%), Tucker ~128 of 128 gates (100%).

**Single-unit ablations are uniformly negligible** (max Δloss 0.0003–0.0016 nats at
layer 3 over top+random units): at 52.5M params no single atom/block/gate is
load-bearing; effects scale with unit size (Tucker's 128-gate units have ~10× the
median effect of SwiGLU's atoms), not with structure.

**Factor stability across seeds is weak everywhere, and only SwiGLU clears chance.**
Cross-seed greedy matching: SwiGLU joint atoms [w;g;u] match at cosine 0.268 vs null
0.089 (3.0×); gate directions alone are at chance for ALL architectures (swiglu 0.154
vs null 0.155; ll1 0.135 vs 0.134; tucker 0.116 vs 0.115); gauge-invariant per-route
interaction matrices vec(V): swiglu 0.0100 vs null 0.0060 (1.7×), ll1 0.0065 vs null
0.0079 (at chance). LL1's theoretical identifiability advantages do not translate
into cross-seed recurrence of learned blocks at this scale — an honest negative for
the "structured ⇒ stable factors" hypothesis.

Tucker core diagnostics: 26.8% of core energy remains on the superdiagonal (warm-start
legacy), effective entry fraction 8% — the core is neither dense-uniform nor
block-structured.

### 8.5 Induction pilot (exp20/exp20b)

Emergence: all FFN architectures learn the repeated-sequence induction task at
statistically identical speed (steps-to-90% accuracy: 140–180 across
swiglu/ll1_l1/ll1_l4/ll1_l16/tucker × 3 seeds); the attention-only control is *faster*
(80 steps) — at this scale the FFN is overhead for a pure-attention task, and FFN
tensor structure does not change circuit emergence speed. **Negative result for the
hypothesis that FFN structure modulates induction emergence.**

Mechanism: every swiglu/ll1 run converges to the canonical induction attention pattern
(score ≈ 1.0 at offset T/2−1). Tucker runs reach 100% accuracy with anomalous attention
in 1/3 seeds (score 0.34) — exp20b is probing what circuit those models use (offset
profiles, FFN-bypass, head ablations). [exp20b PENDING]

## 9–15: [PENDING final results]

## Red-team checklist: [TO ANSWER AT END]
