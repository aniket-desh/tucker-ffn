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

### 8.2 LM training (exp11) — [PENDING; swiglu seed0 = 4.763 final val loss,
reproducing prior 4.758]

### 8.3 Real-layer distillation (exp21) — [PENDING]

### 8.4 Interpretability proxies (exp19/exp22) — [PENDING]

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
