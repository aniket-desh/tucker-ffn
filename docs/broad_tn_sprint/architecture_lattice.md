# Architecture lattice (sprint 2)

Three axes; cells marked ✓ are tested this sprint, ◐ partially, ✗ deliberately skipped.

## Axis 1: core / tensor structure

| structure | interaction tensor | per-route object | gauge / identifiability | status |
|---|---|---|---|---|
| CP / SwiGLU | $\sum_j \alpha_j(x)\,u_j\otimes w_j\otimes g_j$ | rank-1 atom | Kruskal-unique (perm+scale) | ✓ (baseline) |
| sparse CP (trained) | same + L1 on routes/contributions | rank-1 atom, few active | same + sparsity prior | ✓ Exp C |
| tied-gate CP | groups of $L$ atoms share $g_b$ | rank-$L$ block | = LL1 exactly (verified, 0 err) | ✓ Exp A |
| LL1 / block-CP | $\sum_b \alpha_b(x)\,(U_bA_b^\top)\otimes g_b$ | rank-$L$ block | block-unique up to within-block rotation | ✓ (sprint 1 + controls) |
| sparse/block Tucker | Tucker core w/ L1 or block masks | sparse slices | partial gauge | ◐ (diag-init Tucker is block-ish; trained-L1 core skipped for time) |
| dense Tucker | $\sum_{ij} C_{oij}$ all-to-all | full-rank $V_j$ | $P$-side gauge | ✓ (+ fairness controls Exp A') |
| general BTD $(L_1,L_2,L_3)$ | rank>1 in gate mode | multi-gate block | partial | ✗ (skipped: gate-mode rank>1 destroys the scalar-route reading that motivates the family; noted as future work) |
| TT/MPO interaction tensor | chained 3-tensors | bond-dim slices | TT gauge | ✗ (skipped for time; the TT prior answers a different question — sequential mode coupling — and the factor-matrix axis was higher priority per Thomas) |

## Axis 2: factor-matrix structure (per projection)

| factor | params (in$\to$out) | mixing pattern | hardware | status |
|---|---|---|---|---|
| dense | $io$ | full | 1 GEMM | ✓ |
| low-rank $r$ | $r(i+o)$ | full, rank-limited | 2 GEMMs | ✓ Exp B |
| block-diagonal $nb$ | $io/nb$ | none across blocks | batched GEMM | ✓ Exp B |
| Monarch $nb$ | $io/nb + o\,nb$ | full (via permutation) | 2 batched GEMMs + transpose | ✓ Exp B |
| butterfly ($\log$ stages + blockdiag resize) | $2\,n\log_2 n + io/nb$ | full, hierarchical | $\log n$ elementwise stages | ✓ Exp B |
| Kronecker | $\sqrt{io}$-ish | structured | reshape GEMMs | ✗ (Monarch subsumes the relevant trade at these sizes) |
| TT/MPO linear | bond-dependent | structured | chain GEMMs | ✗ time |

Applied to: SwiGLU $W/G/U$ (all three), LL1 $A/U$ (G kept dense — it is the routing
object under study on axis 3).

## Axis 3: routing structure

| routing | status |
|---|---|
| SiLU gate (sigmoid route) | ✓ baseline |
| L1 on route activations | ✓ Exp C |
| L1 on contributions | ✓ Exp C |
| group lasso on blocks (LL1) | ✓ Exp C |
| top-$k$ routes (inference) | ✓ (sprint-1 top-$k$ decomposability) |
| top-$k$ training / softmax / entmax | ✗ (known-unstable at sprint scale; L1 family covers the sparsity question) |

## Confound-control cells (Exp A / A')

| model | params | routes | atoms | tests |
|---|---|---|---|---|
| SwiGLU $m=1493$ | 2.293M | 1493 | 1493 | baseline |
| LL1 $L=4$, $B=498$ | 2.293M | 498 | 1992 | sprint-1 result |
| atom-matched SwiGLU $m=1992$ | 3.06M (+33%) | 1992 | 1992 | is atom count at ANY cost what matters? |
| route-matched SwiGLU $m=498$ | 0.765M (−67%) | 498 | 498 | how much do 498 private-gate atoms alone buy? |
| tied-gate CP | = LL1 | — | — | algebraic identity (verified 0 err) |
| Tucker, core-lr $\times2$ / $\times0.5$ | 2.296M | 128 | — | optimization fairness |
| Tucker, no warm start | 2.296M | 128 | — | init dependence of loss AND of rank≈4 |
