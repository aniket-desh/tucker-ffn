# Theory notes — sprint 2

## 1. Confound A resolved algebraically: LL1 IS tied-gate CP

Verified (0 error, lib test + inline check): LL1$(B, L)$ equals a width-$BL$ SwiGLU
whose gate rows are tied in groups of $L$, and LL1's parameterization is exactly that
model with the $d \cdot B(L-1)$ redundant gate parameters removed. There is no "LL1
magic": **the scientific object is gate sharing**, i.e. the choice of how many
private routes to buy at a fixed budget. At budget $N$:

$$\text{SwiGLU: } m = \frac{N}{3d} \text{ atoms, } m \text{ routes} \qquad
\text{LL1}(L): \; \frac{3L}{2L+1}\frac{N}{3d} \text{ atoms, } \frac{3}{2L+1}\frac{N}{3d} \text{ routes}$$

The empirical question (Exp A) is which point in (routes, atoms) space matters:

- **atom-matched SwiGLU** ($m = 1992$, $+33\%$ params): if quality tracks atoms
  regardless of cost, the "structure" framing collapses to a capacity scaling law.
- **route-matched SwiGLU** ($m = 498$, $-67\%$ params): how much of LL1's quality do
  498 private-gate atoms alone achieve?
- LL1 sits at (498 routes, 1992 atoms) at the SwiGLU-1493 budget. If LL1 $\approx$
  atom-matched SwiGLU on loss, gate-tying is loss-free and the right summary is
  "routes are over-provisioned in SwiGLU; atoms are the binding resource."

## 2. Structured factor matrices: what each one tests

A structured projection at fixed budget buys **width**: at $d=512$, $N=2.29$M,
Monarch ($nb=4$) affords $m = 5844$ vs dense $1493$ ($3.9\times$). Each family is a
different prior on *which* wide map is reachable:

- **low-rank** ($W = BA$): global mixing, spectrum capped at rank $r$. Tests whether
  FFN projections need full rank (LoRA's success suggests updates don't; projections
  may differ).
- **block-diagonal**: no cross-block mixing at all — an ablation of mixing, not a
  serious candidate; it isolates how much of Monarch's value is mixing vs width.
- **Monarch** ($P^\top B_2 P B_1$ style, here two block-diagonal GEMMs + transpose):
  full mixing through a fixed permutation, $io/nb + o\,nb$ params. The hardware-real
  candidate (Dao et al. 2022).
- **butterfly** ($\log_2 n$ stages of 2×2 mixing + block-diagonal resize): FFT-like
  hierarchical global mixing at $O(n\log n)$; padding to powers of 2 where needed.

Per-token FLOPs equal parameter count for all of these, so parameter-matching is
FLOP-matching; only measured tokens/sec can distinguish hardware behavior
(batched-GEMM and elementwise-stage overheads are invisible to FLOP counts).

## 3. Trained sparsity: what L1 on routes means tensor-theoretically

Sparse CPD in the TensorLab sense = few nonzero rank-one terms. The trained analogue
for a *routed* CP model penalizes the route activations
$s_j(x) = \mathrm{SiLU}(g_j^\top x)$:

$$\mathcal{L} = \mathrm{CE} + \lambda\,\frac{1}{mT}\sum_{t,j} |s_j(x_t)|$$

driving per-token route sparsity (contributions $h_j = (w_j^\top x)s_j$ vanish with
$s_j$). `contrib_l1` penalizes $|h_j|$ directly (allows large gates with small
products); `group_lasso` penalizes $\|h_b\|_2$ per LL1 block (zeroes whole blocks).
The test (Exp C): does the loss-vs-active-units frontier move, and do
context-specific causal effects sharpen?

## 4. Why stable rank can mislead (confound B)

Stable rank $\|V\|_F^2/\|V\|_{op}^2$ equals the participation ratio of the squared
spectrum — it is dominated by the head and says nothing about the tail. A spectrum
$\sigma_i \propto i^{-1}$ truncated at 128 has stable rank $\approx 5$ but numerical
rank 128 at $\tau = 0.01$. Exp 25 therefore reports the full spectrum plus
numerical ranks at $\tau \in \{0.1, 0.01\}$, spectral-entropy effective rank, and
top-$k$ energy fractions, on trained checkpoints AND on freshly initialized models
(diag warm start vs random) to expose init-anchoring.

## 5. Superposition recovery: interpretability with ground truth (Exp D)

Data: $x = Fz$ with sparse $z$ ($K > d$: features in superposition);
$y = \sum_{(i,j)\in P} z_i z_j\, v_{ij}$ — the target IS a routed CP tensor with
known atoms $v_{ij} \otimes f_i \otimes f_j$. Topologies: `random` pairs (CP's
prior) and `hub` (each gate feature shared by $L=4$ mains — LL1's prior).

This makes "interpretability" objective: a student is interpretable to the extent
its learned atoms (after SVD canonicalization for blocks) align with the true
$(v, f_i, f_j)$ triples, and its matched atoms are causally specific (ablating the
matched atom hurts exactly the samples where $z_i z_j \neq 0$). Thomas's claim
"sparse CPD is more interpretable" predicts: SwiGLU+route-L1 recovers true atoms at
higher rates than plain SwiGLU at equal loss.

## 6. Skipped cells and why (pre-registered)

- **BTD$(L_1, L_2, L_3)$ with gate-mode rank > 1**: a vector-valued route breaks the
  scalar input-dependent-coefficient reading that makes the whole routed-tensor
  family analyzable; it is a different research question (mixture-of-projections).
- **TT/MPO**: the FFN interaction tensor has no natural mode ordering for a chain
  prior; factor-matrix structure (axis 2) was Thomas's explicit ask and strictly
  higher priority. Honest gap, listed in limitations.
- **softmax/entmax/trained top-$k$ routing**: unstable at sprint scale in pilot
  literature; the L1 family answers the sparsity question with less optimizer risk.
