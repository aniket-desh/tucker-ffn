# Theory notes: structured tensor-network FFNs

Notation: residual stream $x \in \mathbb{R}^d$. $\mathrm{SiLU}(z) = z\,\sigma(z)$,
$\sigma(z) = (1+e^{-z})^{-1}$. $\mathrm{SiLU}(0) = 0$ and $\mathrm{SiLU}(1) \ne 0$
are the only properties used in the separation arguments.

## 1. SwiGLU as routed CP (recap, exact)

SwiGLU: $y = U^\top h$, $h_j(x) = (w_j^\top x)\,\mathrm{SiLU}(g_j^\top x)$. Using the
SiLU identity,

$$h_j(x) = (w_j^\top x)(g_j^\top x)\,\sigma(g_j^\top x) \qquad \text{[exact, every input]}$$

Define the input-dependent interaction tensor $A(x) \in \mathbb{R}^{d\times d\times d}$
by $y_o = \sum_{a,b} A_{oab}(x)\, x_a x_b$:

$$A(x) = \sum_{j=1}^{m} \alpha_j(x)\; u_j \otimes w_j \otimes g_j, \qquad \alpha_j(x) = \sigma(g_j^\top x).$$

- Modes: output direction ($u_j$), main feature ($w_j$), gate feature ($g_j$).
- Routed: the scalar coefficient $\alpha_j(x)$ per atom. Fixed: the rank-one atoms
  $u_j \otimes w_j \otimes g_j$.
- This is a CP decomposition with shared factors and input-dependent weights — exact,
  not an approximation.
- Viewed as Tucker with $r=s=m$, $P=W$, $Q=G$, $R=U^\top$, the core must be
  superdiagonal: $C_{oij} = \delta_{oi}\delta_{ij}$. **Interpretable** because
  atomized: each hidden index $j$ is one (what, when) pair — one available
  interaction and one routing scalar. **Restrictive** because same-index: $w_i$ never
  interacts with $g_j$ for $i \ne j$, and one route controls exactly one rank-one
  interaction.

## 2. Dense Tucker FFN and the per-gate matrix

Tucker FFN: $p = P^\top x$, $q = Q^\top x$ ($P, Q \in \mathbb{R}^{d\times r}$),
$z_o = \sum_{ij} C_{oij}\, p_i\, \mathrm{SiLU}(q_j)$, $y = Rz$
($R \in \mathbb{R}^{d\times s}$, $C \in \mathbb{R}^{s\times r\times r}$). Grouping by
gate index $j$:

$$y(x) = \sum_{j=1}^{r} V_j\, p\; \mathrm{SiLU}(q_j), \qquad V_j := R\,C^{(j)} \in \mathbb{R}^{d\times r}, \quad C^{(j)}_{oi} = C_{oij}.$$

Each latent gate $q_j$ controls a *matrix-valued* bundle of interactions $V_j$. If
$\operatorname{rank}(V_j) = \rho_j$, gate $j$ simultaneously routes $\rho_j$
independent (output $\otimes$ main) interaction directions. SwiGLU is the corner case
$\rho_j = 1$ for all $j$ (with $r = m$ gates).

**Gauge problem (why dense Tucker resists interpretation).** For any invertible
$M \in \mathbb{R}^{r\times r}$ applied as $P \mapsto P M^{-\top}$ with
$C^{(j)} \mapsto C^{(j)} M$ for all $j$, the function is unchanged (the SiLU
nonlinearity partially breaks the $Q$-side gauge, but the $P$-side gauge is exact).
Latent directions $p_i$ are therefore not individually meaningful — only the per-gate
column spaces $\operatorname{span}(V_j)$ and the gate directions $Q_{:j}$ are.
CP/LL1 do not have this freedom (rank-one terms are identifiable up to
permutation+scaling under Kruskal-type conditions; LL1 terms under De Lathauwer's
conditions). This is the precise sense in which "CPD gives a sparse core for free"
and dense Tucker is "arbitrary."

## 3. Aligned-width theorem, reinterpreted

Theorem (from the prior draft, proof unchanged): with $[P\;Q]$ full column rank, an
aligned SwiGLU (units restricted to $w_l = P a_l$, $g_l \in \text{columns}(Q)$)
exactly realizing the Tucker map needs width
$m = \sum_j \operatorname{rank}(V_j)$; a rank decomposition of each $V_j$ attains it.

Old reading: "Tucker is $k\times$ more parameter-efficient than (aligned) SwiGLU."
New reading: **the control variable is the per-gate rank
$\rho_j = \operatorname{rank}(V_j)$**, not "Tucker vs not-Tucker." The theorem's
constructive direction *is an architecture*: realize each gate's $V_j$ directly as a
rank-$L$ factorization $U_b A_b^\top$. That architecture is LL1.

Empirical anchor (prior draft, App. B.2): a dense-Tucker LM trained from scratch
($r=s=128$, 100M tokens) has per-gate stable rank concentrated at
$\bar\rho \approx 3.97$ (range 2.8–5.3). The trained dense core *is approximately an
LL1 model with $L \approx 4$*. Pre-registered prediction for this sprint: LL1 with
$L \approx 4$ should match dense Tucker end-to-end, and $L$-sweeps should show
diminishing returns beyond $L \approx 4$.

## 4. LL1 / block-CP FFN (the proposed middle ground)

LL1 decomposition of a third-order tensor (De Lathauwer; TensorLab "decomposition in
multilinear rank-$(L,L,1)$ terms"):

$$T = \sum_{b=1}^{B} (A_b B_b^\top) \otimes c_b$$

— each term is rank-$L$ in two modes, rank-1 in the third. Mapping onto the FFN
interaction tensor with the *gate* mode as the rank-1 mode:

Architecture LL1GLUFFN($d$, $B$, $L$): per block $b$, gate direction
$g_b \in \mathbb{R}^d$, main factor $A_b \in \mathbb{R}^{d\times L}$, output factor
$U_b \in \mathbb{R}^{d\times L}$.

$$r_b(x) = A_b^\top x \in \mathbb{R}^L \quad \text{(block main response)}$$
$$s_b(x) = \mathrm{SiLU}(g_b^\top x) \in \mathbb{R} \quad \text{(block route, scalar)}$$
$$y(x) = \sum_b U_b\, r_b(x)\, s_b(x)$$

Expanded into atoms:
$y(x) = \sum_b \sum_{l=1}^{L} u_{b,l}\,(a_{b,l}^\top x)\,\mathrm{SiLU}(g_b^\top x)$ —
**grouped CP**: $L$ rank-one atoms share one route. Interaction tensor:

$$A(x) = \sum_b \alpha_b(x) \sum_l u_{b,l} \otimes a_{b,l} \otimes g_b = \sum_b \alpha_b(x)\,(U_b A_b^\top) \otimes g_b,$$

exactly a routed LL1/BTD tensor. As Tucker: block-sparse core — gate $j$ couples only
to latent block $j$; $V_j = U_j A_j^\top$ with rank $\le L$ *by construction*. The
aligned-width theorem is implemented architecturally: per-gate rank is the
hyperparameter $L$.

Nesting (all exact):
- $L=1$, $B=m$ → SwiGLU-form routed CP (identical function class to SwiGLU width $m$).
- $B=1$, $L=r$, $g$ fixed → single dense block (bilinear layer modulo the one gate).
- $L=r$, $B=r$ with shared latent dictionaries → dense Tucker (each gate gets a
  full-rank $V_j$).

**Equivalence that sharpens the comparison.** An LL1($B$, $L$) block computes the
same function class as a width-$m{=}BL$ SwiGLU whose gate vectors are *tied in groups
of $L$* ($g$ identical within a block). So at matched atom count, LL1 ⊂ SwiGLU
(strictly: SwiGLU can untie the gates). The interesting comparison is at matched
parameters:

$$\#\text{params}_{\text{SwiGLU}}(d, m) = 3dm \quad (m \text{ atoms, } m \text{ gates})$$
$$\#\text{params}_{\text{LL1}}(d, B, L) = B\,d\,(2L+1) \quad (BL \text{ atoms, } B \text{ gates})$$

At equal budget $N$: SwiGLU affords $m = N/(3d)$ atoms; LL1 affords
$BL = N L / (d(2L+1))$ atoms, i.e. a factor $\tfrac{3L}{2L+1}$ more atoms
($1.2\times$ at $L=2$, $1.33\times$ at $L=4$, $\to 1.5\times$). **LL1 trades gate
diversity for interaction atoms.** Whether that trade wins is an empirical question —
this is the cleanest statement of what the LM experiment tests.

- If gates are the scarce resource (routing diversity matters most): SwiGLU wins.
- If bilinear capacity is scarce (atoms matter, routes are redundant): LL1 wins.
- Dense Tucker spends its budget on an all-to-all core ($sr^2$ params), maximally
  many interactions but only $r$ routes and gauge-entangled latents.

## 5. Parameter / FLOP accounting (per token, multiply-adds, no bias)

| arch | params | fwd MACs | structure |
|---|---|---|---|
| SwiGLU($d$,$m$) | $3dm$ | $3dm$ | 3 dense GEMMs |
| Dense FFN (GELU, width $m'$) | $2dm'$ | $2dm'$ | 2 dense GEMMs |
| Tucker($d$,$r$,$s$) | $d(2r+s)+sr^2$ | $d(2r+s)+sr^2$ | 3 GEMMs + core contraction $z = C\cdot(p \otimes \mathrm{SiLU}(q))$ |
| LL1($d$,$B$,$L$) | $dB(2L+1)$ | $dB(2L+1)$ | 3 dense GEMMs ($A$,$U$ stacked as $d\times BL$; $G$ as $d\times B$) + cheap broadcast multiply |
| Aligned SwiGLU($m$; $P$,$Q$ $r$-dim) | $m(r+d)$ | $2dr + m(r+d)$ | — |

LL1 is GEMM-friendly: stack $A = [A_1 \dots A_B] \in \mathbb{R}^{d\times BL}$, $U$
likewise, $G \in \mathbb{R}^{d\times B}$; forward is
`(x@A).reshape(B,L) * silu(x@G)[:,None]` then `@U.T` — three dense GEMMs, same shape
regime as SwiGLU. No batched-small-GEMM penalty like the Tucker core contraction (the
throughput gap observed for Tucker: 75K vs 112K tok/s on A100 in prior work). Expect
LL1 throughput ≈ SwiGLU throughput.

Matched-budget configs used in experiments ($d=512$, LM): target $N = 3dm$ with
$m=1493$ (≈2.293M/layer). LL1: $B = \mathrm{round}(N/(d(2L+1)))$:
- $L=1$: $B=1493$ (degenerate SwiGLU-tied, sanity)
- $L=2$: $B=896$, atoms $=1792$
- $L=4$: $B=498$, atoms $=1992$
- $L=8$: $B=263$, atoms $=2104$
- $L=16$: $B=136$, atoms $=2176$
- $L=64$: $B=35$, atoms $=2240$ (Tucker-like, few routes)

Tucker: $r=s=128$ (2.296M). Dense-core fraction: $sr^2 = 2.097$M of 2.296M (91% of
budget in core).

## 6. Sparse CP / sparse Tucker variants

- Sparse routed CP: SwiGLU + L1 penalty on per-token contributions $|c_j(x)|$ or on
  $\alpha_j$; post-hoc top-$k$ gating as an inference diagnostic. Tests "Level 2" of
  the ladder.
- Sparse-core Tucker: L1 on $C$ entries. Tests whether dense Tucker, when pushed
  toward sparsity, rediscovers block/diagonal structure.
- Both are secondary to the LL1 axis; implement post-hoc top-$k$ + core-sparsity
  metrics first (cheap), trained-penalty variants only if time.

## 7. Interpretability proxies (defined before measuring)

For trained models, per token $x$ at a given layer; $c_j(x)$ = atom-level
contribution (SwiGLU: $\alpha_j(x)\,(w_j^\top x)(g_j^\top x)\,\|u_j\|$; LL1
block-level: $\|U_b\, r_b(x)\| \cdot |s_b(x)|$; Tucker gate-level:
$\|V_j\, p\| \cdot |\mathrm{SiLU}(q_j)|$):

1. **Effective active count**: $\exp(H(p))$ with $p_j = |c_j| / \sum_k |c_k|$. Range
   $[1, \#\text{units}]$. Compare *as a fraction of available units* and in absolute
   terms (both reported; architectures differ in unit count at matched params).
2. **90% mass fraction**: fraction of units covering 90% of $\sum_j |c_j|$.
3. **Ablation locality**: $\Delta$loss from zeroing one unit (atom/block/gate-slice),
   distribution over units; interpretable structure ⇒ heavy-tailed (few units matter
   a lot per context, most matter little).
4. **Per-gate stable rank** (Tucker/LL1): $\|V_j\|_F^2 / \|V_j\|_{\mathrm{op}}^2$.
   For LL1 $\le L$ by construction; for Tucker measures what the training actually
   used.
5. **Core density / entropy** (Tucker): fraction of $|C_{oij}|$ above threshold,
   entropy of normalized $|C|$. Block structure detection: best block-permutation
   energy concentration.
6. **Factor stability across seeds**: match atoms/blocks across seeds by greedy
   cosine assignment on $(u, w, g)$ concatenations; report mean matched cosine.
   Identifiable decompositions should be more stable. (Caveat: stability ≠ semantic
   meaning.)
7. **Gate sparsity**: distribution of $\alpha_b(x)$ across tokens (bimodality,
   fraction near 0).

These are proxies. Lower entropy or higher locality does not by itself establish
human-legible mechanisms; treat as necessary-not-sufficient.

## 8. Hypotheses (pre-registered)

H1 (synthetic recovery): LL1 students fit LL1($L^*$) teachers with an error knee at
$L_{\text{student}} = L^*$ under matched atom budget; CP students need
$\sim L^*\times$ more gates; dense Tucker fits everything but with diffuse core.
Falsifier: no knee / optimization noise dominates.

H2 (LM, matched params): LL1 at $L \in \{2,\dots,8\}$ ≥ SwiGLU ≈ Tucker in val loss;
per the $\bar\rho \approx 4$ anchor, $L=4$ ≈ Tucker. Falsifier: LL1 below SwiGLU at
all $L$.

H3 (interpretability): LL1 proxies between SwiGLU and Tucker, closer to SwiGLU;
Tucker core diffuse (no recoverable block structure). Falsifier: Tucker core
spontaneously block-sparse (would itself be a finding).

H4 (throughput): LL1 ≈ SwiGLU > Tucker tokens/sec at matched params.

## 9. References for the writeup

- Kolda & Bader 2009 (CP/Tucker; CP = superdiagonal Tucker).
- De Lathauwer 2008/2011 (BTD, LL1 = decomposition in multilinear rank-$(L,L,1)$
  terms, uniqueness conditions); TensorLab docs (cpd.html, ll1.html, btd.html).
- Pearce, Dooms, Rigg, Oramas, Sharkey 2025 (arXiv:2410.08417): bilinear MLPs (GLU
  sans nonlinearity) are exactly third-order tensors; eigendecomposition reveals
  interpretable low-rank structure; the $\alpha \equiv 1$ limit of our routed-CP
  picture.
- Dooms & Wilhelm 2024 (arXiv:2406.03947): weight-based decomposition case for
  bilinear MLPs.
- Dao et al. 2022 (arXiv:2204.00595): Monarch (products of block-diagonals +
  permutations); efficiency axis, orthogonal to core structure.
- Novikov et al. 2015: tensor-train compression of dense layers (compression, not
  interp).
- Shazeer 2020 (GLU variants); Olsson et al. 2022 (induction heads); Elhage et al.
  2021 (circuits framework); Hu et al. 2021 (LoRA, rank constraints work).
- Jayakumar et al. 2020 (multiplicative interactions).
