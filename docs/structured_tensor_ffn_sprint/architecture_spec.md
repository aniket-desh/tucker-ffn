# Architecture spec

All counts are per FFN block, no biases, residual width $d$. "Atoms" = rank-one
routed interaction terms in the CP expansion; "routes" = independent gate scalars.

## SwiGLUFFN($d$, $m$) — lib/tucker_ffn.py
- $y = U^\top[(W^\top x) \odot \mathrm{SiLU}(G^\top x)]$; routed CP with $m$ atoms,
  $m$ routes.
- params = FLOPs/token (MACs) = $3dm$. LM config: $d=512$, $m=1493$ → 2.293M.
- Per-gate rank: 1 by definition. Expected interpretability: atomized, identifiable
  decomposition; diffuse usage in practice (prior draft: ~2500/4864 channels active).
- Experiments: exp11 (`--archs swiglu`), exp18 (`swiglu` student), exp20 (`swiglu`).

## TuckerFFN($d$, $r$, $s$) — lib/tucker_ffn.py
- $z_o = \sum_{ij} C_{oij}\, p_i\, \mathrm{SiLU}(q_j)$, $y = Rz$. $r$ routes;
  per-route interaction matrix $V_j = R\,C^{(j)}$, generically full rank $\min(r,s)$.
- params $= d(2r+s) + sr^2$. LM config: $r=s=128$ → 2.296M (91% of budget in core
  $C$).
- Trained with variance-preserving diagonal warm start ($C_{ooo}=1$, off-diag std
  $10^{-2}/r$, see exp11 flags) — the only init that matched SwiGLU in prior work.
- Gauge freedom: latent bases not identifiable. Expected interpretability: poor at
  the factor level; core diagnostics measured in exp19.
- Experiments: exp11 (`--archs tucker --tucker_diagonal_bias_init`), exp18, exp19,
  exp20.

## LL1FFN($d$, $B$, $L$) — lib/ll1_ffn.py  [new this sprint]
- $y = \sum_b U_b\,(A_b^\top x)\,\mathrm{SiLU}(g_b^\top x)$. $B$ routes, $BL$ atoms;
  per-route rank $\le L$ by construction. Routed sum of multilinear rank-$(L,L,1)$
  terms (LL1/BTD).
- params = MACs = $dB(2L+1)$. Three dense GEMMs ($A$,$U$ stacked $(d, BL)$; $G$
  $(d, B)$) + broadcast.
- Exactness tests (lib/ll1_ffn.py main): $L=1$, $B=m$ equals SwiGLUFFN to 0 error;
  LL1($B$,$L$) equals block-superdiagonal TuckerFFN($r=s=BL$) to $10^{-6}$;
  $\operatorname{rank}(V_b) \le L$ verified.
- LM configs at budget 2.293M ($d=512$): $L=1$: $B=1493$ | $L=2$: $B=896$ | $L=4$:
  $B=498$ | $L=8$: $B=263$ | $L=16$: $B=136$. Atom counts: 1493 / 1792 / 1992 /
  2104 / 2176 (LL1 trades route diversity for $\tfrac{3L}{2L+1}\times$ atoms).
- Experiments: exp11 (`--archs ll1_l{L}`), exp18, exp19, exp20.

## SwiGLUFFNAligned($d$, $m$, $P$, $Q$, assignment) — lib/tucker_ffn.py
- Theorem hypothesis class (fixed dictionaries). Used in exp10 only (prior work).

## Monarch/butterfly factors — NOT implemented
- Deprioritized per TASK §14: efficiency axis only attempted after LL1 results are
  in; LL1 already achieves SwiGLU-shaped GEMMs, and the sprint's question is core
  structure. Listed as future work.

## Experiment configs
- exp11 LM: $d=512$, 8 layers, 8 heads, seq 1024, GPT-2 BPE vocab 50257, tied
  embeddings, AdamW $\beta=(0.9, 0.95)$ wd $=0.1$ (Tucker core wd $=0$), peak lr
  $3\times 10^{-4}$ cosine, warmup 200 steps, batch $24 \times 1024$ tok, 100M
  FineWeb-Edu (sample-10BT) tokens, bf16 autocast, val = $128 \times 1024$ held-out
  tokens (deterministic seed 12345). Archs: swiglu, tucker, ll1_l{1,2,4,8,16}; seeds
  0–4 for swiglu, 0–2 for tucker/ll1_l2/ll1_l4/ll1_l8, seed 0 for ll1_l1/ll1_l16.
- exp18 synthetic: $d=64$, teachers {CP (32 atoms), LL1 ($16$, $L^*=4$), Tucker
  ($r=16$)}, unit-std outputs; students at budget 9216 ($L$ sweep) and
  $\{0.25, 0.5, 1, 2\}\times$ budgets; Adam $3\times 10^{-3}$ cosine, 5000 steps,
  batch 512, 50K train / 8K val gaussians, 3 seeds.
- exp19 proxies: $32 \times 1024$-token val set, 4096-token contribution
  sample/layer, ablation on layer 3 (48 units: top-16 + 32 random).
- exp20 induction: vocab 128, $[s; s]$ with $|s|=64$, $d=128$, 2 layers, 4 heads,
  FFN budget 131072 params (swiglu $m=341$ / ll1_l4 $B=114$ / tucker $r=48$ / none),
  AdamW $10^{-3}$, 3000 steps, batch 128, 3 seeds.
