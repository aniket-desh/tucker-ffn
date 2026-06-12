# Model configurations (sprint 2)

## LM runs ($d=512$, 8 layers; FFN params/layer; total model params)

| run | config | FFN params/layer | total | routes | atoms | note |
|---|---|---|---|---|---|---|
| SwiGLU base (s1) | $m=1493$ | 2.293M | 52.5M | 1493 | 1493 | sprint-1 baseline, $n=5$ |
| LL1 $L=4$ (s1) | $B=498$ | 2.295M | 52.5M | 498 | 1992 | sprint-1, $n=3$ |
| atom-matched SwiGLU | $m=1992$ | 3.060M | 58.6M | 1992 | 1992 | +33% FFN params |
| route-matched SwiGLU | $m=498$ | 0.765M | 40.3M | 498 | 498 | −67% FFN params |
| Tucker core-lr ×2 / ×0.5 | $r=s=128$, diagWS | 2.296M | 52.5M | 128 | — | 30M-token probes |
| Tucker no warm start | $r=s=128$, random | 2.296M | 52.5M | 128 | — | 30M-token probe |
| struct_monarch4 | monarch $nb=4$, $m=5844$ | ≤2.293M | 52.5M | 5844 | 5844 | width via structure |
| sparse SwiGLU | $m=1493$ + route-L1 $\lambda$ | 2.293M | 52.5M | 1493 | 1493 | Exp C |
| sparse LL1 | $B=498$, $L=4$ + group lasso | 2.295M | 52.5M | 498 | 1992 | Exp C |

## Distillation students (Qwen2.5-0.5B, $d=896$; budgets 0.6M / 1.2M)

| student | 0.6M config | 1.2M config |
|---|---|---|
| swiglu (dense) | $m=223$ | $m=446$ |
| swiglu_lowrank | $m=446$, $r=149$ | $m=892$, $r=...$ |
| swiglu_blockdiag4 | $m=892$ | $m=1784$ |
| swiglu_monarch4 | $m=876$ | $m=1764$ |
| swiglu_butterfly4 | $m$ = max pow-2 fitting | — |
| ll1_l4 (dense) | $B=148$ | $B=297$ |
| ll1_l4_monarch4 / blockdiag4 | $B$ fit to budget | $B$ fit to budget |

## Superposition task (exp24; $d=64$, budget 9216)

| student | config | atoms |
|---|---|---|
| swiglu / swiglu+L1 | $m=48$ | 48 |
| ll1_l2 | $B=29$ | 58 |
| ll1_l4 | $B=16$ | 64 |
| tucker | $r=s=17$ | — |

Task: $K \in \{96$ (superposition), $48$ (separable)$\}$ latents, 32 ground-truth
pairs, topologies {random, hub(L=4)}, $p_{\text{active}}=0.2$.
