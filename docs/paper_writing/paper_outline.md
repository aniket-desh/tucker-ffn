# Paper outline (bullet level; numbers to be finalized)

Narrative spine: *The right amount of tensor structure in a transformer FFN is a
measurable quantity — the per-route interaction rank — and on every real test it is
small but bigger than one.* (Pending LM results; fall back to "routes and ranks both
bind; structure must match function" if LM is a tie.)

## Abstract (sentence-per-claim)
1. FFNs hold most transformer parameters; their multiplicative interactions lack the
   mechanistic vocabulary attention has.
2. SwiGLU admits an exact routed-CP decomposition: rank-1 atoms × sigmoid routes.
3. The superdiagonal core is both the interpretability (atomized, identifiable) and the
   restriction (1 route : 1 rank-1 atom).
4. We study the LL1/block-CP family interpolating CP→Tucker via per-route rank L; at
   matched params LL1 trades routes for atoms ($3L/(2L+1)$).
5. Results: synthetic — structure must match (error min exactly at teacher rank, both
   resources bind; each family ~machine-precision on own class); real Qwen FFN maps —
   LL1 $L\approx 4$–$8$ best compressor, Tucker worst; LM at 52.5M/100M tokens — [pending];
   interp proxies — [pending]; induction — null on emergence.
6. Implication: per-route rank is the design dial; dense Tucker is dominated at fixed
   budget [if confirmed]; SwiGLU sits at L=1 which is [optimal/suboptimal per LM result].

## Intro paragraphs
P1 FFN obstacle. P2 exact routed-CP (state it as a fact in words; one equation).
P3 cost/benefit of superdiagonal. P4 dense Tucker is the naive move; gauge freedom +
budget concentration + Dooms critique; honest revision of prior draft's framing.
P5 LL1 family; theorem reinterpretation (constructive half = architecture); the
route/atom budget identity. P6 experiments: recovery (must-match), real-map
distillation (which structure is real?), matched-budget LM (does it train?), proxies
(what does it cost interpretability?), induction pilot (does it change circuits?).
P7 contributions:
 1. exact framing + nested family + budget identity (algebraic);
 2. matched-budget recovery: per-route rank is bidirectionally binding (synthetic);
 3. real-FFN compression ordering LL1 > CP > Tucker [verify];
 4. LM-scale matched comparison + throughput accounting [pending];
 5. measured interpretability proxies + induction null [pending];

## Figures
F1 ladder diagram (done, polish). F2 exp18 lsweep (done). F3 LM lsweep+curves.
F4 exp21 distillation. F5 interp proxies (topk + eff-active + stable rank).
F6 induction (appendix?). Budget sweep + offset profiles → appendix.

## Experiments section order
4.1 synthetic recovery → 4.2 real-layer distillation (bridge: which structure does a
real FFN's function have?) → 4.3 from-scratch LM (does trainability track structure?)
→ 4.4 interp proxies (what does L buy/cost mechanistically?) → 4.5 induction pilot
(circuit-level null + tucker anomaly).

## Discussion
- What is established: structure-must-match at fixed budget; real maps prefer small
  L>1 for compression; theorem's dial is the right abstraction.
- Plausible not established: LL1 as drop-in FFN improvement [depends on LM]; proxies →
  human-legible mechanisms.
- Failed/open: emergence effects (null); tucker induction anomaly (1/3 seeds).
- Limitations: scale, single dataset, hyperparams inherited from swiglu tuning, proxies.
- Concrete next step: scale LL1 L-sweep to ≥300M params with per-arch lr tuning;
  trained sparse-CP (L1-on-routes) variant; Monarch factors on A/U.
