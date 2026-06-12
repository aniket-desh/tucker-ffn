# TASK 2: Broad Tensor-Network FFN Architecture Sweep + Interpretability Diagnosis

Repository:

`https://github.com/aniket-desh/tucker-ffn`

This is a **12–24 hour wall-clock sprint** on a RunPod machine with **2× A40 GPUs** and roughly **500GB persistent disk**. Track elapsed wall-clock time explicitly. Maintain an hourly `research_log.md`. Use compute seriously but not blindly: smoke-test first, reproduce current headline numbers, then run only high-information experiments.

The previous sprint was valuable, but it became mostly an **LL1/block-CP sprint**, not the broader **principled tensor-network architecture search** Thomas Dooms asked for. This second sprint should broaden the architecture axis, diagnose why the interpretability hypothesis failed, and produce a clearer next-step research direction.

The guiding question is:

> Which tensor-network FFN structures actually buy something over SwiGLU, and on which axis: expressivity, parameter efficiency, compute efficiency, or interpretability?

The prior answer is not enough. LL1 appears structurally honest and efficient, but not interpretable under the metrics used. This sprint should test whether that conclusion is real or an artifact of the architecture set and measurement choices.

---

## 0. Required starting point

Before implementing anything, read:

- `docs/structured_tensor_ffn_sprint/summary.md`
- `docs/structured_tensor_ffn_sprint/theory_notes.md`
- `docs/structured_tensor_ffn_sprint/architecture_spec.md`
- `docs/structured_tensor_ffn_sprint/research_log.md`
- `docs/structured_tensor_ffn_sprint/scripts_used.md`
- `paper/main.tex` if present
- `lib/ll1_ffn.py`
- `lib/tucker_ffn.py`
- `lib/lm.py`
- all experiment scripts used in exp18–exp22

Then create:

`docs/broad_tn_sprint/`

with:

- `plan.md`
- `theory_notes.md`
- `architecture_lattice.md`
- `research_log.md`
- `summary.md`
- `scripts_used.md`
- `figures/`
- `tables/`

The first entry in `research_log.md` must state:

1. What the previous sprint established.
2. What it did **not** establish.
3. Which confounds are most serious.
4. Which architecture families you will test first and why.
5. What result would make you pivot.

---

## 1. Summary of previous sprint results

The previous sprint implemented and tested LL1/block-CP FFNs.

### Main positive findings

1. **Per-route rank is a real capacity dial.**
   In synthetic teacher-student recovery, CP/SwiGLU, LL1, and Tucker each recover their own teacher well, and LL1 succeeds only when student block rank and route count are both sufficient. This supports the theoretical idea that per-gate rank matters.

2. **Pretrained FFN maps prefer small block rank.**
   Distillation of Qwen2.5-0.5B FFN layers showed LL1 with block rank roughly `L=4–8` beating rank-one CP/SwiGLU students by modest but robust margins in relative MSE. Dense Tucker was much worse in the compression regime.

3. **At small LM scale, LL1 and SwiGLU tie on loss, while LL1 is faster.**
   In 52.5M-parameter / 100M-token language model training, LL1 `L=2,4,8` was statistically tied with SwiGLU and slightly nominally better, while Tucker lagged and was much slower. LL1 throughput was 5–7% faster than SwiGLU and ~2× faster than Tucker.

4. **Three measurements suggested effective per-route rank around 4.**
   Dense Tucker stable rank, LL1 saturation, and real-layer distillation all pointed to a natural per-route rank of order 4 in these models.

### Main negative findings

1. **The interpretability hypothesis failed.**
   LL1 did not route sparsely. It did not produce stable blocks across seeds. Single-unit ablations were negligible. Its blocks were not more legible than SwiGLU atoms by the measured proxies.

2. **The induction-head pilot was mostly null.**
   All FFN architectures learned the repeated-sequence task at similar speed; attention-only was faster. FFN tensor structure did not clearly affect induction-head emergence.

3. **The architecture search was too narrow.**
   Monarch, butterfly, low-rank matrix factors, block-diagonal factors, sparse CPD training, structured Tucker penalties, tensor-train/MPO variants, and general block-term variants were not seriously tested.

The previous sprint’s cleanest conclusion is:

> LL1 is a real middle ground on the structure/efficiency axis, but not yet on the mechanistic-interpretability axis. Dense Tucker looks dominated at the tested scale. SwiGLU/CP remains surprisingly strong.

This sprint must test whether this conclusion survives a broader search.

---

## 2. Thomas Dooms’s critique, sharpened

Thomas’s point was not merely “try LL1.” His broader request was:

> Avoid arbitrary tensors. Search over principled tensor-network architectures that improve either parameter efficiency, interpretability, or both.

He specifically contrasted:

- **Monarch / butterfly matrices**: efficiency-first structured matrix families.
- **Sparse CPDs**: interpretability-first sparse atom decompositions.
- **LL1 / structured Tucker**: a possible middle ground.

The previous sprint mostly tested only the LL1 middle ground. This sprint should explicitly explore the missing axes.

### Efficiency axis

Structured linear maps can replace dense projections in FFNs:

- block-diagonal matrices
- low-rank factors
- Monarch matrices
- butterfly matrices
- Kronecker or tensor-product factors
- tensor-train / MPO factorizations of the projection matrices

These should be judged by:

- parameter count
- FLOPs/token
- measured tokens/sec
- memory behavior
- validation loss
- distillation error

### Interpretability axis

Sparse CP-like structures should be judged by:

- route sparsity
- contribution sparsity
- context-specific causal effect
- factor stability
- semantic coherence of top-activating contexts
- auto-interp or human-inspection quality
- pruning/decomposition robustness
- whether atoms/blocks correspond to identifiable mechanisms

Do not conflate algebraic sparsity with human interpretability.

---

## 3. Dmitry Manning-Coe’s suggestion, sharpened

Dmitry suggested:

> Once you have the sweep, pick a model structure like induction heads and sweep on that.

The previous induction pilot was a reasonable first attempt, but the task was attention-dominated. It did not isolate FFN structure.

This sprint should either:

1. pick a **more FFN-loaded mechanistic target**, or
2. redesign the induction/copying task so the FFN has a real computational role.

Better mechanistic targets:

- key-value memory where FFNs store or transform associations
- algorithmic bilinear tasks requiring feature-feature multiplication
- modular arithmetic or modular multiplication
- two-feature conjunction tasks
- synthetic superposition tasks where sparse atoms should be recoverable
- factual-recall-style subject→attribute maps in a tiny model
- IOI/name-mover-like tasks only if the FFN role is clear

The aim is not just accuracy. The aim is to ask:

> Does the FFN tensor structure change the internal mechanism, not merely the loss?

---

## 4. Theoretical work required before coding

Before implementing each architecture, write its math in `theory_notes.md`.

For every model, specify:

1. The represented interaction tensor.
2. Which modes are factorized.
3. Whether it is CP, Tucker, LL1, BTD, TT/MPO, Monarch, butterfly, low-rank, or hybrid.
4. Its parameter count.
5. Its FLOP count.
6. Its expected hardware behavior.
7. Its gauge freedoms / identifiability issues.
8. Its expected interpretability object: atom, block, slice, subspace, or matrix factor.
9. What confound it tests relative to LL1 and SwiGLU.

Do not implement a model whose mathematical role is unclear.

---

## 5. Major theoretical confounds to resolve

### Confound A: LL1 is tied-gate SwiGLU

The previous notes correctly observed:

> LL1(B, L) is equivalent to a width-BL SwiGLU whose gate vectors are tied in groups of L.

At matched parameter count, LL1 has fewer routes but more atoms. Therefore, any LL1 effect may come from the **route/atom trade**, not from a fundamentally new tensor architecture.

Required experiment:

Run a factorial control separating:

- number of routes `B`
- number of atoms `M = B L`
- whether gates are private or tied
- whether output/main factors are grouped into explicit rank-L blocks

Models:

1. standard SwiGLU: private gate per atom
2. tied-gate SwiGLU: groups of L atoms share a gate, but implemented directly as grouped CP
3. LL1: tied gate plus block matrix view
4. route-reduced SwiGLU: same number of gates as LL1 but same total parameters allocated differently
5. atom-matched SwiGLU: same number of atoms as LL1 but more parameters, to separate capacity from budget

If LL1 and tied-gate SwiGLU are identical numerically, call it that. The scientific object is then **gate sharing / route tying**, not “LL1 magic.”

### Confound B: stable rank ≈ 4 may be metric- or init-induced

The previous sprint found stable rank around 4, but this could reflect:

- diagonal warm-start
- optimizer implicit bias
- residual dimension and model size
- regularization
- stable rank being insensitive to spectrum tails
- the fixed `r=s=128` Tucker configuration

Required analysis:

For Tucker and LL1, plot full singular spectra of per-route matrices, not only stable rank. Measure:

- stable rank
- numerical rank at thresholds
- entropy of singular values
- top-k energy fraction
- layerwise variation
- dependence on initialization
- dependence on model size or training duration

If rank ≈ 4 only appears under diagonal warm-start or one setup, weaken the claim.

### Confound C: dense Tucker may have been unfairly weak

Dense Tucker lost badly in several comparisons, but possible reasons include:

- core consumes too much parameter budget
- optimization is harder
- core contraction is hardware-unfriendly
- diagonal warm-start biases toward CP/LL1 structure
- compression budgets force r too low
- no per-architecture hyperparameter tuning

Required controls:

- tune Tucker lr/core-lr separately
- try low-rank Tucker core or CP-initialized Tucker
- try sparse-core Tucker
- try block-sparse Tucker initialized from LL1
- compare at matched routes as well as matched parameters
- report whether Tucker loses because of representational form, optimization, or budget accounting

### Confound D: interpretability proxies were too weak

Previous metrics were mostly global statistics:

- effective active count
- mass90
- single-unit ablation
- cross-seed factor matching

These can fail even if meaningful circuits exist, because:

- mechanisms can be context-specific and rare
- average ablations wash out sparse effects
- signed contributions can cancel
- learned features can be in superposition
- factor matching across seeds may require rotations, subspace alignment, or semantic matching
- blocks may be identifiable only up to within-block rotations

Required improvements:

1. **Context-specific ablation.**
   For each atom/block, ablate it only on its top-activating contexts. Measure local loss/logit changes.

2. **Signed causal contribution.**
   Measure contribution to the correct token logit or task-relevant logit, not only output norm.

3. **Top-context inspection.**
   Save top activating token/context windows for atoms/blocks. Use simple clustering or auto-labeling.

4. **Subspace matching.**
   For LL1 blocks, match subspaces using principal angles / CKA / Procrustes rather than greedy atom cosine.

5. **Within-block canonicalization.**
   Canonicalize each LL1 block by SVD of `V_b = U_b A_b^T` before matching across seeds.

6. **Semantic stability.**
   Compare top-context overlap or label overlap across seeds, not only parameter cosine.

7. **Rare-feature search.**
   Look at high-activation tails, not average activation.

Interpretability claims must be phrased as “proxy evidence,” not proof.

---

## 6. Architecture lattice to test

This sprint should test a **lattice**, not a single line.

### Axis 1: Core/tensor structure

1. **CP / SwiGLU**
   One route per rank-one atom.

2. **Sparse CP**
   CP/SwiGLU with trained route sparsity or contribution sparsity.

3. **Tied-gate CP**
   Groups of atoms share one gate. This is the LL1-equivalent control.

4. **LL1 / block-CP**
   One route controls a rank-L block.

5. **Sparse or block Tucker**
   Tucker core with L1, block sparsity, or block-diagonal structure.

6. **Dense Tucker**
   Expressive upper envelope and diagnostic baseline.

7. **General BTD**
   Sum of block terms with ranks `(L1, L2, L3)`, not only `(L, L, 1)`.
   This asks whether the gate mode should also have rank > 1.

8. **Tensor-train / MPO interaction tensor**
   Factorize the third-order interaction tensor in TT/MPO form.
   This is a different tensor-network prior, not just CP-vs-Tucker.

### Axis 2: Factor-matrix structure

For W/G/U or A/G/U factors, test:

1. dense
2. low-rank
3. block-diagonal
4. Monarch-style product of block-diagonal matrices with permutation
5. butterfly-style sparse hierarchical mixing
6. Kronecker-factorized linear maps
7. tensor-train / MPO linear maps

### Axis 3: Routing structure

1. sigmoid / SiLU gate
2. top-k route selection
3. softmax route selection
4. sparsemax/entmax route selection if easy
5. L1 on routes
6. group lasso on blocks
7. temperature-controlled routes

Do not test all combinations. Choose a small, principled subset.

---

## 7. High-priority experiments

Pick two or three experiments and do them deeply.

### Experiment A: Route/atom confound sweep

Goal:

Determine whether LL1’s effect is really block-rank structure or just tied-gate / route-count tradeoff.

Models:

- SwiGLU
- LL1 L=2,4,8
- tied-gate CP L=2,4,8
- route-matched SwiGLU
- atom-matched SwiGLU
- dense Tucker
- optional sparse CP

Tasks:

1. synthetic LL1 teacher
2. Qwen FFN distillation
3. small LM training for only the most important configs

Metrics:

- MSE or val loss
- parameter count
- route count
- atom count
- active route fraction
- active atom fraction
- throughput

Key plot:

- performance as a function of routes and atoms separately

Decision rule:

If tied-gate CP matches LL1 exactly, rename the finding around **gate sharing**. If LL1 only wins when it has more atoms, do not call it a superior tensor network.

---

### Experiment B: Efficiency-axis sweep — Monarch / butterfly / block / low-rank factors

Goal:

Actually test the architecture families Thomas named.

Implement simple, correct modules:

1. `LowRankLinear`
2. `BlockDiagonalLinear`
3. `MonarchLinear`
4. `ButterflyLinear` or simple butterfly-mixing linear layer

Use them as drop-in replacements in:

- SwiGLU W/G/U
- LL1 A/G/U
- optionally Tucker P/Q/R

Start small. Do not build custom CUDA. Use PyTorch modules and measure real throughput.

Models:

- Dense SwiGLU baseline
- LowRank-SwiGLU
- BlockDiag-SwiGLU
- Monarch-SwiGLU
- Butterfly-SwiGLU
- Dense LL1 L=4
- Monarch-LL1 L=4
- BlockDiag-LL1 L=4

Tasks:

1. Qwen FFN distillation
2. small LM smoke training
3. throughput benchmark

Metrics:

- val MSE / val loss
- parameter count
- symbolic FLOPs
- measured tokens/sec
- GPU memory
- approximation quality when fitting pretrained FFN maps

Key plot:

- quality vs measured throughput
- quality vs parameter count

Decision rule:

If Monarch/butterfly gives quality within 1–2% of dense at significantly fewer params or faster throughput, it is a real efficiency direction. If it lowers params but hurts throughput or quality, report that.

---

### Experiment C: Trained sparsity on CP and LL1

Goal:

Test Thomas’s “sparse CPD is more interpretable” claim directly.

Models:

- SwiGLU baseline
- SwiGLU + L1 on route activations
- SwiGLU + L1 on contribution magnitudes
- SwiGLU + top-k route training or inference
- LL1 L=4 baseline
- LL1 + L1 on block routes
- LL1 + group lasso on block output contributions
- sparse Tucker core if time

Tasks:

1. small LM training
2. Qwen FFN distillation
3. context-specific interpretability analysis

Metrics:

- val loss / MSE
- active atom/block count
- top-k decomposability
- context-specific ablation effects
- top-context coherence
- pruning curve

Key plot:

- loss increase vs effective active units
- local ablation effect distribution
- pruning curve: retained atoms/blocks vs performance

Decision rule:

If trained sparsity gives the same loss with far fewer active atoms/blocks, CPD-first interpretability becomes stronger. If loss degrades immediately, then sparsity is not free.

---

### Experiment D: Better mechanistic target than induction

Goal:

Find a circuit/task where FFN tensor structure matters.

Candidate tasks:

1. **Bilinear feature conjunction**
   Input contains two latent features; target depends on their conjunction. CP should need one atom per conjunction; LL1 should exploit block sharing; Tucker may use dense core.

2. **Key-value memory**
   The model sees `(key, value)` pairs and later a key query. Design it so attention retrieves key context, but FFN must transform or decode value.

3. **Modular multiplication or compositional arithmetic**
   Target depends on product-like interactions between two token features.

4. **Synthetic superposition**
   Data generated by sparse latent features with known ground-truth atoms. Test whether learned atoms recover true features.

Run tiny transformers or even isolated FFN probes first.

Metrics:

- task accuracy
- sample efficiency
- emergence speed
- active atom/block count on task examples
- recovery of known ground-truth factors
- causal ablation of recovered atoms/blocks

Decision rule:

If FFN tensor structure does not matter on a task designed to require multiplicative interactions, rethink the architectural hypothesis.

---

### Experiment E: Real-pretrained-layer interpretability

Goal:

Do not infer interpretability from small LMs only. Analyze real pretrained SwiGLU atoms.

Use Qwen2.5-0.5B or similar.

For selected layers:

1. collect residual inputs and FFN outputs
2. compute SwiGLU atom contributions
3. find top activating contexts for atoms
4. train LL1/SparseCP students on distillation
5. compare top-context coherence between teacher atoms and student atoms/blocks
6. perform layer replacement and local ablation on high-activation contexts

Metrics:

- top-context overlap
- semantic coherence score using simple clustering or auto-labeling
- local causal effect on logits
- route sparsity
- replacement perplexity

This is closer to Thomas’s interpretability concerns than small-model seed matching.

---

## 8. Implementation details

### 8.1 Add structured linear modules

Create:

`lib/structured_linear.py`

with:

- `LowRankLinear`
- `BlockDiagonalLinear`
- `MonarchLinear`
- `ButterflyLinear` or `SimpleButterflyLinear`
- parameter count utilities
- FLOP estimate utilities
- dense materialization for small dimensions for testing

Unit tests:

- shape tests
- parameter count tests
- dense materialization equality for small dims when applicable
- gradient flow tests
- throughput smoke benchmark

### 8.2 Add tied-gate CP module

Create:

`lib/tied_gate_ffn.py`

or integrate into `ll1_ffn.py`.

It should expose the equivalence between LL1 and grouped CP explicitly:

- group size L
- B routes
- BL atoms
- tied gates within each group

Make tests showing:

- tied-gate CP equals LL1 when parameterized equivalently
- `L=1` equals SwiGLU
- grouped atoms can be canonicalized by SVD of each block matrix

### 8.3 Add sparse regularizers

Create:

`lib/sparsity.py`

with:

- route L1 penalty
- contribution L1 penalty
- group lasso over blocks
- top-k route masking for inference
- optional straight-through top-k for training, only if stable

Make every regularizer log its actual realized sparsity.

### 8.4 Add improved interpretability tools

Create:

`lib/interp_metrics.py`

with:

- effective active count
- mass90
- context-specific top activation mining
- local ablation on top contexts
- signed logit contribution
- subspace matching via principal angles
- Procrustes / CKA matching
- SVD canonicalization for LL1 blocks
- pruning curves

Avoid only global averages.

---

## 9. Compute plan

Use the two A40s in parallel.

### GPU 0

Run architecture/efficiency sweeps:

- structured linear distillation
- throughput benchmarks
- route/atom confound runs

### GPU 1

Run sparse CP/LL1 training or mechanistic toy tasks.

Every experiment must start with:

- 1 tiny smoke test
- 1 small run
- then full run only if metrics are finite and logging works

Avoid launching a huge grid before inspecting one batch of results.

---

## 10. Deliverables

Final report:

`docs/broad_tn_sprint/summary.md`

Must start with an executive summary under 600 words.

Required sections:

1. Executive summary.
2. What the previous sprint found.
3. What this sprint corrected.
4. Architecture lattice tested.
5. Theory: route/atom trade and structured factor axes.
6. Experiment A: route/atom confound.
7. Experiment B: Monarch/butterfly/structured linear sweep.
8. Experiment C: trained sparsity and interpretability.
9. Experiment D/E: mechanistic or pretrained-layer interpretability study.
10. What failed.
11. What changed our mind.
12. Limitations.
13. Recommended next direction.
14. Research map of code/scripts/figures/tables.

Also create:

- `docs/broad_tn_sprint/architecture_lattice.md`
- `docs/broad_tn_sprint/theory_notes.md`
- `docs/broad_tn_sprint/tables/model_counts.md`
- `docs/broad_tn_sprint/tables/results_summary.md`
- `docs/broad_tn_sprint/figures/*.png`

---

## 11. Red-team checklist

Before finalizing, answer explicitly:

1. Did we actually test Monarch or butterfly, or did we again only test LL1?
2. Did we separate route count from atom count?
3. Did tied-gate CP explain the LL1 results?
4. Did we compare matched parameters, matched FLOPs, and measured wall-clock?
5. Did any structured matrix improve real throughput, not just symbolic FLOPs?
6. Did sparse CP preserve performance at lower active count?
7. Did interpretability improve semantically, causally, or only statistically?
8. Did local ablations reveal anything that global ablations missed?
9. Did factor matching use the right invariances?
10. Did Tucker lose because of architecture or because of optimization?
11. Are all claims larger than seed noise?
12. Did we tune baselines fairly?
13. Did mechanistic tasks actually require FFN computation?
14. What is the simplest explanation of every positive result?
15. What result would falsify the current favorite story?

If a result is a tie, call it a tie.
If a result is small, call it small.
If a result is inconclusive, say exactly why.

---

## 12. Success modes

### Success mode 1: LL1 reduced to tied-gate CP

You show that the real object is not “LL1” but “groups of atoms sharing routes.” This would simplify the theory and clarify the paper.

### Success mode 2: sparse CPD works

You show that trained sparse CP keeps performance while producing fewer active atoms and stronger context-specific causal effects. This would validate Thomas’s CPD-first instinct.

### Success mode 3: Monarch/butterfly gives a real efficiency frontier

You show that structured factor matrices improve parameter/throughput tradeoffs without large loss degradation. This would open the efficiency axis Thomas explicitly wanted.

### Success mode 4: no structured factor helps

You show Monarch/butterfly/block/low-rank projections hurt quality or throughput. This is a clean negative and narrows the project back toward core/routing structure.

### Success mode 5: better interp metric changes the conclusion

You show global metrics missed context-specific interpretable atoms/blocks, or confirm that the structures are genuinely diffuse. Either result is valuable.

### Success mode 6: FFN-loaded task reveals architecture differences

You find a synthetic or controlled task where CP/LL1/Tucker produce measurably different mechanisms. This validates Dmitry’s suggestion.

---

## 13. Tone

Be ambitious but skeptical. The previous sprint produced a coherent LL1 story, but it did not complete the broader tensor-network search. This sprint should be corrective.

The central update to test is:

> LL1 may be useful because it hard-codes the small per-route rank real FFNs appear to use, but interpretability probably requires explicit sparsity, context-specific causal metrics, and/or semantically grounded tasks. Efficiency likely requires structured factor matrices, not only structured cores.

Do the mathematics first. Then test the claim most likely to be false.

