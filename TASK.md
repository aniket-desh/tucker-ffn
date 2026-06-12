# Claude Code Autoresearch Prompt: Tensor-Network FFNs Between Interpretability and Expressivity

You are running an autonomous research sprint on the repository:

https://github.com/aniket-desh/tucker-ffn

This is a 12–16 hour wall-clock sprint on a RunPod machine with 2× A40 GPUs and roughly 500GB persistent volume disk. Track elapsed wall-clock time explicitly. Create a research log and update it at least once per hour. Use the GPUs seriously but not blindly: smoke-test first, reproduce existing results where possible, then parallelize only high-information experiments.

Your goal is not to generate lots of experiments. Your goal is to produce a small, rigorous, theory-driven research investigation into tensor-network feedforward networks for transformers.

The central question is:

What tensor-network structure gives the best tradeoff between interpretability, expressivity, parameter efficiency, and trainability in transformer FFNs?

The working hypothesis is:

SwiGLU is a routed CP tensor model: highly interpretable and atomized, but restricted to rank-one same-index interactions. Dense Tucker relaxes this bottleneck but may be too unconstrained, less identifiable, and less interpretable. The most promising middle ground may be structured tensor decompositions such as LL1 / block-CP / block-sparse Tucker, possibly combined with structured factor matrices such as Monarch or butterfly matrices.

You should try to either strengthen this story into a real contribution or kill it.

Do not manufacture positive results. If dense Tucker fails, say so. If LL1 does not help, say so. If all tensor variants lose to SwiGLU, that is a valuable negative result if the evidence is clean.

At the end, write a polished report:

docs/structured_tensor_ffn_sprint/summary.md

Spend at least the final 90 minutes on writing, plotting, and red-team review. The summary is the main deliverable. The code matters, but I will only inspect code if the write-up is compelling, clear, and correct.

---

## 0. First principles and research attitude

This is a physics/applied-math style project, not a blind ML sweep.

Before implementing any model, derive the mathematical object it represents.

Before running any experiment, write down:

1. What hypothesis the experiment tests.
2. What mathematical quantity is being varied.
3. What result would support the hypothesis.
4. What result would falsify or weaken the hypothesis.
5. What simple baseline could explain the same result.
6. What plot would make the result obvious.

Do not introduce tensor decompositions arbitrarily. A tensor decomposition is only worth testing if it buys at least one of:

1. Interpretability: atomized components, sparse cores, stable factors, local ablations.
2. Expressivity: more interaction patterns at matched parameter/FLOP budget.
3. Parameter efficiency: same performance with fewer parameters.
4. Compute efficiency: same performance with fewer FLOPs or better wall-clock throughput.
5. Mechanistic clarity: cleaner circuits or more structured feature interactions.

The project should be theory-driven. The workflow is:

derive -> implement -> smoke test -> run -> analyze -> red-team -> write.

---

## 1. Repository context

The repo is:

https://github.com/aniket-desh/tucker-ffn

It accompanies the draft “SwiGLU as a Routed CP Tensor Model.”

The existing repo reportedly includes:

- lib/tucker_ffn.py: TuckerFFN, SwiGLUFFN, SwiGLUFFNAligned.
- lib/lm.py: minimal LLaMA-style LM with swappable FFN.
- lib/routing.py: constant-alpha and related routing ablations.
- lib/permutation.py: same-index pairing permutation utilities.
- lib/activations.py: MLP input/output capture and CP channel quantities.
- lib/model_utils.py: HF model loading and SwiGLU layer enumeration.
- experiments/exp02_routing_stats.py: routing coefficient statistics.
- experiments/exp04_routing_ablation.py: constant-alpha perplexity ablation.
- experiments/exp09_pairing_permutation.py: same-index pairing experiments.
- experiments/exp10_synthetic_fitting.py: synthetic teacher-student verification of the k^2 separation.
- experiments/exp10_svd_construction.py: constructive SVD upper bound.
- experiments/exp11_train_lm.py: from-scratch matched-budget LM training.
- experiments/exp12_trained_tucker_analysis.py: stable rank analysis of trained Tucker slices.
- experiments/exp13_diagonal_projection.py: diagonal projection and per-gate SVD truncation.
- experiments/exp14b_tucker_teacher_distillation.py: Tucker-teacher distillation.
- experiments/exp17_robustness.py: model-family robustness.
- scripts for compute accounting, figures, and appendix generation.

First inspect the repo, read README.md, inspect the code, run tests if present, and run minimal smoke tests before adding new code.

Create:

docs/structured_tensor_ffn_sprint/

with:

- plan.md
- theory_notes.md
- architecture_spec.md
- research_log.md
- summary.md
- figures/
- tables/
- scripts_used.md

---

## 2. Background: what the previous work showed

The original draft argues that SwiGLU has an exact routed CP structure.

At one token position, a SwiGLU FFN computes:

y = U^T h
h_j(x) = (w_j^T x) SiLU(g_j^T x)

Using SiLU(z) = z sigmoid(z), each hidden unit is:

h_j(x) = (w_j^T x)(g_j^T x) sigmoid(g_j^T x)

So each unit is a rank-one bilinear interaction between two learned residual-stream directions, routed by an input-dependent scalar gate.

The output can be written as a third-order interaction tensor field:

A(x) = sum_j alpha_j(x) u_j outer w_j outer g_j

where alpha_j(x) = sigmoid(g_j^T x).

This is a CP decomposition of the interaction tensor at each input x, with shared factors and input-dependent routing weights.

Interpretation:

- u_j is an output direction.
- w_j is a main-feature direction.
- g_j is a gate direction.
- alpha_j(x) decides when the atom is active.
- The atom u_j outer w_j outer g_j says what interaction is available.

The key restriction is same-index pairing. The j-th main feature only interacts with the j-th gate feature and the j-th output atom. There are no direct cross-index interactions between w_i and g_j for i != j.

Viewed as Tucker, SwiGLU is the special case where the latent core is superdiagonal. The core is zero except on entries that pair the same hidden index across all modes.

The original proposed relaxation was a Tucker-core FFN:

p = P^T x
q = Q^T x
z_o = sum_{i,j} C_{oij} p_i SiLU(q_j)
y = R z

This replaces the superdiagonal CP core with a learned dense core C.

The original theorem showed an aligned-width separation:

Given fixed latent dictionaries P and Q, define the gate-wise output-by-main matrix:

V_j = R C[:,:,j]

Then an aligned SwiGLU needs sum_j rank(V_j) units to represent the same function exactly.

For generic Tucker slices, rank(V_j) = min(r, s), so the aligned SwiGLU width requirement is r min(r, s). In the balanced case r = s = k, this is k^2 units.

This theorem is valid, but its interpretation is now under revision.

The earlier interpretation was:

Dense Tucker is a better, richer FFN because it removes the diagonal bottleneck.

The new interpretation should be more cautious:

Dense Tucker is an expressive upper envelope, but it may be too arbitrary and too entangled for interpretability. The better architecture may be a structured relaxation between CP/SwiGLU and dense Tucker.

---

## 3. Thomas Dooms's feedback and what it means

Thomas Dooms gave the following feedback:

“Why the interest in tucker? Empirically, it performs much worse than CPD in a deep learning setting and I'd argue is much less interpretable. CPD gives a sparse core for free.

You can make a structured tucker by sparsifying the weights of the factor matrices using CPD (https://www.tensorlab.net/doc/ll1.html).”

Then, after a follow-up, he said:

“Since a swiglu is just a tensor, the ‘deliberateness’ mainly concerns avoiding arbitrary tensors in favor of principled decompositions.
Monarch/butterfly matrices are more parameter-efficient, while sparse CPDs are more interpretable.
I'd be very interested in a principled search over tensor network architectures that does either or both.”

Interpretation:

Thomas is not rejecting tensor methods. He is saying that a transformer FFN is already a tensorized object. The issue is not “should we use tensors?” The issue is “which tensor decomposition is principled?”

Dense Tucker may be arbitrary: it gives a large all-to-all latent core with many interactions, but the components may be hard to interpret. The learned factors can be basis-rotated and the core can absorb structure. Individual latent directions may not correspond to clean mechanisms.

CPD, by contrast, gives sparse core structure for free. A CP tensor is a sum of rank-one atoms. In Tucker language, CP corresponds to a superdiagonal core. This is restrictive, but it makes the components atomized: each component can be read as one output direction, one main direction, and one gate direction.

Thomas's suggested direction is not “use dense Tucker.” It is:

Search over principled tensor-network FFN architectures that improve either:

1. Interpretability, e.g. sparse CPD, routed CP atoms, block-CP, LL1, sparse cores.
2. Parameter/compute efficiency, e.g. Monarch or butterfly-structured matrices.
3. Both, e.g. sparse/block CP with structured factor matrices.

The project should therefore be reframed as:

A principled architecture search over structured tensor-network FFNs between routed CP/SwiGLU and dense Tucker.

Dense Tucker should be treated as an expressive upper-envelope baseline, not automatically as the main proposal.

---

## 4. Dmitry Manning-Coe's suggestion and what it means

Dmitry suggested:

“Once you have this sweep, pick some model structure like induction heads, etc. and try to sweep on them.”

Interpretation:

Do not only sweep architectures on generic scalar metrics like perplexity or MSE. After identifying candidate tensor-FFN structures, test them on a known mechanistic structure or circuit.

A “model structure” means a recognizable internal mechanism or circuit motif, such as:

- Induction heads.
- Copying circuits.
- Name-mover heads / IOI-style behavior.
- Factual recall circuits.
- Modular arithmetic circuits in algorithmic transformers.
- Feature superposition and sparse feature use in MLPs.
- Gated retrieval or conditional computation.

Induction heads are a canonical first target. An induction head detects repeated token patterns. In a sequence like:

A B ... A

the model should predict B after the second A. Mechanistically, induction heads are attention heads that attend from the second A back to the previous A/B context and copy/complete the repeated pattern.

Dmitry's suggestion means:

First, do the broad architecture sweep:
SwiGLU vs sparse CP vs LL1/block-CP vs structured Tucker vs dense Tucker vs Monarch/butterfly variants.

Then, choose a known circuit-like behavior and repeat the sweep under controlled conditions:

- Train small transformers with different FFN architectures on a synthetic induction/copying task.
- Measure not only loss/accuracy, but also whether induction heads emerge.
- Compare attention patterns, induction scores, ablation effects, and FFN activation structure.
- Ask whether a given FFN tensor structure makes the circuit easier to learn, more robust, or more interpretable.

This converts the project from “architecture X gets lower loss” into “architecture X changes the internal computational mechanism in a measurable way.”

For this sprint, do not attempt an enormous mechanistic interpretability project unless the architecture sweep is already clean. But at minimum, create a pilot experiment for the induction/copying setting or write a detailed plan with a smoke-tested implementation.

---

## 5. Required literature review before implementation

Before implementing new architectures, do a short but real literature review. Use web search and write notes in:

docs/structured_tensor_ffn_sprint/theory_notes.md

Read or skim:

1. Kolda & Bader, “Tensor Decompositions and Applications”
   Focus: CP, Tucker, CP as superdiagonal Tucker, identifiability, core structure.

2. TensorLab CPD docs:
   https://www.tensorlab.net/doc/cpd.html
   Focus: CPD as decomposition into rank-one terms.

3. TensorLab LL1 docs:
   https://www.tensorlab.net/doc/ll1.html
   Focus: decomposition in multilinear rank-(L_r, L_r, 1) terms. This is the key Thomas link.

4. TensorLab BTD docs:
   https://www.tensorlab.net/doc/btd.html
   Focus: block-term decomposition as a sum of low multilinear-rank terms.

5. Monarch matrices:
   https://arxiv.org/abs/2204.00595
   Focus: products of block-diagonal matrices, parameter efficiency, hardware efficiency.

6. Butterfly / structured linear transforms:
   Find a credible paper on butterfly factorizations in neural networks or fast structured transforms. Focus on why butterfly-like products can represent FFT-like transforms with few parameters.

7. Tensorizing Neural Networks:
   Novikov et al. 2015.
   Focus: tensor-train layers as compression of dense neural network matrices.

8. LoRA / low-rank adaptation:
   Focus: low-rank updates as a successful parameter-efficient rank constraint in transformers. Do not overfocus on finetuning; use it as conceptual evidence that rank constraints can be useful.

9. In-context Learning and Induction Heads:
   https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
   Focus: induction heads as a known circuit-like target for Dmitry's suggested circuit-level sweep.

10. Anthropic Transformer Circuits framework:
   https://transformer-circuits.pub/2021/framework/index.html
   Focus: model structures/circuits as objects of study.

Do not get stuck in literature review. Spend 60–90 minutes maximum unless you find something directly load-bearing.

---

## 6. Theoretical derivations to write before coding

Before adding models, derive the relevant math in theory_notes.md.

### 6.1 Derive SwiGLU as routed CP

Write the derivation from:

h_j(x) = (w_j^T x) SiLU(g_j^T x)

to:

h_j(x) = (w_j^T x)(g_j^T x) sigmoid(g_j^T x)

to:

A(x) = sum_j alpha_j(x) u_j outer w_j outer g_j

Explain:

- What are the modes?
- What is routed?
- What is fixed?
- Why is it CP?
- Why is the core superdiagonal if viewed as Tucker?
- Why is this interpretable?
- Why is it restrictive?

### 6.2 Derive dense Tucker FFN

Derive:

p = P^T x
q = Q^T x
z_o = sum_{i,j} C_{oij} p_i SiLU(q_j)
y = R z

Group by gate:

y(x) = sum_j V_j p SiLU(q_j)

where V_j = R C[:,:,j].

Explain:

- Each gate controls a matrix-valued transformation V_j.
- If V_j has rank greater than one, a single gate controls multiple CP atoms.
- Dense Tucker allows arbitrary all-to-all latent interactions.
- This gives expressivity but hurts atom-level interpretability.

### 6.3 Re-derive the aligned-width theorem

Re-derive the previous theorem:

An aligned SwiGLU representation needs at least sum_j rank(V_j) units to exactly represent a Tucker block with fixed P, Q.

Proof sketch:

- Since [P Q] has full column rank, p and q can be varied independently.
- Set all q_k = 0 except q_j = 1.
- This isolates one gate slice.
- The aligned SwiGLU representation of that slice is a sum of rank-one matrices.
- Therefore it needs at least rank(V_j) atoms.
- Summing over j gives the lower bound.
- A rank decomposition of each V_j gives the matching upper bound.

Then interpret the theorem differently from the old draft:

The theorem says the important control variable is not “Tucker or not Tucker,” but the per-gate matrix rank rank(V_j). Therefore a principled architecture should control this rank directly.

### 6.4 Derive LL1 / block-CP FFN

Use the TensorLab LL1 decomposition idea:

A third-order tensor can be written as a sum over blocks:

T = sum_b (A_b B_b^T) outer c_b

Each term is a low-rank matrix in two modes and a vector in the third mode.

Translate this into FFN language.

Candidate architecture:

For block b:

- gate direction g_b
- main factor A_b
- output factor U_b
- block rank L_b

Forward pass:

s_b(x) = SiLU(g_b^T x)

r_b(x) = A_b^T x      # L_b-dimensional main block response

block output = U_b r_b(x) s_b(x)

Full output:

y(x) = sum_b U_b A_b^T x SiLU(g_b^T x)

Expanded:

y(x) = sum_b sum_l u_{b,l} (a_{b,l}^T x) SiLU(g_b^T x)

This is grouped CP: several CP atoms share the same gate direction. When L_b = 1, this recovers CP/SwiGLU-like rank-one routed atoms. When L_b > 1, one route controls a low-rank block of interactions.

Explain why this is a middle ground:

- More expressive than rank-one CP/SwiGLU.
- More interpretable than dense Tucker.
- Per-gate rank is directly controlled by L_b.
- The core is block-sparse rather than dense.
- The architecture implements the constructive side of the aligned-width theorem.

### 6.5 Derive parameter counts and FLOPs

For each architecture, derive:

- parameter count
- approximate multiply-add count per token
- activation memory
- whether implementation uses dense GEMMs, batched GEMMs, sparse ops, or block-diagonal ops
- whether it is likely hardware-friendly

Architectures to include:

1. SwiGLU baseline.
2. Dense Tucker FFN.
3. Diagonal Tucker / SwiGLU-equivalent Tucker.
4. LL1 / block-CP FFN.
5. Sparse routed CP FFN.
6. Monarch-factorized SwiGLU or LL1.
7. Butterfly-factorized SwiGLU or LL1, if feasible.
8. Low-rank factorized dense projections as a simple baseline.

Be careful: parameter count alone is not enough. A lower-parameter architecture can be slower if the contraction is awkward.

### 6.6 Define interpretability metrics mathematically

Define metrics before using them.

Possible metrics:

1. Core density:
   fraction of nonzero or high-magnitude core entries.

2. Core entropy:
   entropy of normalized absolute core weights.

3. Effective atom count:
   for CP-like terms, normalize absolute contribution magnitudes per token and compute exp(entropy).

4. 90% mass fraction:
   fraction of atoms/blocks needed to explain 90% of total absolute contribution.

5. Block rank:
   rank or stable rank of each per-gate matrix V_j.

6. Ablation locality:
   performance drop from ablating one atom/block. Interpretable structures should have localized, non-diffuse effects.

7. Factor stability:
   train multiple seeds, align components by cosine similarity or assignment, measure whether factors recur.

8. Token selectivity:
   for each atom/block, find top activating tokens/contexts and measure whether they are semantically/coherently clustered.

9. Gating sparsity:
   distribution of sigmoid gates or block gates across tokens.

10. Decomposability:
   whether the learned function can be decomposed into a small number of atoms/blocks without a large reconstruction or perplexity penalty.

Do not claim “interpretability” solely from lower entropy. Say “proxy for interpretability.”

---

## 7. Architecture families to implement

Implement only what is feasible in the sprint. Prefer 2–3 architectures deeply over 10 superficially.

### Family A: Baselines

Use existing implementations where possible:

1. SwiGLUFFN.
2. TuckerFFN.
3. Aligned SwiGLU.
4. Standard dense FFN or GELU/SwiGLU baseline if available.

Verify correctness and shapes.

### Family B: LL1 / Block-CP GLU

This is the priority architecture.

Implement a module such as:

LL1GLUFFN(d_model, n_blocks, block_rank, gate_activation="silu")

Forward:

For x with shape [batch, seq, d]:

main = einsum or linear projection producing [batch, seq, n_blocks, block_rank]
gate = linear projection producing [batch, seq, n_blocks]
out_blocks = main * silu(gate)[..., None]
output = contract out_blocks with output factors U, producing [batch, seq, d]

Equivalent parameterization:

- A: [n_blocks, d, block_rank]
- G: [d, n_blocks]
- U: [n_blocks, block_rank, d]

Parameter count:

n_blocks * d * block_rank + d * n_blocks + n_blocks * block_rank * d

= n_blocks * d * (2 block_rank + 1)

Compare to SwiGLU parameter count:

roughly 3 d m

If m = n_blocks * block_rank, LL1 has fewer gates than SwiGLU but each gate controls block_rank atoms. Need match both parameters and compute fairly.

Variants:

- block_rank = 1 should reduce to a grouped CP/SwiGLU-like model.
- block_rank in {1, 2, 4, 8, 16}
- n_blocks adjusted to match parameter count.
- optional orthogonality penalty on U/A within each block.
- optional L1 penalty on block gates.
- optional top-k block gating during inference only as a diagnostic.
- optional learned block-size allocation if time remains.

Important sanity test:

LL1 with block_rank = 1 and appropriate dimensions should match a CP/SwiGLU-like structure. Its output should be equivalent to sum of rank-one routed atoms with one atom per gate.

### Family C: Sparse CPD / Sparse Routed CP

Implement or emulate sparse CP/SwiGLU variants:

1. L1 penalty on channel contributions.
2. L1 penalty on gates.
3. top-k gates at inference.
4. top-k gates during training only if stable.
5. group lasso over atoms.

Goal:

Can sparse CPD preserve performance while improving interpretability proxies?

Do not overcomplicate. Start with post-hoc top-k and L1 gate regularization.

### Family D: Structured Tucker

Implement constrained Tucker variants:

1. Dense Tucker baseline: existing.
2. Block-diagonal Tucker core.
3. Sparse core with L1 penalty.
4. Low-rank per-gate slices V_j.
5. Tucker initialized near diagonal, then measure whether it stays near diagonal.

Goal:

Separate “Tucker is expressive” from “Tucker learns useful structured cross-interactions.”

A good result would show that dense Tucker works only when it becomes low-stable-rank or block-sparse. A bad result would show dense Tucker diffuses everywhere and does not improve performance.

### Family E: Monarch / butterfly factor matrices

Only implement if time allows after LL1.

Motivation:

Thomas suggested Monarch/butterfly matrices as parameter-efficient structured linear maps. Use them not as arbitrary add-ons, but as structured replacements for factor matrices W, G, U, P, Q, R.

Start with a simple block-diagonal or Monarch-like factorization:

W ≈ B2 P B1

where B1 and B2 are block-diagonal matrices and P is a fixed permutation.

Candidate modules:

1. MonarchLinear: product of two block-diagonal matrices with a permutation.
2. BlockDiagonalLinear: simple control.
3. LowRankLinear: simple control.

Use these in:

- SwiGLU projections.
- LL1 A/G/U factors.
- Tucker P/Q/R factors.

Do not spend hours optimizing custom CUDA. Use PyTorch operations that are correct. Report wall-clock speed honestly.

---

## 8. Experiment suite

Pick two primary experiment tracks and one optional mechanistic pilot.

### Experiment 1: Synthetic tensor teacher-student recovery

Purpose:

Test expressivity claims under controlled ground truth.

Teachers:

1. CP/SwiGLU teacher.
2. LL1/block-CP teacher with known block_rank L.
3. Dense Tucker teacher with generic core.
4. Sparse/block Tucker teacher.

Students:

1. SwiGLU/CP.
2. LL1 with block_rank sweep.
3. dense Tucker.
4. sparse Tucker or low-rank Tucker.
5. MLP baseline if available.

Data:

x sampled from Gaussian or normalized residual-like distribution.
Output y = teacher(x).
Use train/val/test splits.

Metrics:

- validation MSE
- parameter count
- sample efficiency
- optimization stability over seeds
- recovered rank/block structure
- factor recovery when identifiable

Key sweeps:

- teacher block_rank L_teacher in {1, 2, 4, 8}
- student block_rank L_student in {1, 2, 4, 8, 16}
- fixed total parameter budget
- fixed number of atoms/blocks

Key plots:

1. MSE vs block_rank at matched parameter budget.
2. MSE vs parameter count for each architecture.
3. learned stable rank/core density vs performance.
4. recovery threshold: does LL1 exhibit a knee at the true block rank?

Hypotheses:

- CP student should recover CP teacher efficiently.
- LL1 student should recover LL1 teacher when block_rank >= teacher rank.
- Dense Tucker should recover everything but may use more parameters and less interpretable structure.
- LL1 should beat CP on LL1 teacher with less interpretability loss than dense Tucker.

Sanity checks:

- exact or near-exact recovery for easy cases.
- no leakage between train/test.
- MSE scale normalized.
- compare against linear/bilinear baseline.

### Experiment 2: Layer distillation from pretrained SwiGLU models

Purpose:

Test whether real pretrained FFN input-output maps are better approximated by CP, LL1, or Tucker structures.

Use existing activation-capture utilities if available.

Model choices:

Start with a small pretrained SwiGLU model that fits on A40, such as Qwen2.5-0.5B or another available HF model. If model download fails, fall back to synthetic and from-scratch experiments.

Data:

- WikiText-2 or a small FineWeb-Edu chunk.
- Capture residual stream inputs x to selected FFN layers and corresponding FFN outputs y.
- Use train/val/test activation splits.

Students:

1. SwiGLU/CP.
2. LL1 block rank {1,2,4,8,16}.
3. Tucker.
4. sparse Tucker.
5. low-rank bilinear baseline.

Metrics:

- output MSE
- cosine similarity to true FFN output
- downstream perplexity when replacing layer, if feasible
- parameter count
- throughput
- interpretability proxies:
  - active atom/block count
  - contribution entropy
  - ablation locality
  - factor stability over seeds
  - gate sparsity
  - block-rank utilization

Key plots:

1. Distillation error vs parameter count.
2. Distillation error vs interpretability proxy.
3. Pareto frontier: error vs effective active atoms.
4. LL1 block_rank sweep.
5. Per-layer comparison: early/middle/late layers.

Hypotheses:

- Dense Tucker may minimize MSE but have dense/high-entropy core.
- CP/SwiGLU may be more interpretable but need many atoms.
- LL1 may give a better Pareto point: much of Tucker’s expressivity with more structured blocks.
- If LL1 does not beat CP or Tucker, report that.

Important:

Do not claim that lower MSE implies better interpretability. Treat interpretability as a separate axis.

### Experiment 3: From-scratch small LM training

Purpose:

Test whether architecture differences survive actual language modeling training.

Use existing exp11_train_lm.py if possible. Keep scale modest.

Architectures:

1. SwiGLU baseline.
2. Dense Tucker.
3. LL1/block-CP.
4. sparse CP/SwiGLU.
5. optional Monarch-SwiGLU or Monarch-LL1.

Datasets:

Use whatever the repo already supports. Prefer a small stable setup:
- WikiText-2 for fast iteration.
- TinyStories or FineWeb-Edu subset if already implemented.

Metrics:

- validation perplexity
- train loss
- parameter count
- FLOPs/token estimate
- measured tokens/sec
- seed variance
- activation sparsity / interpretability proxies

Sweeps:

- block_rank in {1, 2, 4, 8}
- matched parameter budget
- matched hidden/intermediate dimension
- 3 seeds for the most promising comparison

Key plots:

1. Validation perplexity vs parameter count.
2. Validation perplexity vs measured throughput.
3. Pareto frontier: perplexity vs interpretability proxy.
4. Seed distribution for best candidates.

Be careful:

A 12–16h sprint is not enough for large-scale claims. Treat these as pilot results. Prefer honest small claims.

### Experiment 4: Mechanistic target pilot — induction/copying sweep

This is Dmitry’s suggestion.

Purpose:

After the architecture sweep, choose a known model structure/circuit and ask whether tensor-FFN structure changes its emergence or interpretability.

Use induction heads as the first target.

Controlled synthetic task:

Generate sequences with repeated patterns:

A B random random ... A B

or:

prefix contains key-value pairs, later query repeats key, target is associated value.

Train small transformers with identical attention architecture but different FFN modules:

1. SwiGLU.
2. LL1.
3. dense Tucker.
4. sparse CP.
5. optional no-FFN / attention-only control.

Metrics:

Behavioral:

- next-token accuracy on induction examples.
- accuracy as function of distance between repeated tokens.
- in-context copying score.
- generalization to unseen token pairs.

Mechanistic:

- induction-head score: attention from second occurrence of A to previous occurrence / previous B position.
- attention pattern heatmaps.
- causal ablation of candidate induction heads.
- logit contribution from copying path.
- FFN activation sparsity during induction examples vs random examples.
- whether FFN tensor blocks specialize to copying contexts.

Questions:

1. Which FFN architecture learns induction fastest?
2. Which architecture reaches best induction generalization?
3. Does FFN architecture change whether induction heads emerge?
4. Does LL1 produce more localized block activations on induction examples?
5. Does dense Tucker produce diffuse/nonlocal activations?
6. Does sparse CP produce clean but underpowered atoms?

Key plots:

1. Induction accuracy vs training step by architecture.
2. Induction score vs training step.
3. Attention heatmap for the clearest head.
4. FFN active atom/block count on induction vs random prompts.
5. Ablation effect of top atom/block/head.

Important:

Induction heads are mostly attention circuits, so do not expect FFNs to directly “be” induction heads. The question is subtler: does the FFN architecture affect the emergence, robustness, or interpretability of the attention circuit?

If this is too much, implement only a minimal pilot:

- train tiny models for a short run
- verify induction task works
- compute attention pattern score
- write the more complete plan in summary.md

---

## 9. The main architecture-search framing

The architectures should be arranged as a conceptual ladder:

### Level 0: Dense unstructured FFN

High expressivity, low tensor interpretability.

### Level 1: SwiGLU / routed CP

One gate routes one rank-one interaction atom.

Pros:
- atomized
- sparse core for free
- existing strong baseline
- interpretable decomposition

Cons:
- same-index bottleneck
- one route controls only one atom

### Level 2: Sparse CPD

Still atomized, but impose stronger sparsity or selection.

Pros:
- more interpretable
- fewer active atoms

Cons:
- may lose performance
- top-k / L1 may be hard to train

### Level 3: LL1 / block-CP

One gate routes a small low-rank block of atoms.

Pros:
- direct relaxation of CP
- controlled per-gate rank
- implements the theorem’s rank decomposition architecturally
- potentially good expressivity/interp tradeoff

Cons:
- less atomized than CP
- block semantics may be harder to interpret

### Level 4: Structured Tucker

Core allows cross-feature interactions but with sparsity/block/rank constraints.

Pros:
- richer interaction patterns
- can interpolate toward dense Tucker

Cons:
- core may become diffuse
- interpretability may degrade

### Level 5: Dense Tucker

All-to-all learned core.

Pros:
- most expressive in this family
- useful upper envelope
- useful diagnostic

Cons:
- arbitrary
- parameter-heavy
- basis/gauge ambiguity
- likely less interpretable

### Orthogonal axis: structured factor matrices

For any level, replace dense factor matrices with:

- low-rank factors
- block-diagonal factors
- Monarch-like factors
- butterfly-like factors

This targets parameter/compute efficiency rather than core interpretability.

The search should identify a Pareto frontier over:

- validation loss / perplexity
- parameter count
- FLOPs/token
- measured throughput
- active atom/block count
- core sparsity
- factor stability
- ablation locality

---

## 10. Concrete implementation plan

### Phase 1: Setup and reproduction, 1–2 hours

1. Clone/open repo.
2. Create docs/structured_tensor_ffn_sprint.
3. Inspect README.md and existing experiments.
4. Run dependency install.
5. Run tests or import checks.
6. Run a tiny synthetic fitting smoke test.
7. Run a tiny LM smoke test if feasible.
8. Write plan.md and first research_log.md entry.

The first log entry must state:

- what the repo already has
- what you believe the old claims are
- what Thomas’s critique changes
- what you will test first
- what would make you pivot

### Phase 2: Theory notes, 1–2 hours

Before implementation, write theory_notes.md with:

1. SwiGLU as routed CP.
2. Tucker FFN and gate-wise matrices.
3. Aligned-width theorem.
4. LL1/block-CP derivation.
5. Parameter/FLOP counts.
6. Interpretability metrics.
7. Hypotheses for each architecture.

This must be clear enough that another researcher could understand why LL1 is not arbitrary.

### Phase 3: Implement LL1/block-CP, 2–3 hours

Add:

- LL1GLUFFN module.
- tests for shapes.
- parameter count utility.
- equivalence sanity test for block_rank=1.
- integration into LM config / experiment scripts.
- synthetic teacher support if needed.

Keep code simple and robust.

### Phase 4: Synthetic teacher-student sweep, 2–3 hours

Run the controlled experiment first because it is the cleanest theory test.

Prioritize:

- CP teacher
- LL1 teacher with rank 2/4
- dense Tucker teacher
- students: CP/SwiGLU, LL1 rank sweep, Tucker
- matched parameter and matched rank settings
- at least 3 seeds for key points if feasible

Produce:

- MSE vs block rank
- MSE vs parameter count
- recovered rank/core sparsity metrics

### Phase 5: Real-model distillation or small LM training, 3–5 hours

Choose based on repo readiness.

If activation capture works:
Run layer distillation from a pretrained SwiGLU FFN.

If that fails:
Run from-scratch small LM training with SwiGLU vs LL1 vs Tucker.

Do not attempt huge training. The goal is a clean pilot.

Produce:

- validation MSE/perplexity vs parameter count
- interpretability proxy plots
- throughput table
- seed table if time allows

### Phase 6: Dmitry circuit pilot, 1–3 hours optional

If earlier phases are successful and time remains:

Implement a tiny induction/copying task.

Train very small transformers with different FFN modules.

Measure:

- induction accuracy
- attention induction score
- active FFN atom/block count on induction prompts
- simple ablations

If not enough time, write a detailed design and maybe smoke-test data generation.

### Phase 7: Writing and red-team, final 90 minutes minimum

Write summary.md.

Do not write chronological logs as the main summary. Structure around claims.

Run a red-team pass:

- Are the claims too strong?
- Are baselines fair?
- Are parameter counts matched?
- Are compute costs honest?
- Could the result be explained by more parameters?
- Could the result be explained by optimization?
- Did LL1 really help, or did it just change hidden width?
- Is “interpretability” measured or asserted?
- Are plots labeled clearly?
- Would a reader understand each graph without reading code?

---

## 11. Required deliverables

Create:

docs/structured_tensor_ffn_sprint/summary.md
docs/structured_tensor_ffn_sprint/research_log.md
docs/structured_tensor_ffn_sprint/plan.md
docs/structured_tensor_ffn_sprint/theory_notes.md
docs/structured_tensor_ffn_sprint/architecture_spec.md
docs/structured_tensor_ffn_sprint/scripts_used.md
docs/structured_tensor_ffn_sprint/figures/
docs/structured_tensor_ffn_sprint/tables/

summary.md must begin with an executive summary of at most 600 words.

The executive summary must contain:

- 2–5 key findings.
- Each finding supported by one simple graph or table.
- Clear statement of whether evidence is positive, negative, or inconclusive.
- No hype.

Required summary sections:

1. Executive summary.
2. Research question.
3. Background: SwiGLU as routed CP and why dense Tucker is suspicious.
4. Thomas’s critique and the structured-decomposition reframing.
5. Mathematical derivation of LL1/block-CP FFN.
6. Architecture families tested.
7. Experiments run.
8. Results.
9. Interpretability vs expressivity tradeoff.
10. Mechanistic/circuit-level pilot or plan.
11. What changed our mind.
12. What failed or was inconclusive.
13. Limitations.
14. Next steps.
15. Research map: code, scripts, figures, tables.

research_log.md must include hourly entries with:

- elapsed time
- what was tried
- what worked
- what failed
- decision/pivot reasoning
- links to artifacts

architecture_spec.md must include for each model:

- formula
- parameter count
- FLOP estimate
- expected interpretability
- implementation file
- experiment configs

scripts_used.md must include exact commands run.

---

## 12. Red-team checklist

Before finalizing, answer these explicitly in summary.md:

1. Did LL1 beat CP/SwiGLU at matched parameter count?
2. Did LL1 beat CP/SwiGLU at matched FLOPs or only matched parameters?
3. Did LL1 beat dense Tucker, or merely provide a better interpretability tradeoff?
4. Did dense Tucker actually use dense core structure?
5. Did sparse CP improve interpretability proxies without destroying performance?
6. Are any gains larger than seed noise?
7. Are any gains explained by hidden width or parameter count?
8. Are throughput measurements honest?
9. Are interpretability claims based on metrics, examples, or speculation?
10. Were baselines tuned fairly?
11. Did synthetic results transfer to real activations or LM training?
12. Did the induction/circuit pilot actually measure a circuit, or only behavior?
13. Did any experiment falsify the initial story?
14. What is the simplest explanation of the results?

If a result is a tie, call it a tie.
If a result is small, call it small.
If a result is inconclusive, explain exactly what would be needed next.

---

## 13. Suggested success modes

### Success mode 1: LL1 is a real middle ground

You show that LL1/block-CP recovers most of dense Tucker’s expressivity while retaining lower active block count, lower core entropy, more stable factors, or better ablation locality.

This would support the revised theory.

### Success mode 2: Sparse CP is enough

You show that CP/SwiGLU with sparsity or routing regularization matches LL1/Tucker while being much more interpretable.

This would support Thomas’s CPD-first instinct.

### Success mode 3: Dense Tucker wins but is uninterpretable

You show Tucker gets the best loss but with dense, high-rank, unstable core structure.

This is still useful: it separates expressivity from interpretability.

### Success mode 4: Structured matrices help efficiency

You show Monarch/block/butterfly-style factors reduce parameters or improve throughput at little quality loss.

This would address Thomas’s parameter-efficiency axis.

### Success mode 5: Everything loses to SwiGLU

You show that the fancy decompositions do not improve loss, efficiency, or interpretability under fair tests.

This is a clean negative and still valuable.

### Success mode 6: Circuit-level differences emerge

On the induction/copying task, you show that tensor-FFN structure affects emergence, speed, robustness, or interpretability of induction-like circuits.

This would validate Dmitry’s suggestion and open a more mechanistic direction.

---

## 14. Concrete experiments to prioritize if time is short

If you only have time for three things, do these:

1. Implement LL1GLUFFN and derive it clearly.
2. Run synthetic CP/LL1/Tucker teacher-student sweeps.
3. Run one real activation distillation or small LM training comparison.

If time remains, add:

4. Sparse CP/top-k routing diagnostic.
5. Induction/copying pilot.

Do not spend the whole sprint implementing Monarch/butterfly unless LL1 is already done and the initial experiments are clear.

---

## 15. Plot requirements

Every major claim needs a plot.

Make plots simple:

1. MSE vs block rank.
2. Perplexity or distillation MSE vs parameter count.
3. Performance vs effective active atoms/blocks.
4. Core density/stable rank by architecture.
5. Throughput vs architecture.
6. Induction accuracy / induction score vs training step if doing the circuit pilot.

Each plot must have:

- descriptive title
- labeled axes
- units
- legend
- clear caption
- note on seeds/error bars if applicable

Avoid beautiful but confusing plots.

---

## 16. Tone and final instruction

Be ambitious but skeptical.

This project started as “Tucker-core FFN generalizes SwiGLU.” The more mature version is:

SwiGLU is a routed CP tensor network. Dense Tucker is the maximally expressive relaxation but may be too arbitrary. The scientific question is whether structured tensor decompositions — especially LL1/block-CP and sparse CPD, possibly with Monarch/butterfly factor matrices — can find a better point on the interpretability/expressivity/efficiency frontier.

Do the mathematics first. Then run experiments that could prove you wrong.

The final report should make a reader think:

“I understand the tensor structure, I understand why these architectures were chosen, I understand what was tested, and I trust the claims because the author compared to simple alternatives and admitted limitations.”
