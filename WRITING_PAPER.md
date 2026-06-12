# Paper Writing Process Prompt: Structured Tensor FFNs as Mechanistic Tensor Networks

You are assisting with the writing phase of a research project on structured tensor-network feedforward networks for transformers. You will receive:

1. The main autoresearch prompt for the project.
2. The current repo and sprint outputs.
3. The earlier paper drafts on SwiGLU as a routed CP tensor model.
4. Neel Nanda’s writing advice.
5. This document.

Your job is to help turn the sprint into a clean, honest, publishable-style ML paper or workshop paper. The goal is not to make the work sound more impressive than it is. The goal is to make the claims precise, the evidence legible, and the theory-experiment loop clear.

The target style is a hybrid of:

- Thomas Dooms / bilinear-MLP mechanistic interpretability papers: clear tensor object, visual mechanistic diagrams, decompositions that preserve exact computation, experiments that show the decomposition reveals useful low-rank or sparse structure.
- Dmitry Manning-Coe / physics-meets-ML papers: explicit motivating questions, controlled toy tasks, careful comparison of features/encodings/dynamics, phase-diagram or sweep-based reasoning, and honest limitations.
- Neel Nanda-style interpretability writing: direct prose, concrete examples, clear claims, strong figures, no vague mysticism, no hiding weak evidence.

Do not imitate anyone’s writing voice mechanically. Instead, study their structural choices: how they motivate, define, derive, test, transition, visualize, and qualify claims.

---

## 1. Writing mission

The project began with the observation that SwiGLU can be written exactly as a routed CP tensor model. Each hidden channel contributes one rank-one interaction atom, and the SiLU gate routes that atom input-dependently. The early draft proposed dense Tucker as the natural relaxation of this same-index CP bottleneck.

Thomas Dooms’s feedback changed the framing. Dense Tucker may be expressive, but it may also be too arbitrary and too hard to interpret. CP’s superdiagonal core is not just a weakness; it is also what makes the decomposition sparse and atomized. A more principled direction is to search over structured tensor-network FFNs between CP/SwiGLU and dense Tucker, especially LL1/block-CP decompositions and sparse CPD variants. Monarch/butterfly-style factor matrices are a separate efficiency axis.

Dmitry Manning-Coe’s suggestion adds a second level: once the generic architecture sweep exists, test the same sweep on known model structures or circuits, such as induction/copying behavior. This prevents the paper from being just an architecture zoo. It asks whether the tensor structure changes how a recognizable mechanism emerges, decomposes, or becomes interpretable.

The paper should therefore not be titled or framed as “Tucker is better than SwiGLU.” The mature framing is:

> Transformer FFNs already instantiate structured tensor computations. We study which tensor-network structure gives the best tradeoff between expressivity, parameter efficiency, and mechanistic interpretability.

A good paper from this project should make one of the following claims, depending on what the experiments actually show:

1. LL1/block-CP is a useful middle ground between routed CP and dense Tucker.
2. Sparse CP/SwiGLU is already the best interpretable point, and denser cores do not help enough.
3. Dense Tucker is expressive but loses interpretability or parameter efficiency.
4. Structured factor matrices improve efficiency but do not solve interpretability.
5. All proposed variants fail under fair baselines, which is a valuable negative result.
6. Circuit-level sweeps reveal that FFN tensor structure affects the emergence or readability of known mechanisms.

Choose the claim after reading the results. Do not choose the claim first and force the story around it.

---

## 2. First action: produce a writing inventory

Before drafting prose, create:

`docs/paper_writing/writing_inventory.md`

It must contain:

1. A list of all experiments actually run.
2. A list of all figures/tables available.
3. A list of all claims the data could support.
4. A list of claims the data does not support.
5. A list of theory results that are exact.
6. A list of empirical results that are preliminary.
7. A list of baselines and fairness caveats.
8. A list of missing experiments that would be required for a stronger paper.

Do this before writing the introduction. The introduction must be written around what the evidence supports, not around what we hoped would be true.

---

## 3. Study the target paper styles

Read the following papers and produce:

`docs/paper_writing/style_memo.md`

The style memo should summarize the observed prose, argument flow, figure design, theory-experiment transitions, and citation habits.

### 3.1 Thomas Dooms / bilinear MLP style

Read:

- “Bilinear MLPs enable weight-based mechanistic interpretability.”
- Earlier technical note if available: “Weight-based Decomposition: A Case for Bilinear MLPs.”
- The bilinear-decomposition repo tutorials if useful.

Observe the following:

#### Argument style

The paper starts from a concrete interpretability obstacle: ordinary MLP nonlinearities obscure how weights construct features. It then introduces a modified architecture, bilinear MLPs, whose computation can be expressed exactly as a third-order tensor. The argument is not “tensor methods are cool.” The argument is “this architecture exposes a weight-level object we can decompose.”

Imitate this structure for our project:

- Start with the obstacle: transformer FFNs contain much of the model’s parameters, but their multiplicative feature interactions are hard to interpret.
- Show the exact algebraic object: SwiGLU is a routed CP tensor field.
- Explain the bottleneck: one gate routes one rank-one atom.
- Explain the design question: what structured relaxation improves expressivity without destroying interpretability?

#### Theory-to-method transition

Thomas-style papers often move like this:

1. Define an exact computation.
2. Name the object it induces, e.g. interaction matrix or bilinear tensor.
3. Point out a simplification, symmetry, or decomposition.
4. Turn that decomposition into an analysis method.
5. Validate it on controlled tasks, then richer tasks.

For our paper, use:

1. Define SwiGLU’s routed CP tensor.
2. Name the rank-one routed atom.
3. Show dense Tucker replaces the superdiagonal core with an all-to-all core.
4. Show LL1/block-CP replaces one atom per gate with a low-rank block per gate.
5. Validate with synthetic teacher-student experiments, real-layer distillation, and optional small LM or circuit tasks.

#### Prose features

Use direct definitions:

- “We call this matrix...”
- “This tensor allows us to...”
- “The key object is...”
- “The decomposition is exact...”
- “This makes it possible to...”

Avoid vague statements like:

- “This may provide insight.”
- “This is potentially meaningful.”
- “This opens the door to interpretability.”

Replace them with:

- “This gives a directly ablatable unit.”
- “This reduces the per-gate interaction rank from dense to controlled rank L.”
- “This lets us measure whether the model uses sparse atoms or diffuse blocks.”

#### Figures and diagrams

Thomas-style diagrams are unusually useful. They use:

- Panel labels A), B), C).
- Tensor/matrix block diagrams with pastel colors.
- Small algebra directly under the diagram.
- A caption that explains the whole method without requiring the reader to inspect the main text.
- Mechanistic diagrams before quantitative plots.

For this paper, create at least one core diagram:

Panel A: SwiGLU as routed CP.

- show x projected by W and G,
- show elementwise multiplication/gating,
- show one rank-one tensor atom u_j outer w_j outer g_j,
- show alpha_j(x) as the route.

Panel B: Dense Tucker.

- show P and Q latent features,
- show dense core C connecting many p_i to many q_j,
- visually mark the all-to-all core as expressive but diffuse.

Panel C: LL1/block-CP.

- show one gate routing a small low-rank block,
- show grouped CP atoms sharing a route,
- mark block rank L.

Use clean colors consistently:

- input/residual stream: gray or neutral.
- main features: blue.
- gate features/routes: green.
- output directions: purple.
- dense Tucker core: red/orange if emphasizing entanglement.
- LL1 block: blue-green grouped block.

Keep diagrams minimal. Do not make them decorative. Every arrow should correspond to a mathematical operation.

#### Plot style

Thomas-style quantitative figures often combine:

- a mechanistic object visualization,
- a low-rank/truncation curve,
- a circuit or feature-level scatter,
- captions that say what the plot means.

For this paper, preferred plots:

1. Error vs block rank.
2. Error vs parameter count.
3. Error vs active atom/block count.
4. Core density or stable rank vs validation loss.
5. Distillation error vs interpretability proxy.
6. For circuit pilot: induction score vs training step and active block count on induction examples.

Use multi-panel figures when panels answer one question. Do not put unrelated experiments into one figure.

### 3.2 Dmitry Manning-Coe style

Read:

- “Grokking vs. Learning: Same Features, Different Encodings.”
- “Interactions Between Crosscoder Features: A Compact Proofs Perspective.”
- Any related project reports or sprint notes if supplied.

Observe the following:

#### Argument style

Dmitry-style ML writing often begins with a sharp conceptual question. For example:

- Do grokking and ordinary learning lead to fundamentally different models?
- How much of a model’s behavior is explained by sparse features alone, and how much is left to circuits?

The paper then builds controlled comparisons around that question.

For our paper, use a question like:

> Is the SwiGLU diagonal CP structure a useful interpretability bias, an expressivity bottleneck, or both?

or:

> Can structured tensor decompositions improve the expressivity of transformer FFNs without losing the atomized interpretability of CP?

#### Transition pattern

Dmitry-style papers often use transitions like:

- “Having established X, we now ask Y.”
- “To make this comparison clean, we choose...”
- “This raises the question...”
- “We therefore define...”
- “We hence conclude...”
- “We emphasize that...”

This is a good fit for our paper. Use these transitions to connect theory and experiments.

Example:

“Having established that dense Tucker can represent cross-channel interactions that aligned SwiGLU cannot compactly simulate, we now ask whether this extra expressivity is useful in practice. To make the comparison clean, we first train students on synthetic teachers with known tensor structure.”

Then:

“Having shown that LL1 recovers low-rank block teachers at the predicted rank, we now ask whether the same structure appears in real pretrained FFN maps. We therefore distill individual FFN layers from a pretrained SwiGLU transformer.”

Then:

“Having measured approximation quality, we now ask whether the decomposition remains interpretable. We therefore measure active block count, ablation locality, and factor stability across seeds.”

#### Controlled tasks and hand-built measures

Dmitry-style work often defines task-specific measures rather than relying only on generic loss. In the grokking paper, feature measures are designed for Ising and modular addition separately. In the crosscoder paper, the interaction metric is derived from an error term in a proof and then used as an experimental object.

For our paper, do the same:

- For synthetic teachers, measure recovery error and recovered block rank.
- For pretrained FFN distillation, measure output MSE, cosine similarity, and downstream replacement loss if available.
- For interpretability, measure active atom/block count, ablation locality, and factor stability.
- For induction/copying, measure induction score and active block count on induction examples.

Do not rely on perplexity alone.

#### Honesty and limitations

Dmitry-style writing explicitly says when a theoretical object is not yet practically strong enough. Example pattern:

- “Although this procedure is not yet practically applicable to full models, the error term itself is useful as a measure.”

Use similar honesty here:

- “Although our sprint-scale experiments do not establish a scaling law, they isolate a controlled tradeoff between block rank and interpretability proxies.”
- “Although dense Tucker is the most expressive architecture in this family, its learned core is diffuse under our metrics.”
- “Although LL1 improves synthetic recovery, the real-layer distillation results are inconclusive.”

Do not hide weak results.

### 3.3 Citation style

Use standard ML citation style.

Do not write:

- “In reference [29], the authors...”
- “As shown in Ref. 12...”

Prefer:

- “Previous work on bilinear MLPs shows that removing the gate nonlinearity yields a third-order tensor that can be decomposed directly from the weights \citep{pearce2025bilinear}.”
- “Sparse dictionary methods decompose activations into feature directions \citep{bricken2023monosemanticity, cunningham2023sparse}.”
- “Grokking studies often use modular addition because the learned Fourier features are well characterized \citep{nanda2023progress, gromov2024grokking}.”

Use `\citet{}` when the authors are grammatically part of the sentence:

- “\citet{pearce2025bilinear} analyze bilinear MLPs via weight-space decompositions.”

Use `\citep{}` for background claims:

- “GLU variants are common in modern transformers and often outperform ordinary MLPs at matched scale \citep{shazeer2020glu}.”

Citation clusters should support one sentence, not an entire paragraph. Do not attach five citations to a vague sentence. Make the claim precise.

---

## 4. Paper structure

The paper should probably follow this structure, unless the results force a different one.

### Title candidates

Choose a title after the results are known. Possible forms:

1. “Structured Tensor FFNs: Between Routed CP and Dense Tucker”
2. “Routed Block-CP Feedforward Networks for Mechanistic Tensor Decomposition”
3. “How Much Tensor Structure Should a Transformer FFN Have?”
4. “SwiGLU as Routed CP and the Case for Structured Tensor FFNs”
5. “Structured Tensor Decompositions of Transformer Feedforward Networks”

Avoid titles claiming superiority unless the results prove it.

### Abstract template

The abstract should have exactly one central claim.

Template:

“Transformer feedforward networks contain much of a language model’s parameter budget, but their multiplicative interactions are less mechanistically understood than attention. We show that SwiGLU admits an exact routed CP decomposition: each hidden channel contributes a rank-one interaction atom routed by an input-dependent gate. This exposes a diagonal core restriction. Rather than replacing this restriction with an arbitrary dense Tucker core, we study a structured family of tensor FFNs that interpolates between routed CP and dense Tucker by allowing each gate to route a low-rank block of interactions. We derive the corresponding LL1/block-CP form and compare CP, block-CP, sparse CP, and Tucker variants on [experiments]. We find [main result]. These results suggest that [careful conclusion].”

Fill in [main result] only after reading the data.

### Introduction

The introduction should be approximately 5–7 paragraphs.

Paragraph 1: Why FFNs matter.

- Modern transformers spend a large fraction of parameters and FLOPs in FFN blocks.
- Attention has a rich mechanistic vocabulary; FFNs are harder.
- GLU/SwiGLU blocks are multiplicative, so tensor structure is natural.

Paragraph 2: Exact routed CP observation.

- Show in prose that SiLU decomposes into a linear factor times a sigmoid route.
- Each channel is a rank-one interaction atom.
- The full block is a routed CP tensor field.

Paragraph 3: The tension.

- CP is interpretable because it is atomized/sparse.
- CP is restrictive because each gate routes one rank-one atom.
- Dense Tucker relaxes the restriction but creates a diffuse all-to-all core.

Paragraph 4: Proposed reframing.

- The right question is not “CP or Tucker?”
- The right question is the Pareto frontier between expressivity, efficiency, and interpretability.
- Introduce LL1/block-CP as the main structured relaxation.

Paragraph 5: What we test.

- Synthetic teacher-student recovery.
- Real FFN layer distillation or LM training.
- Interpretability proxies.
- Optional circuit-level induction/copying pilot.

Paragraph 6: Contributions.

Use a numbered list. Each contribution must be true.

Possible contributions:

1. We formulate SwiGLU as a routed CP tensor model and identify the diagonal-core restriction.
2. We derive a structured LL1/block-CP FFN in which each route controls a low-rank block of interactions.
3. We compare CP, block-CP, sparse CP, and Tucker variants under matched budgets.
4. We measure both approximation quality and interpretability proxies, showing [actual result].
5. We provide a mechanistic pilot on [induction/copying] showing [actual result or limitations].

Do not include a contribution unless it is supported.

### Related work

Organize by conceptual role, not by chronology.

Subsections:

1. Gated FFNs and bilinear MLPs.
2. Tensor decompositions in neural networks.
3. Sparse dictionary learning and weight-based interpretability.
4. Structured matrices for efficient networks.
5. Circuit-level evaluations and induction heads.

The related work should explain the niche:

- Bilinear MLP work studies architectures where weight-based tensor decomposition is exact.
- This paper studies SwiGLU-like routed tensors and structured relaxations of their CP core.
- Tensorized neural network work often targets compression; this paper targets the expressivity/interp/efficiency frontier.
- Sparse dictionary learning studies activation features; this paper studies architectural tensor factors and routed interactions.

### Theory / model section

This section must be clean and self-contained.

Recommended subsections:

1. SwiGLU as routed CP.
2. Dense Tucker and the diagonal bottleneck.
3. LL1/block-CP as a structured relaxation.
4. Parameter and FLOP counts.
5. Interpretability proxies.

Keep the math minimal but exact. Every equation should earn its place.

The key equations:

1. SwiGLU hidden channel.
2. SiLU factorization.
3. Routed CP tensor.
4. Tucker core FFN.
5. Gate-wise matrix V_j.
6. LL1/block-CP form.
7. Parameter counts.

Do not include a long proof in the main text unless the theorem is central to the final claim. Put detailed proof in appendix.

### Experiments section

Each experiment subsection must have this structure:

1. Question.
2. Setup.
3. Metric.
4. Result.
5. Interpretation.
6. Caveat.

Example:

“Question. Does block-CP recover low-rank routed teachers at the predicted block rank?

Setup. We generate teachers with known CP, LL1, and dense Tucker structure...

Metric. We report validation MSE and recovered stable rank...

Result. ...

Interpretation. ...

Caveat. This is a controlled teacher-student setting and does not by itself imply improved language modeling.”

This pattern prevents experiment sections from becoming a dump of results.

### Discussion

The discussion should distinguish three things:

1. What is established.
2. What is plausible but not established.
3. What failed or remains open.

Use explicit subsections:

- “What the tensor view buys”
- “When expressivity hurts interpretability”
- “What block rank appears to control”
- “Limitations”
- “Next steps”

Do not end with vague future work. End with a concrete next experiment.

---

## 5. Theory-to-experiment transitions

The paper should repeatedly use the same pattern:

1. Derive a structural quantity.
2. Explain why that quantity matters.
3. Design an experiment that directly measures it.

Examples:

### Transition from CP derivation to diagonal bottleneck

“The routed CP form exposes a precise restriction: each route controls one rank-one interaction atom. This makes the decomposition atomized, but it prevents a single route from coordinating multiple main-output interactions. We next compare this restriction with a dense Tucker core, where each route controls a full matrix of interactions.”

### Transition from Tucker theorem to LL1

“The aligned-width theorem shows that the cost of simulating a Tucker slice with routed CP atoms is its matrix rank. This suggests that the useful design variable is not whether the core is CP or Tucker, but the rank each route is allowed to control. We therefore introduce a block-CP architecture in which each gate routes a rank-L interaction block.”

### Transition from LL1 to synthetic experiments

“Because LL1 has an explicit block-rank parameter, synthetic teachers give a clean first test: if the derivation is the right abstraction, student error should drop when the student block rank reaches the teacher block rank.”

### Transition from synthetic to real models

“Synthetic teachers verify the algebraic capacity of the architecture. Real pretrained FFN layers test whether this capacity is relevant to learned transformer computations.”

### Transition from distillation to interpretability

“Approximation error alone does not establish mechanistic value. Dense Tucker should approximate many functions well, but may do so through diffuse cores. We therefore measure whether the learned representation is sparse, stable, and locally ablatable.”

### Transition to induction/circuit pilot

“Architecture-level metrics do not tell us whether the decomposition helps explain known circuits. We therefore run a pilot on an induction/copying task, where a recognizable attention mechanism can be measured directly.”

---

## 6. Figure plan

The paper should be figure-driven. A reader should understand the story by reading the figures and captions.

### Figure 1: Architecture diagram

Goal: explain the entire conceptual ladder.

Panels:

A. SwiGLU as routed CP.
B. Dense Tucker as all-to-all core.
C. LL1/block-CP as grouped CP atoms sharing a route.
D. Pareto cartoon: CP is interpretable/restrictive; Tucker is expressive/diffuse; LL1 is the proposed middle.

Caption should define “route,” “atom,” “block rank,” and “core density.”

### Figure 2: Synthetic teacher-student sweep

Goal: show whether the math predicts behavior.

Panels:

A. CP teacher: CP/SwiGLU and LL1 rank 1 should recover.
B. LL1 teacher: error drops when student block rank reaches teacher rank.
C. Dense Tucker teacher: dense Tucker best; LL1 improves with rank; CP struggles.
D. Parameter-matched comparison.

### Figure 3: Real FFN distillation or LM comparison

Goal: show relevance to real models.

Panels:

A. Error/perplexity vs parameter count.
B. Error/perplexity vs FLOPs or tokens/sec.
C. Error/perplexity vs active atom/block count.
D. Best models highlighted on Pareto frontier.

### Figure 4: Interpretability diagnostics

Goal: avoid handwavy interpretability.

Panels:

A. Effective active atoms/blocks per token.
B. Ablation locality distribution.
C. Factor stability across seeds.
D. Core density/stable rank.

### Figure 5: Circuit pilot if available

Goal: implement Dmitry’s suggestion.

Panels:

A. Induction accuracy over training.
B. Induction head score over training.
C. Attention heatmap for a representative model.
D. FFN block activation contrast on induction vs random examples.

If circuit pilot is not ready, use Figure 5 as a schematic and place it in “Future work” or appendix, not as evidence.

---

## 7. Visual style guide

Use a restrained, scientific aesthetic.

### Diagrams

- Use simple block diagrams with limited colors.
- Use consistent color semantics across all figures.
- Use arrows for actual computations only.
- Avoid decorative gradients or unnecessary 3D effects.
- Label tensors by mode: output, main, gate.
- Use panel letters A), B), C) in the upper-left corner.
- Include the minimal equation under or beside each diagram.

### Plots

- Use large axis labels.
- Use direct titles that state the comparison, not the conclusion.
- Use captions to state the conclusion.
- Use error bars when multiple seeds exist.
- If there is only one seed, label it clearly as a pilot.
- Mark matched-parameter points with a vertical line or marker.
- Use log scale for MSE when appropriate.
- Use consistent colors for architectures.

Suggested color mapping:

- SwiGLU/CP: blue.
- sparse CP: teal.
- LL1/block-CP: green.
- structured Tucker: orange.
- dense Tucker: red.
- standard baseline/MLP: gray.

### Captions

Captions should be self-contained. A good caption has:

1. What was varied.
2. What was measured.
3. What the plot shows.
4. One caveat if necessary.

Bad caption:

“Results of the synthetic experiment.”

Good caption:

“Synthetic LL1 teacher-student recovery. The teacher has block rank 4. Student error drops sharply when the LL1 student reaches block rank 4, while the rank-one CP student remains underparameterized at the matched atom budget. Curves show mean over three seeds; shaded bands show one standard deviation.”

---

## 8. Claim discipline

Use a claim registry.

Create:

`docs/paper_writing/claim_registry.md`

Each claim should be a card:

```md
## Claim N: [short claim]

Status: established / likely / speculative / false / abandoned

Evidence:
- Figure/table:
- Experiment:
- Seeds:
- Baselines:

Possible alternative explanations:
- More parameters?
- Different optimization difficulty?
- Bad hyperparameters?
- Synthetic-only artifact?
- Measurement proxy not valid?

Allowed wording:
- Strong:
- Medium:
- Weak:

Current wording to use in paper:
```

Do not let any claim enter the abstract unless its status is “established” or clearly framed as a preliminary finding.

Examples of disciplined wording:

Strong:

“LL1 exactly recovers CP as the block-rank-one case.”

Medium:

“In synthetic teacher-student experiments, LL1 exhibits the predicted recovery threshold at the teacher block rank.”

Weak:

“Our pilot distillation results suggest that pretrained FFN maps may use low-rank routed blocks, but the evidence is not yet robust across layers and models.”

Forbidden:

“LL1 is a better transformer FFN.”

unless the sprint produced a fair, replicated, matched-budget LM result.

---

## 9. How to write the introduction

Draft the introduction last, after the main results are known.

Use this paragraph-level outline:

### Paragraph 1: FFNs as the underinterpreted part of transformers

Make the motivation concrete. Attention has a developed interpretability vocabulary; FFNs are large, nonlinear, and multiplicative.

### Paragraph 2: Exact tensor view of SwiGLU

Present the core observation in words before equations. SwiGLU is not approximately tensor-like; its multiplicative gate gives an exact routed tensor form.

### Paragraph 3: CP’s benefit and cost

Explain that CP’s superdiagonal core is both interpretable and restrictive.

### Paragraph 4: The wrong naive move

Say dense Tucker is a natural mathematical relaxation but may be too unconstrained. This should sound like an honest revision of the original idea.

### Paragraph 5: Structured relaxation

Introduce LL1/block-CP as the rank-controlled middle ground.

### Paragraph 6: Empirical program

Describe the sweep: synthetic teachers, pretrained layer distillation or LM training, interpretability diagnostics, optional circuit pilot.

### Paragraph 7: Contributions

List true contributions.

Write the introduction with the same clarity as the sprint summary. A tired reviewer should understand the entire paper from the first page.

---

## 10. How to write the theory section

The theory section should feel like a derivation, not a notation dump.

Every subsection should start with an intuitive sentence.

Example:

“Each SwiGLU channel multiplies two scalar projections of the residual stream. This makes the channel a rank-one interaction, routed by the sigmoid part of SiLU.”

Then give the equation.

After each equation, say what the symbols mean and what the equation reveals.

For the LL1 section, write:

“Dense Tucker lets each gate control a full matrix of main-to-output interactions. The aligned-width theorem shows that the relevant quantity is the rank of this matrix. LL1 makes this rank a hyperparameter.”

Then derive:

block output = U_b A_b^T x SiLU(g_b^T x)

Then expand it into atoms:

sum_l u_{b,l} (a_{b,l}^T x) SiLU(g_b^T x)

Then explain:

This is grouped CP: several atoms share one route.

This three-sentence pattern is important:

1. Matrix form.
2. Atom expansion.
3. Interpretation.

---

## 11. How to write experiments

Experiments should be justified by the theory.

Do not write:

“We test several architectures on synthetic data.”

Write:

“The LL1 derivation predicts that block rank, not dense core size, is the relevant control variable. We test this prediction in a teacher-student setting where the teacher block rank is known.”

Do not write:

“We then test on language modeling.”

Write:

“Synthetic teachers test whether the architecture can represent the intended tensor structure. They do not show whether real transformer FFNs use that structure. We therefore distill pretrained FFN layers and measure the approximation/interpretability frontier.”

For every experiment, include a short “Why this experiment?” sentence.

---

## 12. How to discuss negative results

Negative results are valuable if they are precise.

Examples:

If dense Tucker performs poorly:

“Dense Tucker is more expressive as a function class, but in our matched-budget training runs it was harder to optimize than LL1 and SwiGLU. This suggests that removing the diagonal constraint does not automatically yield a useful training prior.”

If LL1 performs poorly:

“LL1 gives a principled rank-controlled relaxation, but our experiments do not show a practical advantage over routed CP under the tested budgets. This supports Thomas Dooms’s concern that CP’s atomization may be the more important property.”

If interpretability proxies are inconclusive:

“Our sparsity and ablation-locality metrics should be treated as proxies. They show that LL1 uses fewer active routes than dense Tucker, but they do not establish that individual blocks are monosemantic.”

If circuit pilot fails:

“The induction pilot was useful primarily as a check on experimental infrastructure. We did not find evidence that FFN tensor structure changes induction-head emergence at this scale.”

Never dress up failure as success. Explain what was learned.

---

## 13. Abstract writing rules

The abstract should not be a compressed related-work section. It should contain:

1. Problem.
2. Exact observation.
3. Proposed structured family.
4. Experiments.
5. Main result.
6. Careful implication.

Avoid:

- “We propose a novel framework...”
- “Extensive experiments show...”
- “Significantly improves...”

unless true.

Prefer:

- “We study...”
- “We show algebraically...”
- “We compare...”
- “In controlled experiments...”
- “Our results suggest...”

---

## 14. Tone guide

The tone should be:

- precise,
- curious,
- skeptical,
- mathematically grounded,
- not overhyped.

Good phrases:

- “This exposes...”
- “This suggests...”
- “To isolate...”
- “To test whether...”
- “We therefore compare...”
- “This supports...”
- “This does not imply...”
- “A simpler explanation is...”

Avoid:

- “revolutionary,”
- “breakthrough,”
- “powerful framework,”
- “clearly superior,”
- “interpretable” without specifying a metric,
- “principled” without saying what principle.

Use “principled” only when the principle is named:

- exact algebraic equivalence,
- nested architecture family,
- controlled rank parameter,
- matched parameter/FLOP budget,
- derived interaction metric,
- identifiable atom/block structure.

---

## 15. Writing schedule for Claude

Use this schedule after the research sprint completes.

### Hour 0–1: inventory and source study

- Read sprint summary, logs, and theory notes.
- Read Neel Nanda advice.
- Read the two old drafts.
- Skim Thomas and Dmitry target papers.
- Write `style_memo.md` and `writing_inventory.md`.

### Hour 1–2: claim registry and figure-first outline

- Create `claim_registry.md`.
- Decide which claims are established.
- Pick the 3–5 figures that tell the story.
- Write captions before writing the main text.

### Hour 2–3: paper outline

- Create `paper_outline.md` with every section and paragraph-level bullets.
- Put each figure next to the claim it supports.
- Decide whether the paper is positive, negative, or mixed.

### Hour 3–5: first draft

Write in this order:

1. Theory/model section.
2. Experiment sections.
3. Results interpretation.
4. Limitations.
5. Related work.
6. Introduction.
7. Abstract.

Do not start with the abstract.

### Hour 5–6: red-team pass

Create `red_team.md`.

Ask:

- What is the strongest claim we can defend?
- What claim would a reviewer attack first?
- Are baselines fair?
- Are metrics meaningful?
- Do figures support captions?
- Are we confusing synthetic and real-model evidence?
- Are we overclaiming interpretability?

Revise accordingly.

### Hour 6–7: polish

- Tighten intro.
- Make captions self-contained.
- Add limitations.
- Fix citation style.
- Remove unsupported adjectives.
- Ensure every section has a clear purpose.

---

## 16. Paper artifacts to produce

Create:

`docs/paper_writing/`

with:

- `style_memo.md`
- `writing_inventory.md`
- `claim_registry.md`
- `figure_plan.md`
- `paper_outline.md`
- `red_team.md`
- `draft.md` or `paper.tex`
- `abstract_versions.md`
- `intro_versions.md`
- `caption_bank.md`

If writing LaTeX, use standard ML conference style with natbib. If writing Markdown, still structure it as if it will become LaTeX.

---

## 17. Final red-team checklist

Before declaring the draft done, answer:

1. Can a reader state the main question after reading the first page?
2. Is the main claim supported by at least one clear figure?
3. Are theory claims separated from empirical claims?
4. Are synthetic results separated from real-model results?
5. Are interpretability claims measured rather than asserted?
6. Are parameter counts and FLOPs reported?
7. Are baselines fair?
8. Are negative results included honestly?
9. Does every experiment have a stated purpose?
10. Does every figure have a self-contained caption?
11. Are citations implicit and natural rather than “in reference 29” style?
12. Are limitations specific rather than generic?
13. Does the paper end with a concrete next step?

If the answer to any of these is no, revise.

---

## 18. North star

The ideal paper should feel like this:

1. A simple algebraic observation reveals that SwiGLU is already a routed CP tensor model.
2. This observation clarifies a tension: CP is interpretable because it is sparse, but restrictive because each route controls only one rank-one atom.
3. Dense Tucker is the obvious mathematical relaxation, but it is not obviously the right scientific object.
4. LL1/block-CP gives a rank-controlled middle ground.
5. Experiments test whether that middle ground matters.
6. The conclusion is honest about what worked, what failed, and what remains unproven.

The paper should not read like a sequence of experiments. It should read like a controlled investigation of one question:

> How much tensor structure should a transformer FFN have if we care about both expressivity and interpretability?

