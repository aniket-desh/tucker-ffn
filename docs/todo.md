# SwiGLU as a Low-Rank Tensor Model — Updated Todo List

## Status after note revision (v2)

The note now establishes the **exact routed-CP decomposition** of SwiGLU: for every input x, the FFN output is y = A(x) · (x ⊗ x), where A(x) = Σ_j α_j(x) u_j ⊗ w_j ⊗ g_j with α_j(x) = σ(g_j^⊤ x). This eliminates the Taylor expansion vulnerability entirely. The Tucker generalization is cleanly stated and SwiGLU recovery via superdiagonal core is explicit.

### What's done
- ~~Taylor expansion regime~~ — **resolved**. The framework is now exact, not approximate. No more "leading-order" hedging.
- ~~Epistemic claim~~ — **committed**. The note claims exact structural decomposition, not approximation.
- ~~Clean proposition for CP structure~~ — **done** (Equation 10).
- ~~Diagonal core definition~~ — **done**, improved to "superdiagonal interaction core" / "same-index channel coupling."
- ~~Tucker generalization as definition + recovery~~ — **done** (Equations 11–12, recovery via C_αij = δ_αi δ_ij).
- ~~Language revision~~ — **done** throughout.
- ~~Literature review~~ — **done**. See Lit_Review.md. No existence threat found. Key findings summarized below.

---

## Phase 0: Literature review — COMPLETE

Full details in Lit_Review.md. Covered 15 papers across 4 research questions: existence threats, identifiability/gauge freedom, expressivity of bilinear/quadratic layers, and SAEs/transcoders on structured activations.

### Key findings

1. **No existence threat.** No one has cast SwiGLU (or any gated MLP with input-dependent routing) as a CP-decomposed interaction tensor. The exact routed-CP formulation is novel.

2. **Closest prior work: Pearce et al. (2024, ICLR 2025).** They show *bilinear* MLPs (GLUs without the sigmoid nonlinearity) can be expressed as a third-order tensor and decomposed via eigendecomposition for interpretability. **Critical difference:** they analyze the static bilinear case only (α_j ≡ 1). Your framework handles the full SwiGLU with input-dependent routing, which is the architecture actually deployed in production. Position as: generalization from bilinear to routed-gated case. Cite prominently.

3. **Jayakumar et al. (ICLR 2020) provides the positioning frame.** Their multiplicative interaction taxonomy describes diagonal gating as a special case of a general 3D-tensor interaction y = z^T W x. Your same-index coupling constraint is exactly their diagonal restriction. Your Tucker generalization is the natural relaxation they describe but never pursue for FFNs. Clean positioning.

4. **Deep TD literature (Zhao et al. 2026 survey) uses TD *on* networks, not *of* networks.** The entire field treats TD as a tool for compressing weights or modeling external data. Viewing the FFN's *computation itself* as a TD is categorically different and appears novel.

5. **CP uniqueness for input-dependent coefficient families is unstudied.** All identifiability results (Kruskal, Bhaskara et al., Chiantini-Ottaviani, Domanov-De Lathauwer) address a single fixed tensor. Your routed-CP generates a *family* A(x) with shared factor matrices — potentially *stronger* identifiability (infinitely many observations of the same factors), but no existing theory covers this. This is a gap worth filling and could be a standalone theoretical contribution.

6. **Transcoders are the closest interpretability analog.** Dunefsky et al. (NeurIPS 2024) and Paulo et al. (2025) independently decompose MLP computation into input-invariant and input-dependent terms. Your routed-CP framework provides the *mathematical basis* for why this factorization works. Strong complementary positioning.

### What remains from the original lit review list
- [x] Search for tensor decomposition views of gated MLPs — **clear, no overlap**
- [x] Search for structured bilinear / multiplicative FFN variants — **Pearce et al. found, positioned against**
- [x] Check Jayakumar et al. (2020) — **diagonal restriction = your same-index coupling, Tucker = their natural relaxation**
- [x] Check ICLR 2025 bilinear MLP paper — **read, complementary not overlapping**
- [x] Survey identifiability/gauge freedom — **4 papers reviewed, gap identified in input-dependent families**
- [x] Survey deep TD literature — **Zhao et al. 2026 survey confirms novelty of your perspective**
- [x] Survey SAEs/transcoders on structured activations — **6 papers reviewed, strong positioning found**
- [ ] **Read Tilde's "Sparsity is Cool" post** — still needed for positioning relative to Tilde specifically
- [ ] **Survey MoE as a competing generalization** — still needed, connects to Tilde's MoMoE work

---

## Phase 1: Tighten the theory

### The diagonal bottleneck question (hardest, highest payoff)

- [ ] **Prove or disprove: is same-index channel coupling an expressivity bottleneck at fixed parameter count?** Given Tucker-core FFN with core C ∈ R^{s×r×r} and SwiGLU with m hidden units, at matched parameter count, can Tucker represent strictly more input-dependent interaction tensors? This should be provable by a rank/dimension argument on the family {A(x) : x ∈ R^d}.
- [ ] **Investigate the depth question.** Can stacking SwiGLU layers simulate what one Tucker layer does? If a single Tucker layer needs depth > 1 in SwiGLU to simulate, that's a separation result. Think about composition of routed CP dictionaries.
- [ ] **Address the output-projection objection.** U already mixes hidden coordinates post-gating. Formalize when post-mixing is vs. isn't sufficient to overcome same-index coupling. This is about whether the *range* of the input-dependent interaction family changes.

### Identifiability and gauge freedom

Lit review covered Bhaskara et al. (robust Kruskal), Chiantini-Ottaviani (generic identifiability up to rank (a+1)(b+1)/16), Domanov-De Lathauwer Parts I & II (overcomplete case). Key gap identified: all results address a *single fixed tensor*. Your setting is different.

- [ ] **State the symmetries explicitly.** CP: scaling + permutation of atoms. Tucker: GL(r) × GL(r) × GL(s) rotations of factor matrices absorbed into the core. Write these down in the note.
- [ ] **Formalize the input-dependent family identifiability question.** A(x) = Σ α_j(x) u_j ⊗ w_j ⊗ g_j generates infinitely many "observations" of the same factor matrices U, W, G with varying coefficients α_j(x). This should be *stronger* than single-tensor identifiability. Can you prove that observing A(x) for sufficiently many x uniquely determines U, W, G up to the standard CP symmetries? This could be a standalone theoretical contribution.
- [ ] **Determine whether identifiability matters for your interpretability claim.** If the Tucker core C is only defined up to rotations in the latent factors, then "which features interact" is basis-dependent. Either:
  - (a) Propose gauge-fixing (orthogonality constraints on P, Q, sparsity penalties on C), or
  - (b) Argue interpretability comes from *structural properties* of C (sparsity pattern, rank structure, spectrum) that are invariant under the gauge group.
- [ ] **Check whether Domanov-De Lathauwer's overcomplete conditions apply.** In SwiGLU, m >> d typically, so U, W, G ∈ R^{d×m} won't have full column rank. Their Khatri-Rao product conditions may still give uniqueness.

---

## Phase 2: Formalize the interpretability hypothesis

### Connect to Tilde's rate-distortion framework
- [ ] **Frame the interpretability claim in information-theoretic terms.** Tucker-core FFNs have more structured interaction geometry → SAE dictionaries achieve better rate-distortion tradeoffs on their activations. Write as a precise conjecture.
- [ ] **Define "cleaner features" operationally.** Candidate metrics: reconstruction loss at matched L_0, feature interpretability scores (auto-interp), feature oversplitting, load balancing across dictionary elements.

### State falsifiable predictions
- [ ] **Prediction 1:** SAEs trained on Tucker-FFN activations achieve lower reconstruction loss at same L_0 vs. SAEs on SwiGLU activations (matched model quality).
- [ ] **Prediction 2:** Learned Tucker core C is empirically sparse or low-rank beyond what parameterization forces.
- [ ] **Prediction 3:** Dictionary features from Tucker-FFN SAEs exhibit less polysemanticity.
- [ ] **Prediction 4 (new):** Applying SAEs to routed channel activations c_j(x) = α_j(x)(w_j^⊤ x)(g_j^⊤ x) yields cleaner features than applying SAEs to raw hidden activations h_j(x).

---

## Phase 3: Experiments

### Experiment 0: Routing behavior in pretrained models (no training needed)

Reframed from the old "gate pre-activation statistics" — now about characterizing routing, not validating an approximation.

- [ ] **Run inference on a pretrained SwiGLU model** (Llama-3.2-1B or Pythia). Collect:
  - Distribution of α_j(x) = σ(g_j^⊤ x) across layers, tokens, hidden units
  - Per-channel variance s_j = Var_x[α_j(x)] — low variance = static bilinear atom, high variance = genuine routing
  - Channel contribution concentration: how many channels account for 90% of Σ_j |c_j(x)| per token?
  - Top-activating tokens per channel — do they cluster semantically?

### Experiment 1: Routing ablation (no training needed)

- [ ] **Replace α_j(x) with constants at inference** in a pretrained model and measure loss increase:
  - α_j = 0.5 (uniform static)
  - α_j = E_x[α_j(x)] (per-channel mean)
  - α_j = layer-specific learned constants
- [ ] This directly measures whether SwiGLU's advantage over bilinear comes from dynamic routing.

### Experiment 2: Small-scale language modeling comparison

- [ ] **Pick a base framework.** NanoGPT or minimal PyTorch loop. Standard transformer + swappable FFN block + logging.
- [ ] **Implement SwiGLU baseline and Tucker-core FFN.** Tucker FFN: project x via P, Q ∈ R^{d×r}, compute gated interaction via core C ∈ R^{s×r×r}, project back via R ∈ R^{d×s}. Make r, s configurable.
- [ ] **Parameter-matched comparisons.** SwiGLU: 3dm params. Tucker: 2dr + ds + sr². Find (r, s) pairs matching 3dm for given d, m.
- [ ] **Train at ~25–50M param scale** on OpenWebText or FineWeb-Edu subset. Compare validation loss curves. This isn't the main result, but Tucker can't hurt or the interpretability story isn't credible.

### Experiment 3: Interaction structure analysis

- [ ] **Extract learned core C** from trained Tucker models.
- [ ] **Analyze structure:** sparsity, rank, deviation from diagonal. For small r, plot C_{α,:,:} as heatmaps.
- [ ] **Key question:** does C stay near-diagonal (SwiGLU's bias was already right) or go substantially off-diagonal (same-index coupling was too restrictive)?

### Experiment 4: SAE comparison (the interpretability payoff)

- [ ] **Train SAEs on hidden activations** of both SwiGLU and Tucker-core models (SAELens or dictionary_learning).
- [ ] **Compare:** reconstruction loss vs. L_0 Pareto frontier, feature frequency distribution, qualitative feature interpretability.
- [ ] **Also try SAEs on channel activations c_j(x)** rather than raw hidden activations — test whether the routed-channel decomposition gives SAEs a cleaner object.

---

## Phase 4: Write-up

### Structure
- **Section 1:** Exact routed-CP analysis of SwiGLU. The tensor framework, propositions, definitions, same-index coupling as structural constraint.
- **Section 2:** Tucker generalization. Recovery of SwiGLU, expressivity argument, parameter-count tradeoffs.
- **Section 3:** Interpretability hypothesis. SAEs, rate-distortion connection, falsifiable predictions, connection to Tilde's work.
- **Section 4:** Experimental design + preliminary results.

### Things to explicitly address
- [ ] MoE as an alternative generalization of diagonal routing
- [ ] Identifiability / gauge freedom and its implications for interpretability
- [ ] Why Tucker specifically (vs. higher-rank CP, vs. generic bilinear, vs. hypernetworks)
- [ ] Computational cost analysis (FLOPs/memory for Tucker contraction vs. elementwise gating)
- [ ] **Relationship to Pearce et al. bilinear MLP work (ICLR 2025)** — position as generalization from static bilinear to routed-gated case
- [ ] **Relationship to transcoders (Dunefsky et al., Paulo et al.)** — your framework provides the mathematical basis for their empirical factorization into input-dependent and input-invariant terms
- [ ] **Relationship to Jayakumar et al. multiplicative interaction taxonomy** — your same-index coupling = their diagonal restriction; Tucker = their natural non-diagonal relaxation

---

## Priority ordering

1. ~~**Lit review**~~ — **DONE.** No existence threat. Clear positioning identified.
2. **Remaining lit review items** — Read Tilde's "Sparsity is Cool" post + survey MoE as competing generalization
3. **Routing statistics + ablation** (Experiments 0–1) — cheap, no training, directly characterizes the routed-CP picture
4. **Tighten theory** (Phase 1) — diagonal bottleneck + identifiability (especially the input-dependent family question) are the core theoretical contributions
5. **Formalize interpretability hypothesis** (Phase 2)
6. **Build training infrastructure** (Phase 3)
7. **Run experiments 2–4** as time/compute allows
8. **Write-up** (Phase 4)
