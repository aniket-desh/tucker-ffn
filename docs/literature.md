# Literature Review: SwiGLU as a Low-Rank Tensor Model

Papers organized by research question. One-sentence relevance assessment for each.

---

## (3) Identifiability and Gauge Freedom in CP/Tucker Decompositions

### Bhaskara, Charikar, Vijayaraghavan (2014). "Uniqueness of Tensor Decompositions with Applications to Polynomial Identifiability." JMLR W&CP vol 35, pp. 1–37.
- **Link:** https://proceedings.mlr.press/v35/bhaskara14a.pdf
- **Relevance:** Provides a **robust** version of Kruskal's uniqueness theorem — if a 3-tensor has a bounded rank-R decomposition satisfying k_A + k_B + k_C ≥ 2R + 2, then the decomposition is approximately recoverable from a tensor known only up to inverse-polynomial error, which is directly applicable to asking whether the routed-CP decomposition of SwiGLU (with factor matrices U, W, G) is essentially unique under realistic perturbations.

### Chiantini & Ottaviani (2012). "On Generic Identifiability of 3-Tensors of Small Rank." SIAM J. Matrix Anal. Appl. 33(3), pp. 1018–1037.
- **Link:** https://epubs.siam.org/doi/10.1137/110829180 (arXiv: https://arxiv.org/abs/1103.2696)
- **Relevance:** Uses algebraic geometry (weak defectivity) to prove that generic 3-tensors of type (a, b, c) have unique CP decomposition for rank k ≤ (a+1)(b+1)/16, substantially extending Kruskal's bound — this gives a much larger regime of generic identifiability that could apply to the d × d × d interaction tensor A(x) when the CP rank m is moderate relative to d.

### Domanov & De Lathauwer (2013). "On the Uniqueness of the Canonical Polyadic Decomposition of Third-Order Tensors — Part II: Uniqueness of the Overall Decomposition." SIAM J. Matrix Anal. Appl. 34(3), pp. 876–903.
- **Link:** https://epubs.siam.org/doi/10.1137/120877258 (arXiv: https://arxiv.org/abs/1301.4603)
- **Relevance:** Establishes CPD uniqueness conditions using Khatri-Rao products of compound matrices, covering cases where **none** of the factor matrices has full column rank — this is critical because in the SwiGLU decomposition A(x) = Σ α_j(x) u_j ⊗ w_j ⊗ g_j, the factor matrices U, W, G ∈ R^{d×m} with m >> d will generically not have full column rank.

### Domanov & De Lathauwer (2015). "Generic Uniqueness Conditions for the Canonical Polyadic Decomposition and INDSCAL." SIAM J. Matrix Anal. Appl. 36(4), pp. 1567–1589.
- **Link:** https://epubs.siam.org/doi/abs/10.1137/140970276
- **Relevance:** Finds conditions guaranteeing that a **generic** third-order tensor's CPD is unique up to permutation — directly relevant to determining whether, for a generic choice of SwiGLU weights (U, W, G), the rank-1 atoms u_j ⊗ w_j ⊗ g_j are identifiable from the family of interaction tensors A(x).

### Key takeaway for the project:
The standard CP uniqueness results (Kruskal, Chiantini-Ottaviani, Domanov-De Lathauwer) apply to a **fixed** tensor. Your setting is different: A(x) is a **family** of tensors parameterized by x, with shared factor matrices but input-dependent coefficients α_j(x). This is potentially **stronger** than single-tensor identifiability (you have infinitely many "observations" of the same factor matrices), but none of these papers addresses this case directly. This is a gap you should note and possibly fill.

---

## (5) Expressivity of Bilinear / Quadratic / Gated Layers

### Tensor Decomposition for Model Reduction in Neural Networks: A Review. IEEE Signal Processing Magazine, 2023.
- **Link:** https://ieeexplore.ieee.org/document/10190238/
- **Relevance:** Reviews six tensor decomposition methods (CP, Tucker, Tensor Train, etc.) for compressing CNNs, RNNs, and Transformers — useful as background for understanding how Tucker has been used for neural network **compression** (replacing weight matrices with factored forms), which is the existing use case you need to distinguish your **architectural design** use of Tucker from.

### Helal (2023). "Enhancing Deep Learning Models through Tensorization: A Comprehensive Survey and Framework." arXiv:2309.02428.
- **Link:** https://arxiv.org/abs/2309.02428
- **Relevance:** Surveys tensorization approaches that bridge multidimensional data representation with deep learning — provides broad context on how tensor methods have been integrated into neural network layers, but does not address gated MLPs or the specific CP/Tucker structure of SwiGLU-type interactions.

### Fan, Li, Wang, Lai, Wang (2023). "On Expressivity and Trainability of Quadratic Networks." IEEE Trans. Neural Netw. Learn. Syst. 36(1), pp. 1228–1242.
- **Link:** https://ieeexplore.ieee.org/document/10327752/ (arXiv: https://arxiv.org/abs/2110.06081)
- **Relevance:** Proves (via spline theory and algebraic geometry) that quadratic neurons — which replace the inner product with a quadratic function — have strictly greater expressivity than conventional neurons, and proposes ReLinear initialization to stabilize training — directly relevant to the diagonal bottleneck question, since SwiGLU's bilinear interaction is a **constrained** quadratic form and this paper quantifies what unconstrained quadratic buys you.

---

## (8) SAEs / Transcoders on Structured Activations

### Park, Kim, Lee (2024). "Discrete Dictionary-based Decomposition Layer for Structured Representation Learning." NeurIPS 2024.
- **Link:** https://proceedings.neurips.cc/paper_files/paper/2024/hash/259762417183b58aa5bb842c1e502076-Abstract-Conference.html
- **Relevance:** Proposes a discrete learnable dictionary layer (D3) for Tensor Product Representation models that improves systematic generalization by mapping inputs to pre-learned symbolic features — conceptually parallel to your idea that making the interaction dictionary more explicit (via Tucker core) could give SAEs a cleaner decomposition target.

### Qiu et al. (2025). "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free." arXiv:2505.06708.
- **Link:** https://arxiv.org/abs/2505.06708
- **Relevance:** Systematically studies gating mechanisms applied to softmax attention (sigmoid gate after SDPA), finding that gating introduces beneficial non-linearity and query-dependent sparsity — relevant because it demonstrates that the input-dependent routing/gating mechanism (analogous to α_j(x) in SwiGLU) provides measurable expressivity and sparsity benefits, supporting the claim that dynamic routing does real work beyond what static bilinear interaction provides.

### Ge, Zhu, Shu, Wang, He, Qiu (2024). "Automatically Identifying Local and Global Circuits with Linear Computation Graphs." arXiv:2405.13868.
- **Link:** https://arxiv.org/abs/2405.13868
- **Relevance:** Inserts SAEs and transcoders into transformers to make the computation graph strictly linear for circuit discovery, enabling fine-grained identification of both local and global circuits — directly relevant to the interpretability hypothesis, as it demonstrates that **replacing MLP nonlinearities with structured sparse decompositions** (transcoders) enables cleaner mechanistic analysis, which is analogous to what Tucker-core FFNs might provide.

### Dunefsky, Chlenski, Nanda (2024). "Transcoders Find Interpretable LLM Feature Circuits." NeurIPS 2024.
- **Link:** https://arxiv.org/abs/2406.11944
- **Relevance:** Shows that transcoders — wide, sparsely-activating MLPs trained to approximate dense MLP sublayer behavior — yield circuit attributions that cleanly **factorize into input-dependent and input-invariant terms**, achieving comparable interpretability to SAEs while enabling weights-based circuit analysis; this factorization mirrors your exact decomposition of SwiGLU into fixed rank-1 atoms (input-invariant) and routing coefficients α_j(x) (input-dependent), suggesting transcoders and your routed-CP framework are addressing the same structural insight from different angles.

### (Also relevant) OpenReview forum 89wVrywsIy — Ge et al. "Automatically Identifying and Interpreting Sparse Circuits with Hierarchical Tracing."
- **Link:** https://openreview.net/forum?id=89wVrywsIy
- **Relevance:** Applies hierarchical tracing with SAEs and transcoders for Transformer circuit analysis at both local and global levels — reinforces the approach of using structured sparse decompositions for interpretability of MLP computations.

### Paulo, Mallen, Juang (2025). "Transcoders Beat Sparse Autoencoders for Interpretability." arXiv:2501.18823.
- **Link:** https://arxiv.org/abs/2501.18823
- **Relevance:** Demonstrates that transcoder features are significantly more interpretable than SAE features on the same models, and proposes skip transcoders with affine skip connections for improved reconstruction — suggests that decomposing MLP **computation** (input→output) rather than just **activations** yields better interpretability, supporting your hypothesis that analyzing the routed interaction channels c_j(x) = α_j(x)(w_j^⊤ x)(g_j^⊤ x) may be more fruitful than analyzing raw hidden activations.

---

---

## (1) Existence Threats — Has anyone already cast gated MLPs as tensor decompositions?

### Pearce, Dooms, Rigg (2024). "Weight-based Decomposition: A Case for Bilinear MLPs." arXiv:2406.03947.
- **Link:** https://arxiv.org/abs/2406.03947
- **Relevance:** **Closest existing work.** Shows that bilinear MLPs (GLUs without the nonlinearity, g(x) = (Wx) ⊙ (Vx)) can be fully expressed as a third-order tensor and linear operations, and decomposes this tensor via eigendecomposition into sparsely interacting eigenvectors with interpretable properties — this is essentially the α_j(x) ≡ 1 special case of your routed-CP framework, and they explicitly note the connection to tensor decomposition for interpretability. **Critical difference:** they analyze the *bilinear* case (no sigmoid gating), not SwiGLU's input-dependent routed structure. Your contribution is showing that the *full* SwiGLU (with sigmoid routing) admits an exact input-dependent CP decomposition, which is strictly richer.

### Pearce, Dooms, Rigg (2024). "Bilinear MLPs enable weight-based mechanistic interpretability." ICLR 2025 (arXiv:2410.08417).
- **Link:** https://arxiv.org/abs/2410.08417
- **Relevance:** **Extended version of the above, accepted at ICLR 2025.** Demonstrates that the third-order bilinear tensor exhibits interpretable low-rank structure across image classification and language modeling, and that eigendecomposition of weight slices reveals circuits (e.g., a sentiment-negation circuit) directly from weights alone — this is the paper referenced in your todo list as the ICLR 2025 bilinear MLP interpretability paper. **Key positioning point:** they work with bilinear (no gating nonlinearity) and use eigendecomposition of weight matrices. Your framework handles the full SwiGLU case with input-dependent routing, and uses CP/Tucker decomposition rather than eigendecomposition. The two approaches are complementary: theirs is weight-only analysis of a simplified architecture; yours is an exact structural analysis of the architecture actually deployed in production models.

### Jayakumar, Menick, Czarnecki, Schwarz, Rae, Osindero, Teh, Harley, Pascanu (2020). "Multiplicative Interactions and Where to Find Them." ICLR 2020.
- **Link:** https://openreview.net/forum?id=rylnK6VtDH
- **Relevance:** Provides the unifying framework for multiplicative interactions in neural networks (gating, attention, hypernetworks, dynamic convolutions) that you already cite. Shows multiplicative interaction layers strictly enrich representable function classes and conjectures they offer a powerful inductive bias for conditional computation. **Key for your project:** their taxonomy describes *diagonal* multiplicative interactions (gating = Hadamard product with context-dependent vector) as a special case of a general 3D-tensor multiplicative interaction y = z^T W x. Your same-index coupling constraint in SwiGLU is exactly this diagonal restriction, and your Tucker generalization is the natural relaxation they describe but don't pursue for FFN blocks specifically.

### Inayatullah & Shafiq (2025). "Element-Wise Multiplicative Operators in Vision, Language, and Multimodal Learning." Preprints.org, doi:10.20944/preprints202505.1290.v1.
- **Link:** https://www.preprints.org/frontend/manuscript/57267329ef38e7720e15cb97ca563f81/download_pub
- **Relevance:** Comprehensive survey of the Hadamard/Schur product as a computational primitive across deep learning — covers GLU/SwiGLU gating, attention mechanisms, LoRA, FiLM conditioning, and proposes "Feature-Aligned Multiplicative Conditioning" (FAMC) as a meta-architecture pattern. Useful as broad context but does not formalize the tensor decomposition structure or connect to CP/Tucker. Not peer-reviewed.

---

## (6) Deep Tensor Decomposition (general survey)

### Zhao, Hu, Li, Wang, Sun (2026). "Deep Tensor Decomposition: A Survey." Neurocomputing 664, 132074.
- **Link:** https://www.sciencedirect.com/science/article/pii/S0925231225027468
- **Relevance:** Comprehensive survey of deep TD methods, covering both linear deep TD (hierarchical CP/Tucker where factor matrices are recursively decomposed across layers) and nonlinear deep TD (integrating MLPs/CNNs with tensor decomposition for data completion and prediction tasks). **Key distinction for your project:** all surveyed methods use tensor decomposition either for *compressing* neural network weights or for *modeling external data* (e.g., knowledge graphs, traffic tensors). None of them analyze the *computation performed by a neural network layer* as itself being a tensor decomposition — which is your central contribution. Your work is categorically different: you're not using TD to compress or complete data, you're showing that SwiGLU's computation *is* a TD.

---

## Summary of Gaps and Opportunities

1. **The bilinear MLP tensor decomposition exists, but your exact routed-CP formulation does not.** Pearce et al. (2024, ICLR 2025) show that *bilinear* MLPs (no sigmoid gating) can be expressed as a third-order tensor — but they analyze the static bilinear case only. Your key contribution is showing that *SwiGLU with its sigmoid routing* admits an exact *input-dependent* CP decomposition, which is structurally richer. You should cite Pearce et al. prominently and position your work as the generalization to the gated case that production models actually use.

2. **CP uniqueness for input-dependent coefficient families is unstudied.** All existing identifiability results (Kruskal, Bhaskara et al., Chiantini-Ottaviani, Domanov-De Lathauwer) address a single fixed tensor. Your routed-CP decomposition generates a *family* A(x) sharing factor matrices across all x. This may admit stronger identifiability results and is worth formalizing.

3. **Transcoders are the closest interpretability analog.** The transcoder literature (Dunefsky et al., Paulo et al.) independently arrives at the same structural insight: decomposing MLP computation into sparse, interpretable channels with input-dependent and input-invariant terms. Your tensor-theoretic framework provides a *principled mathematical basis* for why this factorization works, which the transcoder papers lack.

4. **Jayakumar et al. describe the general multiplicative interaction taxonomy, including the diagonal special case.** Your same-index coupling constraint is exactly their diagonal restriction. Your Tucker generalization is the natural non-diagonal relaxation they describe but never pursue for FFN blocks. This is clean positioning.

5. **Deep tensor decomposition literature uses TD *on* networks, not *of* networks.** The entire deep TD field (Zhao et al. 2026 survey) treats TD as a tool applied to data or to compress weights. Viewing the FFN's *computation itself* as a tensor decomposition is a fundamentally different perspective that appears novel.

6. **Quadratic expressivity results exist but don't address gated/routed structure.** Fan et al. prove quadratic networks are more expressive than linear ones, but don't analyze the specific constraint of same-index diagonal coupling vs. cross-channel interaction. Your diagonal bottleneck question is open.
