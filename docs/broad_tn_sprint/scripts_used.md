# Exact commands run (sprint 2)

```bash
# unit tests
.venv/bin/python lib/structured_linear.py
.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from lib.structured_ffn import _tests; _tests()"
.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from lib.sparsity import _test; _test()"
.venv/bin/python -c "import sys; sys.path.insert(0,'.'); import lib.interp_metrics as im; im._tests()"
# tied-gate equivalence: inline check (LL1(B,L) == grouped-gate SwiGLU(BL), 0 err)

# Exp A: route/atom factorial (GPU0) — 100M tokens each
exp11 --archs swiglu --swiglu_m 1992 --seeds 0,1 --results_dir results/s2_lm/atom_matched
exp11 --archs swiglu --swiglu_m 498  --seeds 0,1 --results_dir results/s2_lm/route_matched

# Confound C: tucker fairness probes (GPU1) — 30M tokens each
exp11 --archs tucker --seeds 0 --tucker_diagonal_bias_init --tucker_core_lr_scale 2.0 --results_dir results/s2_lm/tucker_clr2
exp11 --archs tucker --seeds 0 --tucker_diagonal_bias_init --tucker_core_lr_scale 0.5 --results_dir results/s2_lm/tucker_clr05
exp11 --archs tucker --seeds 0 --results_dir results/s2_lm/tucker_noWS

# Exp B: structured distillation + throughput
.venv/bin/python experiments/exp23_structured_distill.py --archs swiglu,swiglu_lowrank,swiglu_blockdiag4,swiglu_monarch4,ll1_l4,ll1_l4_monarch4,ll1_l4_blockdiag4 --layers 4,12 --budgets 600000,1200000 --results_dir results/exp23b
# butterfly (caveated, 1000 steps): /tmp/butterfly_run.py -> results/exp23_butterfly.json
exp11 --archs struct_monarch4 --seeds 0 --results_dir results/s2_lm/monarch   # 100M tok
# idle-GPU bench: /tmp/bench_struct.py -> results/s2_throughput.json

# Exp C: trained sparsity — 100M tokens each
exp11 --archs swiglu --seeds 0 --sparsity route_l1   --sparsity_lambda 1e-3 --results_dir results/s2_lm/sparse_swiglu
exp11 --archs ll1_l4 --seeds 0 --sparsity group_lasso --sparsity_lambda 1e-3 --results_dir results/s2_lm/sparse_ll1
exp11 --archs swiglu --seeds 0 --sparsity route_l1   --sparsity_lambda 0.03 --results_dir results/s2_lm/sparse_swiglu_l03
exp11 --archs swiglu --seeds 0 --sparsity route_l1   --sparsity_lambda 0.3  --results_dir results/s2_lm/sparse_swiglu_l3
exp11 --archs swiglu --seeds 0 --sparsity contrib_l1 --sparsity_lambda 0.03 --results_dir results/s2_lm/sparse_contrib03  # gauge-invariant
# eff-active post-hoc: exp19_interp_proxies.py on each sparse ckpt -> results/exp19_sparse*

# Exp D: superposition recovery (swap-aware scoring, 3 seeds)
.venv/bin/python experiments/exp24_superposition.py --results_dir results/exp24                      # K=96, p=0.2
.venv/bin/python experiments/exp24_superposition.py --K 48 --results_dir results/exp24_easy          # separable
.venv/bin/python experiments/exp24_superposition.py --K 48 --p_active 0.08 --results_dir results/exp24_sparse

# Exp E + confound B
.venv/bin/python experiments/exp25_spectra.py --results_dir results/exp25     # + exp25b on s2 probes
.venv/bin/python experiments/exp26_qwen_contexts.py --results_dir results/exp26

# (exp11 = .venv/bin/python experiments/exp11_train_lm.py --max_tokens 100000000 unless noted)
```
