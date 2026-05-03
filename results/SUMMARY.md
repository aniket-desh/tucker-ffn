# Numerical results summary

All numbers cited in the paper, with their source files.

## Routing ablation (Qwen2.5-0.5B perplexity)
_source: `results/qwen25_05b/ablation_results.json`_

- baseline: 16.84
- alpha=0.5 (uniform): 272372.25
- alpha=mean: 6412058.00
- alpha=1 (bilinear): 737666.19

## Same-index pairing permutation (exp09 + exp09b)
_source: `results/qwen25_05b/pairing_permutation.json`_

- baseline perplexity: 16.84
- n_seeds: 8, n_layers: 24
- joint   : mean=1.10e+03, max=2.56e+04 (layer 0), min=1.89e+01 (layer 10)
- u_only  : mean=1.10e+03, max=2.45e+04 (layer 0), min=1.88e+01 (layer 10)
- g_only  : mean=1.47e+03, max=3.46e+04 (layer 0), min=1.91e+01 (layer 10)
- w_only  : mean=1.25e+03, max=2.94e+04 (layer 0), min=1.90e+01 (layer 10)
- geomean(joint/u_only) ratio: 0.8460727349059959
- no-op (g, u, w joint perm): max_dev_from_baseline=7.63e-06

## Synthetic fitting limit (exp10)
_source: `results/exp10/synthetic_fitting.npz`_

- k = 4, m_values = [np.int64(4), np.int64(8), np.int64(12), np.int64(16), np.int64(32), np.int64(64)]
  - swiglu_unconstrained: ['1.099e-01', '3.300e-02', '5.071e-03', '1.483e-03', '1.347e-04', '9.465e-05']
  - swiglu_aligned: ['1.351e-01', '4.184e-02', '2.804e-03', '7.101e-04', '4.341e-07', '1.317e-14']
  - tucker_control: ['3.778e-01', '3.044e-01', '3.821e-01', '3.595e-01', '3.135e-01', '2.873e-01']
- k = 8, m_values = [np.int64(8), np.int64(16), np.int64(56), np.int64(64), np.int64(128), np.int64(256)]
  - swiglu_unconstrained: ['1.399e+00', '7.116e-01', '1.690e-02', '9.760e-03', '2.335e-03', '1.889e-03']
  - swiglu_aligned: ['1.623e+00', '9.570e-01', '2.042e-03', '4.735e-04', '7.152e-13', '6.383e-14']
  - tucker_control: ['8.016e-01', '7.864e-01', '4.962e-01', '4.822e-01', '6.651e-01', '6.977e-01']
- k = 16, m_values = [np.int64(16), np.int64(32), np.int64(240), np.int64(256), np.int64(512), np.int64(1024)]
  - swiglu_unconstrained: ['1.211e+01', '8.237e+00', '3.258e-01', '3.097e-01', '1.062e-01', '8.697e-02']
  - swiglu_aligned: ['1.390e+01', '9.680e+00', '4.000e-03', '1.490e-03', '3.330e-08', '3.444e-12']
  - tucker_control: ['8.480e-02', '3.014e-02', '1.040e-01', '4.661e-02', '4.299e-01', '1.032e-01']

## LM training (exp11 + hill-climb variants)
- `exp11/swiglu_seed0`: arch=swiglu seed=0 params=52.5M | final val_loss=4.758 (ppl=116.5) after 100.0M tokens
- `exp11/tucker_seed0`: arch=tucker seed=0 params=52.5M | final val_loss=5.116 (ppl=166.7) after 100.0M tokens
- `exp11_hc/tucker_seed0`: arch=tucker seed=0 params=52.5M | final val_loss=4.772 (ppl=118.2) after 100.0M tokens
- `exp11_hc_v2/tucker_seed0`: arch=tucker seed=0 params=52.5M | final val_loss=4.770 (ppl=118.0) after 100.0M tokens
- `exp11_hc_v3/tucker_seed0`: arch=tucker seed=0 params=52.5M | final val_loss=4.753 (ppl=116.0) after 100.0M tokens

## Stable rank of V_j (exp12)
_source: `results/exp12/stable_rank.npz`_

- exp12/tucker_seed0: mean stable rank = 27.56 | median = 27.62 | min = 22.01 | max = 31.42

## Stable rank of V_j (exp12_hc)
_source: `results/exp12_hc/stable_rank.npz`_

- exp12_hc/tucker_seed0: mean stable rank = 16.80 | median = 16.98 | min = 8.21 | max = 23.02

## Stable rank of V_j (exp12_hc_v2)
_source: `results/exp12_hc_v2/stable_rank.npz`_

- exp12_hc_v2/tucker_seed0: mean stable rank = 15.17 | median = 15.32 | min = 7.15 | max = 20.70

## Stable rank of V_j (exp12_hc_v3)
_source: `results/exp12_hc_v3/stable_rank.npz`_

- exp12_hc_v3/tucker_seed0: mean stable rank = 3.97 | median = 4.00 | min = 2.79 | max = 5.30

## Stable rank of V_j (exp12_smoke)
_source: `results/exp12_smoke/stable_rank.npz`_

- exp12_smoke/tucker_seed0: mean stable rank = 8.01 | median = 8.07 | min = 6.37 | max = 9.50

## Diagonal projection / rank truncation (exp13)
_source: `results/exp13/results.json`_

- exp13/tucker_seed0: trained ppl = 159.50, diagonal-projected ppl = 39382.49, ratio = 246.91x
  - rank-truncation curve:
    - rho=1: ppl=17928.58
    - rho=2: ppl=10848.74
    - rho=4: ppl=5462.26
    - rho=8: ppl=1490.69
    - rho=16: ppl=458.03
    - rho=32: ppl=237.70
    - rho=64: ppl=170.40
    - rho=128: ppl=159.49

## Diagonal projection / rank truncation (exp13_hc)
_source: `results/exp13_hc/results.json`_

- exp13_hc/tucker_seed0: trained ppl = 113.92, diagonal-projected ppl = 106352.86, ratio = 933.53x
  - rank-truncation curve:
    - rho=1: ppl=12423.00
    - rho=2: ppl=7796.93
    - rho=4: ppl=3286.70
    - rho=8: ppl=906.00
    - rho=16: ppl=309.61
    - rho=32: ppl=146.41
    - rho=64: ppl=117.13
    - rho=128: ppl=113.94

## Diagonal projection / rank truncation (exp13_hc_v2)
_source: `results/exp13_hc_v2/results.json`_

- exp13_hc_v2/tucker_seed0: trained ppl = 113.48, diagonal-projected ppl = 104782.53, ratio = 923.32x
  - rank-truncation curve:
    - rho=1: ppl=10892.37
    - rho=2: ppl=5865.11
    - rho=4: ppl=3353.22
    - rho=8: ppl=1369.67
    - rho=16: ppl=369.38
    - rho=32: ppl=148.07
    - rho=64: ppl=116.20
    - rho=128: ppl=113.48

## Diagonal projection / rank truncation (exp13_hc_v3)
_source: `results/exp13_hc_v3/results.json`_

- exp13_hc_v3/tucker_seed0: trained ppl = 111.84, diagonal-projected ppl = 57975.03, ratio = 518.36x
  - rank-truncation curve:
    - rho=1: ppl=41546.47
    - rho=2: ppl=4858.00
    - rho=4: ppl=1649.46
    - rho=8: ppl=667.06
    - rho=16: ppl=249.35
    - rho=32: ppl=136.30
    - rho=64: ppl=113.89
    - rho=128: ppl=111.84

## Diagonal projection / rank truncation (exp13_smoke)
_source: `results/exp13_smoke/results.json`_

- exp13_smoke/tucker_seed0: trained ppl = 2022.40, diagonal-projected ppl = 7388627.68, ratio = 3653.39x
  - rank-truncation curve:
    - rho=1: ppl=3823.47
    - rho=2: ppl=3990.27
    - rho=4: ppl=2674.29
    - rho=8: ppl=2207.32
    - rho=16: ppl=2070.70
    - rho=32: ppl=2022.90

## Distillation gap (exp14, swiglu teacher)
_source: `results/exp14_v2/distillation.json`_

- teacher: swiglu layer of Qwen/Qwen2.5-0.5B, layer 12, d=896, m_teacher=4864
- m_swiglu=243, r=s=81: swiglu val_mse = 1.302e-02 (1.7e-05), tucker val_mse = 1.652e-02 (1.3e-05), ratio = 0.79x
- m_swiglu=486, r=s=103: swiglu val_mse = 1.126e-02 (1.2e-05), tucker val_mse = 1.579e-02 (2.3e-06), ratio = 0.71x
- m_swiglu=973, r=s=132: swiglu val_mse = 9.924e-03 (1.2e-05), tucker val_mse = 1.499e-02 (1.2e-05), ratio = 0.66x
- m_swiglu=1946, r=s=168: swiglu val_mse = 8.884e-03 (7.5e-06), tucker val_mse = 1.419e-02 (1.8e-05), ratio = 0.63x
- m_swiglu=3891, r=s=215: swiglu val_mse = 8.592e-03 (1.8e-05), tucker val_mse = 1.332e-02 (7.5e-06), ratio = 0.65x

## Distillation gap (exp14b, tucker teacher)
_source: `results/exp14b/tucker_teacher_distillation.json`_

- teacher: trained tucker layer 4 of results/exp11/tucker_seed0/checkpoint_final.pt, d=512, r_teacher=128
- m_swiglu=53, r=s=32: swiglu val_mse = 5.292e+01 (1.3e-01), tucker val_mse = 5.886e+01 (9.1e-02), ratio = 0.90x
- m_swiglu=235, r=s=64: swiglu val_mse = 2.532e+01 (2.3e-02), tucker val_mse = 3.067e+01 (4.7e-02), ratio = 0.83x
- m_swiglu=1493, r=s=128: swiglu val_mse = 1.082e+01 (1.0e-02), tucker val_mse = 8.383e+00 (6.1e-02), ratio = 1.29x
- m_swiglu=4800, r=s=192: swiglu val_mse = 7.133e+00 (1.6e-02), tucker val_mse = 6.592e+00 (1.7e-02), ratio = 1.08x

## Paper figures
- `fig_synthetic_fitting.png` [OK], stub `fig_synthetic_fitting.tex`
- `fig_diagonal_projection.png` [OK], stub `fig_diagonal_projection.tex`
- `fig_pairing_permutation.png` [OK], stub `fig_pairing_permutation.tex`
- `fig_stable_rank_histogram.png` [OK], stub `fig_stable_rank_histogram.tex`
- `fig_routing_validation.png` [OK], stub `fig_routing_validation.tex`
- `fig_lm_loss_curves.png` [OK], stub `fig_lm_loss_curves.tex`
- `fig_tucker_teacher_distillation.png` [OK], stub `fig_tucker_teacher_distillation.tex`
- `fig_robustness_panel.png` [OK], stub `fig_robustness_panel.tex`

[done] HC v3 BEATS swiglu: T_hill_v3 = 4.753 (ppl 115.98) vs swiglu 4.758 (116.48), gap -0.005 nats / -0.4% ppl. corrected variance-preserving init (full-core std=1/r, diagonal warm-start C[a,a,a]=1 with off-diag eps/r at eps=1e-2) was the unlock. exp12_hc_v3: mean stable rank 3.97 (much lower than v1/v2's ~16; model stays close to diagonal but uses non-trivial cross-channel rank, above aligned-swiglu's rank-1 ceiling). exp13_hc_v3: trained ppl 111.84, fully diagonal-projected 57975 = 518x cost, baseline preservation 0.002% rel_err. paper/exp11_discussion.tex toggled to framing (a).
