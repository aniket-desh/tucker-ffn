# Exact commands run (sprint)

```bash
# unit tests
.venv/bin/python lib/tucker_ffn.py
.venv/bin/python lib/ll1_ffn.py

# smoke tests
.venv/bin/python experiments/exp11_train_lm.py --archs ll1_l4,swiglu --seeds 0 \
  --d 128 --n_heads 4 --n_layers 2 --seq_len 256 --batch_size 8 --max_tokens 300000 \
  --eval_every_tokens 100000 --n_val_seqs 8 --warmup_steps 10 --tucker_r 32 \
  --tucker_s 32 --results_dir results/smoke_ll1
.venv/bin/python experiments/exp18_ll1_synthetic.py --steps 300 --n_seeds 1 \
  --n_train 5000 --n_val 1000 --teachers cp --lsweep_archs swiglu,ll1_l2,ll1_l4,tucker \
  --budget_scales 1 --results_dir results/smoke_exp18
.venv/bin/python experiments/exp20_induction.py --steps 200 --eval_every 100 \
  --kinds swiglu,ll1_l4,tucker,none --seeds 0 --results_dir results/smoke_exp20

# main LM runs (launched T+0:35, 100M FineWeb-Edu tokens each, ~52.5M params)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/exp11_train_lm.py \
  --archs swiglu,ll1_l4 --seeds 0,1,2 --max_tokens 100000000 \
  --results_dir results/sprint_lm                      # > logs/lm_gpu0.log
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp11_train_lm.py \
  --archs tucker --seeds 0,1,2 --tucker_diagonal_bias_init --tucker_diag_bias_eps 1e-2 \
  --max_tokens 100000000 --results_dir results/sprint_lm \
&& CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp11_train_lm.py \
  --archs ll1_l1,ll1_l2,ll1_l8,ll1_l16 --seeds 0 --max_tokens 100000000 \
  --results_dir results/sprint_lm                      # > logs/lm_gpu1.log

# synthetic sweep (launched T+0:40, GPU0 shared)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/exp18_ll1_synthetic.py \
  --results_dir results/exp18                          # > logs/exp18.log

# induction pilot (launched T+1:05, GPU1 shared)
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp20_induction.py \
  --vocab 128 --half_len 64 --steps 3000 --eval_every 20 \
  --kinds none,swiglu,ll1_l1,ll1_l4,ll1_l16,tucker --seeds 0,1,2 \
  --results_dir results/exp20                          # > logs/exp20.log
```

(exp19 interp proxies + scripts/sprint_throughput.py: commands appended when run.)

```bash
# analyses on trained checkpoints
.venv/bin/python experiments/exp19_interp_proxies.py --ckpts <8 headline ckpts> \
  --results_dir results/exp19      # + results/exp19b for tucker_seed2 & L-sweep ckpts
.venv/bin/python experiments/exp22_factor_stability.py --ckpt_pairs \
  "<swiglu 0,1>;<swiglu 1,2>;<ll1_l4 0,1>;<ll1_l4 1,2>;<tucker 0,1>" \
  --results_dir results/exp22      # + exp22b for tucker 1,2
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/sprint_throughput.py   # idle GPU

# rebalanced final LM runs (after killing GPU1 queue at T+3:05)
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp11_train_lm.py --archs tucker \
  --seeds 2 --tucker_diagonal_bias_init --max_tokens 100000000 --results_dir results/sprint_lm
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp11_train_lm.py --archs ll1_l2,ll1_l16 \
  --seeds 0 --max_tokens 100000000 --results_dir results/sprint_lm
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/exp11_train_lm.py --archs ll1_l1,ll1_l8 \
  --seeds 0 --max_tokens 100000000 --results_dir results/sprint_lm

# mechanism probe + figures
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp20b_tucker_mechanism.py
.venv/bin/python scripts/make_lm_summary.py
.venv/bin/python scripts/make_interp_figures.py
.venv/bin/python scripts/make_fig1_diagram.py

# paper
~/.local/bin/tectonic paper/main.tex
```
