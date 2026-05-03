#!/usr/bin/env python3
"""Walk training/eval scripts to extract argparse defaults and dump a clean
LaTeX subsection. Output: snippets/repro_appendix.tex (paste-ready)."""
import sys, pathlib, ast


def extract_defaults(script_path):
    tree = ast.parse(pathlib.Path(script_path).read_text())
    defaults = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "add_argument"):
            name = None
            for arg in node.args:
                if (isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and arg.value.startswith("--")):
                    name = arg.value[2:]
                    break
            if name is None:
                continue
            for kw in node.keywords:
                if kw.arg == "default":
                    if isinstance(kw.value, ast.Constant):
                        defaults[name] = kw.value.value
                    elif isinstance(kw.value, ast.UnaryOp):
                        defaults[name] = ast.unparse(kw.value)
    return defaults


lm = extract_defaults("experiments/exp11_train_lm.py")
syn = extract_defaults("experiments/exp10_synthetic_fitting.py")
dist = extract_defaults("experiments/exp14b_tucker_teacher_distillation.py")

batch_tokens = lm.get("batch_size", 16) * lm.get("seq_len", 1024)

body = rf"""\paragraph{{Tucker LM and SwiGLU LM training.}}
We train both architectures from scratch on \textsc{{FineWeb-Edu}}
\texttt{{sample-10BT}} (HuggingFace
\texttt{{HuggingFaceFW/fineweb-edu}}, version pinned to the dataset hash on
record at the time of the run) tokenized with the GPT-2 BPE tokenizer
(vocabulary $50{{,}}257$). Context length $T{{=}}{lm.get('seq_len', 1024)}$,
batch size $B{{=}}{lm.get('batch_size', 16)}$ sequences
(${batch_tokens // 1000}$K tokens per optimizer step). Optimizer:
AdamW ($\beta_1{{=}}0.9, \beta_2{{=}}0.95$), weight decay $0.1$ on all
non-bias, non-RMSNorm parameters; the Tucker core $C$ is held in a
separate parameter group with weight decay $0$. Gradient clipping at $1.0$.
LR schedule: linear warmup over $1\%$ of total steps, cosine decay to $1\%$
of peak; peak LR ${lm.get('lr', 3e-4)}$. Precision: bf16 autocast for the
forward/backward, fp32 master parameters and optimizer state.
Hardware: single NVIDIA A100 (UIUC NCSA Delta).
Total: $\approx{lm.get('n_steps', 6000)}$ optimizer steps,
${batch_tokens * lm.get('n_steps', 6000) // 10**6}$M training tokens.
We log validation cross-entropy every $\approx 2$M tokens on a held-out
$\approx 2$M-token slice of FineWeb-Edu (disjoint from the training stream
by random seed); the curves in Figure~\ref{{fig:lm_loss_curves}} use these
checkpoints.

The variance-preserving Tucker initialization is
$C_{{oij}} \sim \mathcal{{N}}(0, 1/r^2)$ for the off-diagonal entries with
diagonal warm start $C_{{ooo}} \leftarrow 1$ (using $o$ for the core output
index to avoid collision with the routing scalar $\alpha_j(x)$), so that at
initialization the layer evaluates the SwiGLU recovery form
$z_o = p_o\,\mathrm{{SiLU}}(q_o)$ plus $O(\varepsilon/r)$ noise at
$\varepsilon{{=}}10^{{-2}}$. Latent projections $P, Q$ and output projection
$R$ use He-uniform init.

\paragraph{{Constant-$\alpha$ ablation.}}
Calibration set: $32$ sequences of length $1024$ from the WikiText-2 train
split (${32 * 1024 // 1000}$K tokens), used to estimate the per-channel mean
$\bar\alpha_j = \mathbb{{E}}_x[\alpha_j(x)]$ for the
$\alpha\!=\!\bar\alpha$ condition. Activations are tapped at the input to the
gate projection (\textit{{after}} the FFN's input RMSNorm, which is the
standard tap point in HuggingFace SwiGLU implementations). All $24$ layers
are ablated simultaneously by replacing $\sigma(g_j^\top x)$ with the
constant for that condition. Evaluation perplexity: $4{{,}}096$-token chunk
of WikiText-2 validation (Section~\ref{{sec:eval_calibration}} reports
robustness to a $32{{,}}768$-token chunk).

\paragraph{{Synthetic Tucker teacher (Figure 1).}}
$d{{=}}{syn.get('d', 64)}$. Per teacher, $P, Q, R \sim \mathcal{{N}}(0, 1/d)$;
$C \in \mathbb{{R}}^{{k \times k \times k}}$ is sampled
$C_{{oij}} \sim \mathcal{{N}}(0, 1)$ and resampled until every frontal slice
$C^{{(j)}} = C[:, :, j]$ has full rank $k$ (typically the first sample
suffices). Inputs $x \sim \mathcal{{N}}(0, I_d)$;
train/val sizes ${syn.get('n_train', 50000)}$/${syn.get('n_val', 5000)}$.
Adam, lr ${syn.get('lr', 3e-3)}$, batch ${syn.get('batch_size', 512)}$,
${syn.get('n_steps', 8000)}$ steps with cosine schedule
($1\%$ floor, no warmup). ${syn.get('n_seeds', 8)}$ seeds per
$(k, m)$ cell, val MSE reported as the minimum across seeds (best fit
achievable). Aligned-SwiGLU students use the teacher's $P, Q$ as fixed
latent dictionaries (the matched-coordinates assumption of
Theorem~\ref{{thm:separation}}); each unit's gate index is assigned
round-robin over the $k$ gates. The SVD-construction marker (Figure 1)
is computed analytically from the teacher's $V_j = R C^{{(j)}}$ via SVD
without any optimization.

\paragraph{{Distillation (Figure 5).}}
Teacher: layer $\ell{{=}}{dist.get('teacher_layer', 4)}$ of a trained Tucker
LM ($d{{=}}512$, $r{{=}}s{{=}}128$). Inputs are sampled by streaming
FineWeb-Edu through the teacher LM and capturing the residual-stream
activations entering the chosen FFN; train/val sizes
${dist.get('n_train', 80000)}$/${dist.get('n_val', 8000)}$ tokens.
Adam, lr ${dist.get('lr', 1e-3)}$, batch ${dist.get('batch_size', 512)}$,
${dist.get('n_steps', 8000)}$ steps with cosine schedule
($1\%$ floor, $5\%$ warmup) and gradient clipping $1.0$.
${dist.get('n_seeds', 3)}$ seeds per budget; error bars are
$\pm$one std across seeds.

\paragraph{{Code.}} Code will be released upon acceptance.
"""

pathlib.Path("snippets").mkdir(exist_ok=True)
pathlib.Path("snippets/repro_appendix.tex").write_text(body)
print(body)
