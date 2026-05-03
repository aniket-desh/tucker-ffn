#!/usr/bin/env python3
"""Read diagonal-core Tucker from-scratch run and emit a Table 2 row.
Output: snippets/diag_tucker_row.tex."""
import json, pathlib, numpy as np

candidates = [
    "results/exp11_diag/tucker_diag_seed0/loss_log.json",
    "results/exp11_diag/tucker_diag_seed0/loss_log.json",
]
ll = None
for p in candidates:
    if pathlib.Path(p).exists():
        with open(p) as f:
            ll = json.load(f)
        used = p
        break
if ll is None:
    raise SystemExit("no diagonal Tucker run found yet")

final = float(ll[-1]["val_loss"])
ppl = float(np.exp(final))
SW_REF = 4.758  # results/exp11/swiglu_seed0 final val loss
delta = final - SW_REF

# emit a single \tabular row + a comment line. The LaTeX agent decides where
# to splice it into Table 2.
row = (
    rf"% Diagonal-core Tucker from scratch (same arch, core constrained to "
    rf"superdiagonal). Source: {used}\n"
    rf"\textsc{{Tucker-diagonal}} (var-preserving, single seed) "
    rf"& $52.5$M & ${final:.3f}$ & ${ppl:.2f}$ & "
    rf"${delta:+.3f}$ \\\\" "\n"
)
pathlib.Path("snippets").mkdir(exist_ok=True)
pathlib.Path("snippets/diag_tucker_row.tex").write_text(row)
print(row)
print(f"final val_loss = {final:.4f}  ppl = {ppl:.2f}  delta vs swiglu = {delta:+.3f} nats")
