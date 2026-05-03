#!/usr/bin/env python3
"""Compute FLOPs/token + measured tokens/sec for SwiGLU vs Tucker FFN at the
LM config. Emits snippets/compute_accounting.tex."""
import sys, time, pathlib, torch
sys.path.insert(0, ".")
from lib.lm import make_lm

D, NH, NL, V, SL = 512, 8, 8, 50257, 1024
M_SW = 1493
R_TK = S_TK = 128
dev = "cuda" if torch.cuda.is_available() else "cpu"
gpu_name = (torch.cuda.get_device_name(0) if dev == "cuda" else "CPU").replace(
    "NVIDIA ", "")


def swiglu_ffn_flops(d, m):    return 3 * d * m * 2          # gate, up, down
def tucker_ffn_flops(d, r, s): return (3 * d * r) * 2 + (s * r * r) * 2


F_S = NL * swiglu_ffn_flops(D, M_SW)
F_T = NL * tucker_ffn_flops(D, R_TK, S_TK)


def measure(arch, **kw):
    model = make_lm(arch, d=D, n_heads=NH, n_layers=NL, vocab_size=V,
                    max_seq_len=SL, **kw).to(dev).eval()
    if dev == "cuda":
        model = model.to(torch.bfloat16)
    B = 8
    x = torch.randint(0, V, (B, SL), device=dev)
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        N = 20
        for _ in range(N):
            _ = model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return B * SL * N / dt


tps_S = measure("swiglu", m=M_SW)
tps_T = measure("tucker", r=R_TK, s=S_TK)

if F_T < F_S * 0.95:
    cmp = "smaller"
elif F_T > F_S * 1.05:
    cmp = "larger"
else:
    cmp = "comparable"

snippet = (
    "At the matched-parameter LM config ($d{{=}}512$, $L{{=}}8$), SwiGLU and "
    "Tucker-core FFNs have $F_S \\approx {fs:.2e}$ and $F_T \\approx {ft:.2e}$ "
    "forward FLOPs per token respectively, with measured throughput "
    "${ts:.0f}$ and ${tt:.0f}$ tokens/sec on a single {gpu}; "
    "Tucker is {cmp} in compute despite matched parameters.\n"
).format(fs=F_S, ft=F_T, ts=tps_S, tt=tps_T, gpu=gpu_name, cmp=cmp)

pathlib.Path("snippets").mkdir(exist_ok=True)
pathlib.Path("snippets/compute_accounting.tex").write_text(snippet)
print(snippet)
print(f"F_S = {F_S:.3e}  F_T = {F_T:.3e}  ratio T/S = {F_T/F_S:.3f}")
print(f"tps_S = {tps_S:.0f}  tps_T = {tps_T:.0f}  ratio T/S = {tps_T/tps_S:.3f}")
