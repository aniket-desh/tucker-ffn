#!/usr/bin/env python3
"""Measure FLOPs/token + throughput (fwd and fwd+bwd) for all sprint archs at
the LM config. Run on an IDLE gpu. Writes results/sprint_throughput.json."""
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, ".")
from lib.lm import make_lm, matched_swiglu_for_tucker  # noqa: E402
from lib.ll1_ffn import ll1_blocks_for_params  # noqa: E402

D, NH, NL, V, SL = 512, 8, 8, 50257, 1024
R = S = 128
M = matched_swiglu_for_tucker(D, R, S)
TARGET = 3 * D * M
dev = "cuda"
gpu = torch.cuda.get_device_name(0).replace("NVIDIA ", "")

ffn_flops = {
    "swiglu": 3 * D * M * 2,
    "tucker": (3 * D * R) * 2 + (S * R * R) * 2,
}
archs = {"swiglu": dict(kind="swiglu", m=M),
         "tucker": dict(kind="tucker", r=R, s=S)}
for L in (1, 2, 4, 8, 16):
    B = ll1_blocks_for_params(D, L, TARGET)
    archs[f"ll1_l{L}"] = dict(kind="ll1", n_blocks=B, block_rank=L)
    ffn_flops[f"ll1_l{L}"] = D * B * (2 * L + 1) * 2


def measure(kind_kwargs, train=False, B_=8, N=20):
    kw = dict(kind_kwargs)
    kind = kw.pop("kind")
    model = make_lm(kind, d=D, n_heads=NH, n_layers=NL, vocab_size=V,
                    max_seq_len=SL, **kw).to(dev)
    x = torch.randint(0, V, (B_, SL), device=dev)
    tgt = torch.randint(0, V, (B_, SL), device=dev)
    if train:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        def step():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    else:
        model.eval()
        def step():
            with torch.no_grad(), torch.amp.autocast(device_type="cuda",
                                                     dtype=torch.bfloat16):
                model(x)
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    del model
    torch.cuda.empty_cache()
    return B_ * SL * N / dt


out = {"gpu": gpu, "d": D, "n_layers": NL, "archs": {}}
for name, kw in archs.items():
    fwd = measure(kw, train=False)
    trn = measure(kw, train=True)
    out["archs"][name] = {"ffn_flops_per_token_per_layer": ffn_flops[name],
                          "fwd_tokens_per_sec": fwd,
                          "train_tokens_per_sec": trn}
    print(f"{name:10s} ffnFLOPs/tok/layer={ffn_flops[name]:.3e} "
          f"fwd={fwd:,.0f} tok/s  train={trn:,.0f} tok/s")

pathlib.Path("results").mkdir(exist_ok=True)
with open("results/sprint_throughput.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved results/sprint_throughput.json")
