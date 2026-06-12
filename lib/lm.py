"""minimal llama-style transformer with swappable swiglu/tucker ffn.

architecture per block: rmsnorm -> attn(rope) -> residual -> rmsnorm -> ffn
-> residual. ffn is either SwiGLUFFN(d, m) or TuckerFFN(d, r, s); both
inherit from nn.Module so the block does not care which. param counting
helpers in lib.tucker_ffn make it easy to size for matched ffn budget.

we use a learned positional embedding implicit in rope (no absolute pos
embed), tied input/output embeddings to keep params reasonable.

minimal — no attention masking other than causal, no kv cache, no dropout,
no flash attention. fp32 weights with bf16 autocast on cuda.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ll1_ffn import LL1FFN
from .tucker_ffn import SwiGLUFFN, TuckerFFN


# ── rope ────────────────────────────────────────────────────────────────────

def _rope_cache(seq_len, head_dim, device, base=10000.0):
    """precompute cos and sin tables for rope, shape (seq_len, head_dim/2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2,
                                            device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, head_dim/2)
    return freqs.cos(), freqs.sin()


def _apply_rope(x, cos, sin):
    """rotate pairs of features by per-position angles. x: (b, h, t, hd)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    cos = cos[None, None, : x.shape[-2], :]
    sin = sin[None, None, : x.shape[-2], :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ── blocks ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms).to(x.dtype) * self.weight


class CausalAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # to (b, h, t, hd)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        # use sdpa with is_causal
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, t, self.d)
        return self.out(out)


@dataclass
class FFNConfig:
    kind: str = "swiglu"   # "swiglu", "tucker", or "ll1"
    m: int = None          # for swiglu
    r: int = None          # for tucker
    s: int = None          # for tucker (defaults to r)
    diagonal_only: bool = False  # for tucker
    diagonal_bias_init: bool = False  # init tucker C with superdiagonal bias
    diag_bias_eps: float = 1e-2       # off-diag noise std multiplier (eps/r)
    legacy_init: bool = False         # reproduce older (incorrect) init scaling
    n_blocks: int = None   # for ll1
    block_rank: int = None # for ll1
    struct_kind: str = None    # for swiglu_struct: monarch/blockdiag/lowrank/butterfly
    struct_nb: int = 4         # matrix blocks for structured kinds


def build_ffn(d, cfg):
    if cfg.kind == "swiglu":
        assert cfg.m is not None
        return SwiGLUFFN(d, cfg.m, bias=False)
    if cfg.kind == "swiglu_struct":
        from .structured_ffn import StructuredSwiGLU
        assert cfg.m is not None and cfg.struct_kind is not None
        return StructuredSwiGLU(d, cfg.m, cfg.struct_kind,
                                n_blocks=cfg.struct_nb)
    if cfg.kind == "ll1":
        assert cfg.n_blocks is not None and cfg.block_rank is not None
        return LL1FFN(d, n_blocks=cfg.n_blocks, block_rank=cfg.block_rank,
                      bias=False)
    if cfg.kind == "tucker":
        assert cfg.r is not None
        s = cfg.s if cfg.s is not None else cfg.r
        return TuckerFFN(d, r=cfg.r, s=s, diagonal_only=cfg.diagonal_only,
                          diagonal_bias_init=cfg.diagonal_bias_init,
                          diag_bias_eps=cfg.diag_bias_eps,
                          legacy_init=cfg.legacy_init,
                          bias=False)
    raise ValueError(cfg.kind)


class Block(nn.Module):
    def __init__(self, d, n_heads, ffn_cfg: FFNConfig):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = CausalAttention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.ffn = build_ffn(d, ffn_cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


@dataclass
class LMConfig:
    vocab_size: int = 50257
    d: int = 512
    n_heads: int = 8
    n_layers: int = 8
    max_seq_len: int = 1024
    ffn: FFNConfig = field(default_factory=FFNConfig)
    tied: bool = True


class LM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d)
        self.blocks = nn.ModuleList([
            Block(cfg.d, cfg.n_heads, cfg.ffn) for _ in range(cfg.n_layers)
        ])
        self.norm_f = RMSNorm(cfg.d)
        if cfg.tied:
            self.lm_head = None  # tied to tok_embed
        else:
            self.lm_head = nn.Linear(cfg.d, cfg.vocab_size, bias=False)
        # init
        nn.init.normal_(self.tok_embed.weight, std=1.0 / math.sqrt(cfg.d))
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, std=1.0 / math.sqrt(cfg.d))
        self._rope_cache = None

    def get_rope(self, seq_len, device):
        if self._rope_cache is None or self._rope_cache[0].shape[0] < seq_len \
                or self._rope_cache[0].device != device:
            head_dim = self.cfg.d // self.cfg.n_heads
            self._rope_cache = _rope_cache(seq_len, head_dim, device)
        cos, sin = self._rope_cache
        return cos[:seq_len], sin[:seq_len]

    def forward(self, idx, targets=None):
        b, t = idx.shape
        cos, sin = self.get_rope(t, idx.device)
        x = self.tok_embed(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = x @ self.tok_embed.weight.T

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def num_params(self, exclude_embed=False):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embed:
            n -= self.tok_embed.weight.numel()
            if self.lm_head is not None:
                n -= self.lm_head.weight.numel()
        return n


# ── helpers for matched-budget configs ──────────────────────────────────────

def matched_swiglu_for_tucker(d, r, s):
    """return swiglu hidden width m such that 3*d*m matches tucker(d,r,s)
    parameter count as closely as possible."""
    tucker_p = d * (2 * r + s) + s * r * r
    return max(1, int(round(tucker_p / (3 * d))))


def make_lm(kind, d, n_heads, n_layers, vocab_size, max_seq_len,
            m=None, r=None, s=None, diagonal_only=False,
            diagonal_bias_init=False, diag_bias_eps=1e-2,
            legacy_init=False, tied=True,
            n_blocks=None, block_rank=None,
            struct_kind=None, struct_nb=4):
    """convenience factory. kind in {"swiglu","tucker","ll1"}."""
    if kind == "swiglu":
        ffn = FFNConfig(kind="swiglu", m=m)
    elif kind == "swiglu_struct":
        ffn = FFNConfig(kind="swiglu_struct", m=m,
                        struct_kind=struct_kind, struct_nb=struct_nb)
    elif kind == "ll1":
        ffn = FFNConfig(kind="ll1", n_blocks=n_blocks, block_rank=block_rank)
    elif kind == "tucker":
        ffn = FFNConfig(kind="tucker", r=r, s=s,
                          diagonal_only=diagonal_only,
                          diagonal_bias_init=diagonal_bias_init,
                          diag_bias_eps=diag_bias_eps,
                          legacy_init=legacy_init)
    else:
        raise ValueError(kind)
    cfg = LMConfig(
        vocab_size=vocab_size, d=d, n_heads=n_heads, n_layers=n_layers,
        max_seq_len=max_seq_len, ffn=ffn, tied=tied,
    )
    return LM(cfg)
