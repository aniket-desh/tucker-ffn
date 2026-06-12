#!/usr/bin/env python3
"""Figure 1: the CP -> LL1 -> Tucker ladder.

A/B/C: schematic of one routed unit for SwiGLU (rank-1 atom per route),
LL1 (rank-L block per route), Tucker (dense core, full-rank V_j per route).
D: at matched parameters (d=512, N=2.293M), routes vs per-route rank rectangles
   (area = CP atoms controlled).
"""
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, ".")
from lib import setup_plot_style  # noqa: E402

setup_plot_style()

C_CP = "#2c7fb8"      # blue
C_LL1 = "#31a354"     # green
C_TUCKER = "#d7301f"  # red
C_GATE = "#41ab5d"
C_GRAY = "#777777"

fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.4))


def box(ax, x, y, w, h, color, label, fs=8, alpha=0.25, lw=1.2, text_color="k"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, alpha=alpha,
                           edgecolor=color, lw=lw))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            color=text_color)


def arrow(ax, x1, y1, x2, y2, color="k", ls="-", lw=1.1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, color=color, ls=ls, lw=lw))


for ax in axes[:3]:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

# ── A: SwiGLU unit ──────────────────────────────────────────────────────────
ax = axes[0]
ax.set_title(r"A)  SwiGLU: routed CP", fontsize=10, loc="left")
box(ax, 0.2, 4.2, 1.6, 1.6, C_GRAY, r"$x$")
box(ax, 3.2, 6.6, 2.0, 1.5, C_CP, r"$w_j^\top x$")
box(ax, 3.2, 1.9, 2.0, 1.5, C_GATE, r"$g_j^\top x$")
arrow(ax, 1.8, 5.4, 3.2, 7.3)
arrow(ax, 1.8, 4.6, 3.2, 2.7)
ax.add_patch(mpatches.Circle((6.6, 5.0), 0.55, facecolor="white",
                             edgecolor="k", lw=1.1))
ax.text(6.6, 5.0, r"$\times$", ha="center", va="center", fontsize=12)
arrow(ax, 5.2, 7.0, 6.3, 5.5)
arrow(ax, 5.2, 2.7, 6.3, 4.5)
ax.text(5.6, 1.0, r"route $\alpha_j(x)=\sigma(g_j^\top x)$", fontsize=8,
        color=C_GATE)
arrow(ax, 5.0, 1.7, 6.5, 4.4, color=C_GATE, ls=":")
box(ax, 7.9, 4.2, 1.9, 1.6, "#807dba", r"$u_j$")
arrow(ax, 7.15, 5.0, 7.9, 5.0)
ax.text(5.0, 9.55, r"1 route $\to$ 1 rank-1 atom", fontsize=9, ha="center")
ax.text(5.0, 8.7, r"$m$ routes, $m$ atoms", fontsize=8, ha="center",
        color=C_GRAY)

# ── B: LL1 block ────────────────────────────────────────────────────────────
ax = axes[1]
ax.set_title(r"B)  LL1 / block-CP", fontsize=10, loc="left")
box(ax, 0.2, 4.2, 1.6, 1.6, C_GRAY, r"$x$")
box(ax, 3.0, 6.4, 2.4, 1.7, C_LL1, r"$A_b^\top x \in \mathbb{R}^{L}$")
box(ax, 3.0, 1.9, 2.4, 1.5, C_GATE, r"$g_b^\top x$")
arrow(ax, 1.8, 5.4, 3.0, 7.2)
arrow(ax, 1.8, 4.6, 3.0, 2.7)
for dy in (-0.5, 0.0, 0.5):
    ax.add_patch(mpatches.Circle((6.5, 5.0 + dy * 1.7), 0.42,
                                 facecolor="white", edgecolor="k", lw=1.0))
    ax.text(6.5, 5.0 + dy * 1.7, r"$\times$", ha="center", va="center",
            fontsize=9)
    arrow(ax, 5.4, 7.0, 6.15, 5.1 + dy * 1.7)
arrow(ax, 5.0, 2.6, 6.2, 4.2, color=C_GATE, ls=":")
ax.text(5.4, 1.0, r"one route $\alpha_b(x)$, shared", fontsize=8, color=C_GATE)
box(ax, 7.9, 4.2, 1.9, 1.6, "#807dba", r"$U_b$")
for dy in (-0.85, 0.0, 0.85):
    arrow(ax, 6.95, 5.0 + dy, 7.9, 5.0)
ax.text(5.0, 9.55, r"1 route $\to$ 1 rank-$L$ block", fontsize=9, ha="center")
ax.text(5.0, 8.7, r"$B$ routes, $BL$ atoms,  $V_b = U_b A_b^\top$",
        fontsize=8, ha="center", color=C_GRAY)

# ── C: dense Tucker ─────────────────────────────────────────────────────────
ax = axes[2]
ax.set_title(r"C)  dense Tucker", fontsize=10, loc="left")
box(ax, 0.2, 4.2, 1.6, 1.6, C_GRAY, r"$x$")
box(ax, 3.0, 6.6, 1.9, 1.5, C_CP, r"$P^\top x$")
box(ax, 3.0, 1.9, 1.9, 1.5, C_GATE, r"$Q^\top x$")
arrow(ax, 1.8, 5.4, 3.0, 7.3)
arrow(ax, 1.8, 4.6, 3.0, 2.7)
# dense core grid
gx, gy, cs = 5.6, 3.7, 0.42
for i in range(5):
    for j in range(5):
        ax.add_patch(Rectangle((gx + i * cs, gy + j * cs), cs * 0.92,
                               cs * 0.92, facecolor=C_TUCKER,
                               alpha=0.18 + 0.62 * ((i * 7 + j * 3) % 5) / 5,
                               edgecolor="none"))
ax.text(gx + 2.5 * cs, gy - 0.65, r"core $C\in\mathbb{R}^{s\times r\times r}$",
        fontsize=8, ha="center", color=C_TUCKER)
arrow(ax, 4.9, 7.3, 6.4, 5.9)
arrow(ax, 4.9, 2.7, 6.4, 3.6, color=C_GATE, ls=":")
box(ax, 8.2, 4.2, 1.6, 1.6, "#807dba", r"$R$")
arrow(ax, 7.8, 4.9, 8.2, 4.9)
ax.text(5.0, 9.55, r"1 route $\to$ full-rank $V_j = RC^{(j)}$", fontsize=9,
        ha="center")
ax.text(5.0, 8.7, r"$r$ routes, all-to-all core, gauge freedom", fontsize=8,
        ha="center", color=C_GRAY)

# ── D: matched-budget geometry ──────────────────────────────────────────────
ax = axes[3]
ax.set_title(r"D)  matched parameters ($d{=}512$, $N{=}2.29$M)", fontsize=10,
             loc="left")
configs = [
    ("SwiGLU", 1493, 1, C_CP),
    ("LL1 $L{=}4$", 498, 4, C_LL1),
    ("LL1 $L{=}16$", 136, 16, C_LL1),
    ("Tucker", 128, 128, C_TUCKER),
]
x0 = 0
for name, routes, rank, color in configs:
    ax.add_patch(Rectangle((x0, 0), routes, rank, facecolor=color, alpha=0.45,
                           edgecolor=color, lw=1.4))
    ytxt = min(rank * 1.45, 320)
    ax.text(x0 + routes / 2, ytxt, name, fontsize=8, ha="center", color=color)
    ax.text(x0 + routes / 2, max(rank * 0.32, 0.45),
            f"{routes}×{rank}", fontsize=7, ha="center")
    x0 += routes + 110
ax.set_yscale("log", base=2)
ax.set_xlabel("routes (independent gates)", fontsize=9)
ax.set_ylabel("per-route rank", fontsize=9)
ax.set_xlim(-40, x0)
ax.set_ylim(0.4, 800)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(f"docs/structured_tensor_ffn_sprint/figures/fig1_ladder.{ext}",
                dpi=200, bbox_inches="tight")
print("saved fig1_ladder")
