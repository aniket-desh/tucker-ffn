"""siam-preprint plot theme, modeled after singh et al. (arxiv:2510.17734).

no titles, all spines, serif fonts, muted distinguishable colors,
markers + linestyle variation, light dotted grid, 300 dpi.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PALETTE = {
    "primary": "#1f77b4",      # steel blue — main data series
    "secondary": "#d4820e",    # dull orange — secondary
    "accent": "#2ca02c",       # muted green — highlights
    "neutral": "#7f7f7f",      # gray — reference lines, annotations
    "ablation": "#c44e52",     # muted red — ablation conditions
    "fill": "#1f77b4",         # same as primary, used with low alpha
    "black": "#2d2d2d",        # near-black for primary emphasis
}

# layer-position colormap: early layers cool, late layers warm
LAYER_CMAP = "coolwarm"

# ordered cycle for multi-series plots (matches siam convention)
COLOR_CYCLE = ["#2d2d2d", "#1f77b4", "#d4820e", "#2ca02c", "#c44e52", "#9467bd"]
MARKER_CYCLE = ["o", "s", "^", "v", "D", "p"]
LS_CYCLE = ["-", "--", "-.", ":", "-", "--"]


def setup_plot_style():
    """apply siam-style theme: serif fonts, all spines, light grid, 300 dpi."""
    plt.rcParams.update({
        # fonts — match latex computer modern
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif",
                        "Times New Roman", "Times"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        # spines — all four visible (siam standard)
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.6,
        # ticks — inward
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        # grid — light dotted, barely visible
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        # figure
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.05,
        # lines
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
    })
