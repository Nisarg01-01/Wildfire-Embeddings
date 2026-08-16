"""
paper_style.py
==============
Single source of truth for figure styling, ported verbatim from notebook Cell 2
so regenerated figures match the published ones. Import and call apply_style().
"""
import os
import matplotlib
matplotlib.use("Agg")          # headless file output; avoids Qt/PySide backend
import matplotlib.pyplot as plt
import seaborn as sns

FONT_FAMILY = "DejaVu Sans"
FS_TICK, FS_LABEL, FS_PANEL, FS_SUPER, FS_ANNOT, FS_LEGEND = 8, 9, 10, 11, 8, 8

C_FIRE_1, C_FIRE_2, C_FIRE_3, C_FIRE_4 = "#B71C1C", "#E64A19", "#F9A825", "#78909C"
C_LOG_1, C_LOG_2, C_LOG_3 = "#1565C0", "#00695C", "#6A1B9A"
C_PRE, C_POST, C_VLINE, C_HIST, C_GRID = "#2E7D32", "#C62828", "#37474F", "#C62828", "#ECEFF1"

FIRE_PALETTE = [C_FIRE_1, C_FIRE_2]
LOG_PALETTE  = [C_LOG_1, C_LOG_2]
STAGE_MARKERS = {"Pre": "o", "Post": "X"}
STAGE_PALETTE = {"Pre": C_PRE, "Post": C_POST}

SCATTER_S, MARKER_EW, LINE_W, SPINE_W = 65, 0.3, 1.8, 0.6
TICK_W, TICK_LEN, HM_LW, FIG_PAD, CBAR_SHRINK = 0.6, 3, 0.3, 0.8, 0.82

LEGEND_KW = dict(frameon=True, framealpha=0.95, edgecolor="0.78",
                 fancybox=False, borderpad=0.5,
                 fontsize=FS_LEGEND, title_fontsize=FS_LEGEND)
GRID_KW = dict(linestyle=":", linewidth=0.45, color=C_GRID, alpha=1.0)
CBAR_KW = dict(shrink=CBAR_SHRINK, pad=0.02)


def apply_style():
    sns.reset_orig()
    sns.set_style("ticks")
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": FS_TICK,
        "axes.titlesize": FS_PANEL, "axes.titlepad": 7, "axes.titleweight": "bold",
        "axes.labelsize": FS_LABEL, "axes.labelpad": 4,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND, "legend.title_fontsize": FS_LEGEND,
        "axes.linewidth": SPINE_W,
        "xtick.major.width": TICK_W, "ytick.major.width": TICK_W,
        "xtick.major.size": TICK_LEN, "ytick.major.size": TICK_LEN,
        "xtick.minor.size": TICK_LEN * 0.6, "ytick.minor.size": TICK_LEN * 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": LINE_W, "patch.linewidth": 0.5,
        "figure.dpi": 150, "savefig.dpi": 600,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "0.3", "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_figure(fig, path, dpi=600):
    ext = os.path.splitext(path)[1].lower()
    kw = dict(bbox_inches="tight", pad_inches=0.05, facecolor="white", edgecolor="none")
    fig.savefig(path, dpi=None if ext in {".pdf", ".svg", ".eps"} else dpi, **kw)
    print(f"Saved: {path}")
    if ext in {".pdf", ".svg", ".eps"}:
        png = os.path.splitext(path)[0] + ".png"
        fig.savefig(png, dpi=dpi, **kw)
        print(f"Saved: {png}")
