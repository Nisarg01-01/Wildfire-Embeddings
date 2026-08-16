#!/usr/bin/env python
"""
make_graphical_abstract.py
==========================
Build the Elsevier graphical abstract for the manuscript: the global study-site
map (left) beside the pairwise cosine-similarity distribution (right), as a single
landscape image meeting Elsevier's minimum of 531 x 1328 px (height x width).

Inputs : data/Figures/fig1_study_sites.png  (existing map)
         data/cache/site_deltas.npz         (50-site deltas, for the distribution)
Output : data/Figures/graphical_abstract.{png,pdf}

Usage  : python make_graphical_abstract.py
"""
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import prithvi_common as pc
import paper_style as ps


def main():
    ps.apply_style()
    import matplotlib.pyplot as plt
    import seaborn as sns

    map_path = os.path.join(pc.FIGURES_OUT, "fig1_study_sites.png")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Missing site map: {map_path}")
    cache = pc.load_delta_cache()

    traj = pc.l2_normalize_rows(np.asarray(cache["delta"], dtype=np.float64))
    sim = cosine_similarity(traj)
    pairwise = sim[np.triu_indices_from(sim, k=1)]
    mean = pairwise.mean()

    # 2.5:1 landscape; 2656 x 1062 px at 300 dpi (>= Elsevier min 1328 x 531).
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(8.853, 3.54), gridspec_kw={"width_ratios": [1.55, 1.0]})

    # Left: the global site map (pre-rendered)
    axL.imshow(plt.imread(map_path))
    axL.axis("off")

    # Right: pairwise similarity distribution
    sns.histplot(pairwise, kde=True, color=ps.C_HIST, ax=axR, bins=30,
                 edgecolor="white", linewidth=0.3)
    axR.axvline(mean, color=ps.C_VLINE, ls="--", lw=ps.LINE_W,
                label=f"Mean {mean:.3f}")
    axR.set_title("Wildfire encoded as one consistent\ndirection across biomes",
                  fontsize=ps.FS_PANEL, fontweight="bold")
    axR.set_xlabel("Pairwise cosine similarity", fontsize=ps.FS_LABEL)
    axR.set_ylabel("Count", fontsize=ps.FS_LABEL)
    axR.legend(**ps.LEGEND_KW)
    axR.grid(True, axis="y", **ps.GRID_KW)
    sns.despine(ax=axR)

    fig.tight_layout(pad=0.6)
    out = os.path.join(pc.FIGURES_OUT, "graphical_abstract.pdf")
    ps.save_figure(fig, out, dpi=300)
    plt.close(fig)

    # Report final pixel size of the PNG
    png = os.path.splitext(out)[0] + ".png"
    try:
        from PIL import Image
        w, h = Image.open(png).size
        print(f"Graphical abstract: {w} x {h} px (Elsevier min 1328 x 531).")
    except Exception:
        pass


if __name__ == "__main__":
    main()
