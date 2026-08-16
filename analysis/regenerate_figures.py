#!/usr/bin/env python
"""
regenerate_figures.py
=====================
Regenerate the paper figures from cached analysis outputs, so figures can be
rebuilt without re-running the Prithvi model.

Figures produced
----------------
  * Figure 2  global_fire_similarity_consensus.{pdf,png}
                 pairwise cosine-similarity heatmap + distribution (notebook Cell 7)
  * null_baselines.{pdf,png}
                 empirical/anisotropy-aware nulls for the 0.813 result
                 (see null_analysis.py)
  * Figure 3  recovery_curve_3panel.{pdf,png}
                 temporal recovery curves from temporal_sim.csv
  * Figure 4  fire_vs_logging_combined.{pdf,png}
                 fire-vs-logging t-SNE and cross-class similarity heatmap

Usage
-----
    python regenerate_figures.py              # all available figures (default)
    python regenerate_figures.py --consensus  # Fig 2 only
    python regenerate_figures.py --null       # null figure only
    python regenerate_figures.py --fig3       # temporal figure only
    python regenerate_figures.py --fig4       # fire-vs-logging figure only

Required caches
---------------
  * data/cache/site_deltas.npz   (Figure 2 and null baselines)
  * data/cache/temporal_sim.csv  (Figure 3)
  * data/cache/selectivity.npz   (Figure 4)
"""
import argparse
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import prithvi_common as pc
import paper_style as ps


def fig2_consensus(cache):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ps.apply_style()

    traj = pc.l2_normalize_rows(np.asarray(cache["delta"], dtype=np.float64))
    labels = [f"{c} - {i}" for c, i in zip(cache["countries"], cache["ids"])]
    sim = cosine_similarity(traj)
    pairwise = sim[np.triu_indices_from(sim, k=1)]
    avg = pairwise.mean()
    print(f"Figure 2: {len(labels)} sites, {pairwise.size} pairs, mean cosine = {avg:.6f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    sns.heatmap(sim, xticklabels=labels, yticklabels=labels, cmap="rocket_r",
                cbar_kws=dict(label="Cosine Similarity", **ps.CBAR_KW),
                linewidths=ps.HM_LW, linecolor="white", ax=ax1)
    ax1.set_title("A. Global Wildfire Embedding Trajectory Consistency",
                  fontsize=ps.FS_PANEL, fontweight="bold")
    ax1.tick_params(labelsize=ps.FS_TICK - 1, length=2)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

    sns.histplot(pairwise, kde=True, color=ps.C_HIST, ax=ax2, bins=30,
                 edgecolor="white", linewidth=0.4)
    ax2.axvline(avg, color=ps.C_VLINE, linestyle="--", lw=ps.LINE_W, label=f"Mean: {avg:.4f}")
    ax2.set_title("B. Pairwise Similarity Distribution", fontsize=ps.FS_PANEL, fontweight="bold")
    ax2.set_xlabel("Cosine Similarity Score", fontsize=ps.FS_LABEL)
    ax2.set_ylabel("Count", fontsize=ps.FS_LABEL)
    ax2.legend(**ps.LEGEND_KW)
    ax2.grid(True, axis="y", **ps.GRID_KW)
    sns.despine(ax=ax2)

    fig.tight_layout(pad=ps.FIG_PAD)
    ps.save_figure(fig, os.path.join(pc.FIGURES_OUT, "global_fire_similarity_consensus.pdf"))
    plt.close(fig)


def fig_null(cache, nperm=1000):
    import null_analysis as na
    delta = np.asarray(cache["delta"], dtype=np.float64)
    obs, _ = na.mean_pairwise_cosine(pc.l2_normalize_rows(delta))
    perm = na.permutation_null(delta, nperm=nperm)
    rand = na.random_unit_null(delta.shape[0], delta.shape[1], nsim=min(nperm, 2000))
    pre_floor, _  = na.mean_pairwise_cosine(pc.l2_normalize_rows(cache["pre_mean"]))
    post_floor, _ = na.mean_pairwise_cosine(pc.l2_normalize_rows(cache["post_mean"]))
    na.make_figure(obs, perm, rand, pre_floor, post_floor)


# --------------------------------------------------------------------------- #
# Figure 3: temporal recovery curves (Leading / Trailing windows)
# --------------------------------------------------------------------------- #
def fig3_recovery():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    ps.apply_style()

    if not os.path.exists(pc.TEMPORAL_CACHE):
        raise FileNotFoundError(
            f"No temporal cache at {pc.TEMPORAL_CACHE}. Run compute_fig34_cache.py first.")
    df = pd.read_csv(pc.TEMPORAL_CACHE)
    palette = {"Leading": ps.C_PRE, "Trailing": ps.C_POST}
    sites = list(dict.fromkeys(df["site"]))            # preserve order
    n = len(sites)

    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 3.6), squeeze=False)
    axes = axes[0]
    for ax, sid in zip(axes, sites):
        sub = df[df["site"] == sid]
        label = sub["label"].iloc[0]
        sns.lineplot(data=sub, x="offset", y="sim", hue="window", style="window",
                     hue_order=["Leading", "Trailing"], style_order=["Leading", "Trailing"],
                     palette=palette, markers=True, dashes=False,
                     lw=ps.LINE_W, markersize=6, ax=ax)
        ax.axvline(0, color=ps.C_VLINE, ls="--", lw=ps.LINE_W * 0.7)
        ax.set_title(f"{label}  ({sid})", fontsize=ps.FS_PANEL, fontweight="bold")
        ax.set_xlabel("Years from fire", fontsize=ps.FS_LABEL)
        ax.set_ylabel("Cosine similarity to reference", fontsize=ps.FS_LABEL)
        ax.set_xticks(sorted(sub["offset"].unique()))
        ax.legend(title="Window", loc="lower right", **ps.LEGEND_KW)
        ax.grid(True, axis="y", **ps.GRID_KW)
        sns.despine(ax=ax)

    fig.tight_layout(pad=ps.FIG_PAD)
    ps.save_figure(fig, os.path.join(pc.FIGURES_OUT, "recovery_curve_3panel.pdf"))
    plt.close(fig)
    print(f"Figure 3: {n} sites, {len(df)} points.")


# --------------------------------------------------------------------------- #
# Figure 4: combined t-SNE + cross-class similarity (per author comment)
# --------------------------------------------------------------------------- #
def fig4_combined():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    from sklearn.metrics.pairwise import cosine_similarity
    ps.apply_style()

    if not os.path.exists(pc.SELECTIVITY_CACHE):
        raise FileNotFoundError(
            f"No selectivity cache at {pc.SELECTIVITY_CACHE}. Run compute_fig34_cache.py first.")
    z = np.load(pc.SELECTIVITY_CACHE, allow_pickle=True)
    ids   = [str(x) for x in z["ids"]]
    types = [str(x) for x in z["types"]]
    labels = [str(x) for x in z["labels"]]
    pre_embs  = list(z["pre_embs"])
    post_embs = list(z["post_embs"])
    deltas = z["deltas"]

    fire_idx = [i for i, t in enumerate(types) if t == "Fire"]
    log_idx  = [i for i, t in enumerate(types) if t == "Logging"]

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.28)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ---- Panel A: t-SNE of pre/post patch embeddings with pre->post arrows ----
    pts, meta = [], []
    for i in range(len(ids)):
        for stage, arr in (("Pre", pre_embs[i]), ("Post", post_embs[i])):
            pts.append(arr)
            meta.extend([{"i": i, "type": types[i], "stage": stage}] * len(arr))
    X = np.vstack(pts)
    perp = min(30, max(5, X.shape[0] // 3))
    X2 = TSNE(n_components=2, perplexity=perp, random_state=pc.RANDOM_SEED,
              init="pca").fit_transform(X)

    hue = [f"{m['type']}: {labels[m['i']]}" for m in meta]
    stage = [m["stage"] for m in meta]
    palette = {}
    for k, i in enumerate(fire_idx):
        palette[f"Fire: {labels[i]}"] = ps.FIRE_PALETTE[k % len(ps.FIRE_PALETTE)]
    for k, i in enumerate(log_idx):
        palette[f"Logging: {labels[i]}"] = ps.LOG_PALETTE[k % len(ps.LOG_PALETTE)]

    sns.scatterplot(ax=axA, x=X2[:, 0], y=X2[:, 1], hue=hue, style=stage,
                    markers=ps.STAGE_MARKERS, palette=palette, alpha=0.75,
                    s=ps.SCATTER_S, edgecolor="w", linewidth=ps.MARKER_EW)
    pos = 0
    for i in range(len(ids)):
        n_pre, n_post = len(pre_embs[i]), len(post_embs[i])
        p_pre  = X2[pos: pos + n_pre].mean(0)
        p_post = X2[pos + n_pre: pos + n_pre + n_post].mean(0)
        col = palette[f"{types[i]}: {labels[i]}"]
        axA.annotate("", xy=p_post, xytext=p_pre,
                     arrowprops=dict(arrowstyle="fancy,tail_width=0.4,head_width=1.0",
                                     fc=col, ec="black", lw=0.5, alpha=0.85))
        pos += n_pre + n_post
    axA.set_title("A. Disturbance trajectories (t-SNE of patch embeddings)",
                  fontsize=ps.FS_PANEL, fontweight="bold")
    axA.set_xlabel("t-SNE dim 1", fontsize=ps.FS_LABEL)
    axA.set_ylabel("t-SNE dim 2", fontsize=ps.FS_LABEL)
    axA.legend(loc="best", title="Class: site (stage)", **ps.LEGEND_KW)
    axA.grid(True, **ps.GRID_KW)
    sns.despine(ax=axA)

    # ---- Panel B: cross-class similarity heatmap ----
    fd = pc.l2_normalize_rows(deltas[fire_idx])
    ld = pc.l2_normalize_rows(deltas[log_idx])
    cross = cosine_similarity(fd, ld)
    intra = (cosine_similarity(fd, fd)[np.triu_indices(len(fire_idx), k=1)].mean()
             if len(fire_idx) > 1 else float("nan"))
    sns.heatmap(cross, annot=True, fmt=".3f", cmap="vlag", center=0, vmin=-1, vmax=1,
                xticklabels=[labels[i] for i in log_idx],
                yticklabels=[labels[i] for i in fire_idx],
                linewidths=ps.HM_LW, linecolor="white",
                cbar_kws=dict(label="Cosine similarity", **ps.CBAR_KW), ax=axB)
    axB.set_title("B. Cross-class similarity\n(wildfire × logging shifts)",
                  fontsize=ps.FS_PANEL, fontweight="bold")
    axB.set_xlabel("Logging sites", fontsize=ps.FS_LABEL)
    axB.set_ylabel("Wildfire sites", fontsize=ps.FS_LABEL)
    if not np.isnan(intra):
        axB.text(0.5, -0.22, f"Intra-class wildfire similarity: {intra:.3f}",
                 transform=axB.transAxes, ha="center", va="top",
                 fontsize=ps.FS_ANNOT, color="0.35", style="italic")

    fig.tight_layout(pad=ps.FIG_PAD)
    ps.save_figure(fig, os.path.join(pc.FIGURES_OUT, "fire_vs_logging_combined.pdf"))
    plt.close(fig)
    print(f"Figure 4: cross mean = {cross.mean():.3f}, range "
          f"{cross.min():.3f}-{cross.max():.3f}, intra-fire = {intra:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--consensus", action="store_true", help="Figure 2 (needs site_deltas.npz)")
    ap.add_argument("--null", action="store_true", help="null-baselines figure (needs site_deltas.npz)")
    ap.add_argument("--fig3", action="store_true", help="Figure 3 recovery curves (needs temporal_sim.csv)")
    ap.add_argument("--fig4", action="store_true", help="Figure 4 combined t-SNE+heatmap (needs selectivity.npz)")
    ap.add_argument("--cache", default=pc.DELTA_CACHE)
    ap.add_argument("--nperm", type=int, default=1000)
    args = ap.parse_args()

    selected = (args.consensus, args.null, args.fig3, args.fig4)
    do_all = not any(selected)

    # Figures 2 and the null share the 50-site delta cache; load it only if needed.
    if args.consensus or args.null or do_all:
        try:
            cache = pc.load_delta_cache(args.cache)
            if args.consensus or do_all:
                fig2_consensus(cache)
            if args.null or do_all:
                fig_null(cache, nperm=args.nperm)
        except FileNotFoundError as e:
            if do_all:
                print(f"[skip Fig 2 / null] {e}")
            else:
                raise

    if args.fig3 or do_all:
        try:
            fig3_recovery()
        except FileNotFoundError as e:
            if do_all:
                print(f"[skip Fig 3] {e}")
            else:
                raise
    if args.fig4 or do_all:
        try:
            fig4_combined()
        except FileNotFoundError as e:
            if do_all:
                print(f"[skip Fig 4] {e}")
            else:
                raise


if __name__ == "__main__":
    main()
