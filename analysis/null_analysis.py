#!/usr/bin/env python
"""
null_analysis.py
================
Empirical, anisotropy-aware null baselines for the cross-site wildfire-direction
alignment (the headline mean pairwise cosine of ~0.813).

Motivation
----------
The manuscript currently benchmarks 0.813 only against random unit vectors
(expected cosine 0, std 1/sqrt(1024) ~= 0.031). That floor assumes the embedding
space is ISOTROPIC, which transformer embeddings generally are not, so "~25 sigma
above random" can overstate the effect. This script computes nulls that control
for anisotropy and pipeline structure, converting the claim from "far from random
noise" to "far from what the same embedding space produces without a coherent
disturbance":

  1. Observed       : mean pairwise cosine of the 50 site delta vectors.
  2. Permutation null: independently shuffle each of the 1024 embedding dimensions
                       across the 50 sites. This DESTROYS cross-site co-orientation
                       while PRESERVING each dimension's marginal distribution
                       (hence the anisotropic per-dimension scale). The gap between
                       observed and this null isolates genuine cross-site alignment.
  3. Anisotropy floor: mean pairwise cosine of the raw per-site mean PRE (and POST)
                       embeddings. Shows how aligned arbitrary embeddings already
                       are in this space, before any differencing.
  4. Analytical floor: isotropic random-unit-vector reference (mean 0, std 1/sqrt d)
                       plus a Monte-Carlo simulation of the MEAN over 1225 pairs.

Usage
-----
    python null_analysis.py                 # uses cached deltas; 1000 permutations
    python null_analysis.py --nperm 5000
    python null_analysis.py --no-fig

Outputs
-------
    code/data/null_analysis_results.json
    code/data/Figures/null_baselines.pdf (+ .png)
"""
import argparse
import json
import os

import numpy as np

import prithvi_common as pc


def mean_pairwise_cosine(mat_unit):
    """Mean of upper-triangle cosine similarities for L2-normalised rows."""
    sim = mat_unit @ mat_unit.T
    iu = np.triu_indices_from(sim, k=1)
    return float(sim[iu].mean()), sim[iu]


def permutation_null(delta, nperm=1000, seed=pc.RANDOM_SEED):
    """Column-wise (per-dimension) row shuffle null. Returns array of nperm
    mean-pairwise-cosine values."""
    rng = np.random.default_rng(seed)
    n, d = delta.shape
    col = np.arange(d)
    out = np.empty(nperm, dtype=np.float64)
    for k in range(nperm):
        # independent permutation of the 50 sites within each of the d dimensions
        idx = np.argsort(rng.random((n, d)), axis=0)
        shuffled = delta[idx, col]                 # (n, d), marginals preserved per col
        unit = pc.l2_normalize_rows(shuffled)
        out[k], _ = mean_pairwise_cosine(unit)
    return out


def random_unit_null(n, d, nsim=1000, seed=pc.RANDOM_SEED):
    """Monte-Carlo distribution of the MEAN pairwise cosine for n isotropic random
    unit vectors in d dims (fair comparison to the observed mean, not per-pair)."""
    rng = np.random.default_rng(seed + 1)
    out = np.empty(nsim, dtype=np.float64)
    for k in range(nsim):
        v = rng.standard_normal((n, d))
        out[k], _ = mean_pairwise_cosine(pc.l2_normalize_rows(v))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=pc.DELTA_CACHE)
    ap.add_argument("--nperm", type=int, default=1000, help="permutation-null draws")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    cache = pc.load_delta_cache(args.cache)
    delta = np.asarray(cache["delta"], dtype=np.float64)
    n, d = delta.shape

    # 1. observed
    obs_mean, obs_pairs = mean_pairwise_cosine(pc.l2_normalize_rows(delta))

    # 2. permutation null
    perm = permutation_null(delta, nperm=args.nperm)
    perm_mean, perm_std = float(perm.mean()), float(perm.std())
    z_perm = (obs_mean - perm_mean) / perm_std if perm_std > 0 else float("inf")
    # one-sided empirical p (add-one smoothing)
    p_perm = float((np.sum(perm >= obs_mean) + 1) / (perm.size + 1))

    # 3. anisotropy floor (raw pre / post embeddings, not differences)
    pre_floor, _  = mean_pairwise_cosine(pc.l2_normalize_rows(cache["pre_mean"]))
    post_floor, _ = mean_pairwise_cosine(pc.l2_normalize_rows(cache["post_mean"]))

    # 4. analytical + simulated isotropic floor
    analytic_perpair_std = 1.0 / np.sqrt(d)
    rand = random_unit_null(n, d, nsim=min(args.nperm, 2000))
    rand_mean, rand_std = float(rand.mean()), float(rand.std())

    results = {
        "n_sites": int(n), "dim": int(d), "n_pairs": int(obs_pairs.size),
        "observed_mean_pairwise_cosine": obs_mean,
        "observed_pairwise_std": float(obs_pairs.std()),
        "permutation_null": {
            "nperm": int(args.nperm), "mean": perm_mean, "std": perm_std,
            "z_score": z_perm, "p_one_sided": p_perm,
        },
        "anisotropy_floor": {"pre_embeddings": pre_floor, "post_embeddings": post_floor},
        "isotropic_reference": {
            "analytic_per_pair_std": float(analytic_perpair_std),
            "simulated_mean_over_pairs": rand_mean,
            "simulated_std_over_pairs": rand_std,
        },
    }

    out_json = os.path.join(os.path.dirname(pc.CACHE_DIR), "null_analysis_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # ---- console summary (paper-ready numbers) ----
    print("\n" + "=" * 64)
    print(f"Observed mean pairwise cosine (n={n}, {obs_pairs.size} pairs): {obs_mean:.4f}")
    print("-" * 64)
    print(f"Permutation null (per-dimension shuffle, {args.nperm} draws):")
    print(f"    mean = {perm_mean:.4f}   std = {perm_std:.4f}")
    print(f"    observed is {z_perm:.1f} sigma above this null   (p = {p_perm:.2e})")
    print(f"Anisotropy floor (raw embeddings): pre = {pre_floor:.4f}  post = {post_floor:.4f}")
    print(f"Isotropic reference: per-pair std = {analytic_perpair_std:.4f}; "
          f"simulated mean over {obs_pairs.size} pairs = {rand_mean:.4f} +/- {rand_std:.4f}")
    print("=" * 64)
    print(f"Wrote {out_json}")

    if not args.no_fig:
        make_figure(obs_mean, perm, rand, pre_floor, post_floor)

    return results


def make_figure(obs_mean, perm, rand, pre_floor, post_floor):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import paper_style as ps
    ps.apply_style()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    # Panel A: permutation-null distribution vs observed
    sns.histplot(perm, kde=True, color=ps.C_LOG_1, ax=axA, bins=30,
                 edgecolor="white", linewidth=0.4, stat="density")
    sns.histplot(rand, kde=True, color="0.6", ax=axA, bins=30,
                 edgecolor="white", linewidth=0.3, stat="density", alpha=0.5)
    axA.axvline(obs_mean, color=ps.C_HIST, ls="--", lw=ps.LINE_W,
                label=f"Observed: {obs_mean:.3f}")
    axA.set_title("A. Cross-site alignment vs empirical nulls",
                  fontsize=ps.FS_PANEL, fontweight="bold")
    axA.set_xlabel("Mean pairwise cosine similarity", fontsize=ps.FS_LABEL)
    axA.set_ylabel("Density", fontsize=ps.FS_LABEL)
    axA.legend(**ps.LEGEND_KW)
    axA.text(0.02, 0.82,
             "blue = per-dimension shuffle null\ngrey = isotropic random vectors",
             transform=axA.transAxes, fontsize=ps.FS_ANNOT, color="0.35", va="top")
    axA.grid(True, axis="y", **ps.GRID_KW)
    sns.despine(ax=axA)

    # Panel B: ladder of reference points
    labels = ["Observed\n(fire deltas)", "Pre-emb\nfloor", "Post-emb\nfloor",
              "Shuffle\nnull", "Isotropic\nrandom"]
    vals = [obs_mean, pre_floor, post_floor, float(perm.mean()), float(rand.mean())]
    colors = [ps.C_HIST, ps.C_FIRE_3, ps.C_FIRE_4, ps.C_LOG_1, "0.6"]
    axB.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(vals):
        axB.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=ps.FS_ANNOT)
    axB.set_ylim(min(0, min(vals) - 0.05), max(vals) + 0.12)
    axB.set_title("B. Reference comparison", fontsize=ps.FS_PANEL, fontweight="bold")
    axB.set_ylabel("Mean pairwise cosine", fontsize=ps.FS_LABEL)
    axB.tick_params(axis="x", labelsize=ps.FS_TICK - 1)
    axB.grid(True, axis="y", **ps.GRID_KW)
    sns.despine(ax=axB)

    fig.tight_layout(pad=ps.FIG_PAD)
    ps.save_figure(fig, os.path.join(pc.FIGURES_OUT, "null_baselines.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
