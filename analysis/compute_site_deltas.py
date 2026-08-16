#!/usr/bin/env python
"""
compute_site_deltas.py
======================
Run Prithvi-EO-2.0-300M over the 50 wildfire sites once and cache the per-site
delta (post-fire minus pre-fire) embedding vectors to disk. This is the single
expensive step (it runs the model); the null analysis and figure regeneration
then load the cache and run in seconds.

Usage
-----
    python compute_site_deltas.py                 # uses code/data/50_Selected_sites
    python compute_site_deltas.py --sites DIR     # custom site directory
    python compute_site_deltas.py --print-sim     # also report mean pairwise cosine

Output
------
    code/data/cache/site_deltas.npz   (ids, deltas, std, pre/post means, n_resamples)

This reproduces notebook Cell 5; the cached deltas yield the published global mean
pairwise cosine similarity of ~0.813.
"""
import argparse

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import prithvi_common as pc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", default=pc.SITES_DIR, help="directory of *_pre/_post .tif files")
    ap.add_argument("--out", default=pc.DELTA_CACHE, help="output .npz cache path")
    ap.add_argument("--print-sim", action="store_true", help="report mean pairwise cosine after computing")
    args = ap.parse_args()

    results = pc.compute_site_deltas(sites_dir=args.sites)
    if not results:
        raise SystemExit("No site deltas computed; check the sites directory.")
    pc.save_delta_cache(results, path=args.out)

    if args.print_sim:
        traj = pc.l2_normalize_rows(np.vstack([r["delta"] for r in results]))
        sim = cosine_similarity(traj)
        pairwise = sim[np.triu_indices_from(sim, k=1)]
        print(f"\nSites: {len(results)}   pairs: {pairwise.size}")
        print(f"Mean pairwise cosine similarity: {pairwise.mean():.6f}")


if __name__ == "__main__":
    main()
