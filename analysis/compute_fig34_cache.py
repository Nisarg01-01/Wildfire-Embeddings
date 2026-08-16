#!/usr/bin/env python
"""
compute_fig34_cache.py
======================
The one model pass needed for Figures 3 and 4. It runs Prithvi-EO-2.0-300M over
only the 7 tiles those figures use (3 temporal sites + 2 fire + 2 logging sites)
and caches the results so the figures can be replotted instantly afterwards
(legend relabels, combining panels, restyling) with no further model calls.

Outputs
-------
  data/cache/temporal_sim.csv   (Fig 3: cosine-to-reference per site/offset/window)
  data/cache/selectivity.npz    (Fig 4: per-site delta + first-draw patch embeddings)

Ports notebook Cell 8 (temporal) and Cell 9A (selectivity extraction). Fire year
and offset files are discovered by globbing the temporal folder, so no site CSV is
needed.

Usage
-----
    python compute_fig34_cache.py
"""
import glob
import os
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import prithvi_common as pc


# --------------------------------------------------------------------------- #
# Figure 3: temporal cosine-to-reference curve (Cell 8)
# --------------------------------------------------------------------------- #
def _temporal_files(site_id):
    """Return {(year, stage): path} for a temporal site, discovered by glob."""
    out = {}
    pat = re.compile(rf"^{re.escape(site_id)}_.+_(\d{{4}})_(pre|post)\.tif$", re.I)
    for f in glob.glob(os.path.join(pc.TEMPORAL_DIR, f"{site_id}_*.tif")):
        m = pat.match(os.path.basename(f))
        if m:
            out[(int(m.group(1)), m.group(2).lower())] = f
    return out


def compute_temporal(site_ids=pc.TEMPORAL_SITE_IDS, seed=pc.RANDOM_SEED):
    rows = []
    for sid in site_ids:
        files = _temporal_files(sid)
        if not files:
            print(f"  {sid}: no temporal tiles found, skipping")
            continue
        years = sorted({y for (y, _) in files})
        fire_year = int(np.median(years))            # center of the ±2-year window
        offsets = [-2, -1, 0, 1, 2]

        # Patch coords fixed from the fire-year dNBR map (held across all years)
        pre0, post0 = files.get((fire_year, "pre")), files.get((fire_year, "post"))
        if pre0 is None or post0 is None:
            print(f"  {sid}: missing fire-year tiles, skipping")
            continue
        dnbr, _, _ = pc.compute_dnbr(pre0, post0)
        coords = pc.select_burn_patch_coords(
            dnbr, num_patches=pc._cap_patches(dnbr.shape, pc.NUM_PATCHES),
            dnbr_lo=0.15, seed=seed)
        if not coords:
            print(f"  {sid}: no burn coords, skipping")
            continue

        # Reference = earliest available leading (pre) window: year-2 then year-1
        ref = None
        for yr in (fire_year - 2, fire_year - 1):
            p = files.get((yr, "pre"))
            if p:
                emb = pc.embed_patches(pc.load_hls(p, normalize=True), coords, normalize=True)
                if emb is not None:
                    ref = emb.mean(0).astype(np.float32)
                    break
        if ref is None:
            print(f"  {sid}: no reference embedding, skipping")
            continue

        for off in offsets:
            yr = fire_year + off
            for stage in ("pre", "post"):
                p = files.get((yr, stage))
                if not p:
                    continue
                emb = pc.embed_patches(pc.load_hls(p, normalize=True), coords, normalize=True)
                if emb is None:
                    continue
                sim = float(cosine_similarity([ref], [emb.mean(0)])[0][0])
                rows.append({
                    "site": sid,
                    "label": pc.SITE_LABELS.get(sid, sid),
                    "offset": off,
                    # leading = pre-ignition window, trailing = post-extinction window
                    "window": "Leading" if stage == "pre" else "Trailing",
                    "sim": sim,
                })
        print(f"  {sid}: fire_year={fire_year}, {len([r for r in rows if r['site']==sid])} points")
    return rows


# --------------------------------------------------------------------------- #
# Figure 4: fire vs logging selectivity (Cell 9A)
# --------------------------------------------------------------------------- #
def _find_pair(folder, sid):
    clean = sid.replace(" ", "_")
    pre  = sorted(glob.glob(os.path.join(folder, f"*{clean}*[Pp][Rr][Ee]*.tif")))
    post = sorted(glob.glob(os.path.join(folder, f"*{clean}*[Pp][Oo][Ss][Tt]*.tif")))
    return (pre[0] if pre else None, post[0] if post else None)


def compute_selectivity(seed=pc.RANDOM_SEED):
    entries = []
    for sid in pc.FIRE_COMPARE_IDS:
        pre, post = _find_pair(pc.SITES_DIR, sid)
        entries.append((sid, "Fire", pre, post, True))
    for sid in pc.LOGGING_COMPARE_IDS:
        pre, post = _find_pair(pc.LOGGING_DIR, sid)
        entries.append((sid, "Logging", pre, post, False))

    recs = []
    for sid, kind, pre, post, use_burn in entries:
        if not pre or not post:
            print(f"  {kind} {sid}: missing tiles, skipping")
            continue
        stats = pc.estimate_site_delta(pre, post, seed=seed, use_burn_coords=use_burn)
        if stats is None:
            print(f"  {kind} {sid}: no valid patches, skipping")
            continue
        recs.append({
            "id": sid, "type": kind,
            "label": pc.SITE_LABELS.get(sid, sid),
            "delta": stats["delta_mean"].astype(np.float32),
            "delta_std": stats["delta_std"].astype(np.float32),
            "pre_emb": stats["pre_emb"].astype(np.float32),
            "post_emb": stats["post_emb"].astype(np.float32),
            "n_resamples": stats["n_resamples"],
        })
        print(f"  {kind} {sid} ({pc.SITE_LABELS.get(sid, sid)}): "
              f"resamples={stats['n_resamples']}, patches={stats['pre_emb'].shape[0]}")
    return recs


def save_selectivity(recs, path=pc.SELECTIVITY_CACHE):
    # Variable patch counts per site -> store object arrays.
    np.savez(
        path,
        ids=np.array([r["id"] for r in recs]),
        types=np.array([r["type"] for r in recs]),
        labels=np.array([r["label"] for r in recs]),
        deltas=np.vstack([r["delta"] for r in recs]).astype(np.float32),
        delta_stds=np.vstack([r["delta_std"] for r in recs]).astype(np.float32),
        n_resamples=np.array([r["n_resamples"] for r in recs]),
        pre_embs=np.array([r["pre_emb"] for r in recs], dtype=object),
        post_embs=np.array([r["post_emb"] for r in recs], dtype=object),
    )
    print(f"Saved selectivity cache -> {path}")


def main():
    import pandas as pd
    pc.load_model()
    pc.set_global_determinism()

    print("\n[Figure 3] temporal cosine-to-reference ...")
    rows = compute_temporal()
    if rows:
        pd.DataFrame(rows).to_csv(pc.TEMPORAL_CACHE, index=False)
        print(f"Saved temporal cache -> {pc.TEMPORAL_CACHE}")

    print("\n[Figure 4] fire vs logging selectivity ...")
    recs = compute_selectivity()
    if recs:
        save_selectivity(recs)
        # quick numeric summary (matches Cell 9D)
        fire = [r for r in recs if r["type"] == "Fire"]
        logg = [r for r in recs if r["type"] == "Logging"]
        if fire and logg:
            fd = pc.l2_normalize_rows(np.vstack([r["delta"] for r in fire]))
            ld = pc.l2_normalize_rows(np.vstack([r["delta"] for r in logg]))
            cross = cosine_similarity(fd, ld)
            intra = (cosine_similarity(fd, fd)[np.triu_indices(len(fire), k=1)].mean()
                     if len(fire) > 1 else float("nan"))
            print(f"  intra-class fire = {intra:.4f}; cross mean = {cross.mean():.4f}; "
                  f"range = {cross.min():.3f}-{cross.max():.3f}")


if __name__ == "__main__":
    main()
