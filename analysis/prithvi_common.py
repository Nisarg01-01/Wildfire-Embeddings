"""
prithvi_common.py
=================
Shared infrastructure for the Prithvi-EO-2.0 wildfire embedding probe, refactored
out of `Prithvi_EO_2_Embedding_Analysis.ipynb` into a plain importable module that
runs LOCALLY (no Colab, no Google Drive, no Hugging Face download).

It provides, in one place:
  * local paths (derived from this file's location; override with env vars)
  * the paper figure style + save_figure()
  * Prithvi-EO-2.0-300M model loading (from the local checkpoint/config)
  * the core patch-embedding / delta-vector computation functions
  * compute_site_deltas(): the 50-site delta pass, with on-disk caching

The compute functions are faithful ports of notebook Cells 3-5 (same seeds, same
greedy-thinning, same L2 normalisation), so cached deltas reproduce the published
mean pairwise cosine of 0.813.

Requirements: torch, numpy, rasterio, scikit-learn, matplotlib, seaborn, tqdm.
The Prithvi weights (Prithvi_EO_V2_300M.pt), config.json and prithvi_mae.py must
sit next to this file (they already do in this repo).
"""

from __future__ import annotations

import glob
import json
import os
import random
import warnings
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------- #
# Paths  (local layout; override any of these with an environment variable)
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))


def _path(env_key, *default_parts):
    return os.environ.get(env_key, os.path.join(HERE, *default_parts))


CKPT_PATH      = _path("PRITHVI_CKPT",   "Prithvi_EO_V2_300M.pt")
CONFIG_PATH    = _path("PRITHVI_CONFIG", "config.json")
SITES_DIR      = _path("SITES_DIR",      "data", "50_Selected_sites")
TEMPORAL_DIR   = _path("TEMPORAL_DIR",   "data", "Temporal_Data", "temporal_analysis")
LOGGING_DIR    = _path("LOGGING_DIR",    "data", "Logging_Data")
CSV_PATH       = _path("SITES_CSV",      "data", "50_Selected_sites.csv")
FIGURES_OUT    = _path("FIGURES_OUT",    "data", "Figures")
CACHE_DIR      = _path("CACHE_DIR",      "data", "cache")

os.makedirs(FIGURES_OUT, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Model / patch constants  (identical to notebook Cell 2)
# --------------------------------------------------------------------------- #
NIR_IDX           = 3
SWIR2_IDX         = 5
MODEL_INPUT_SIZE  = 224
PATCH_SIZE        = 64
PATCH_HALF        = PATCH_SIZE // 2
NUM_PATCHES       = 16
PREPOST_RESAMPLES = 20
EPSILON           = 1e-12
RANDOM_SEED       = 42

# Site IDs used by the temporal (Fig. 3) and selectivity (Fig. 4) analyses
TEMPORAL_SITE_IDS   = ["gf_24179489", "gf_21043348", "gf_26577816"]
FIRE_COMPARE_IDS    = ["gf_26577816", "gf_24587934"]
LOGGING_COMPARE_IDS = ["log_2018_romania_toplita", "log_2022_cardamoms"]

SITE_LABELS = {
    "gf_21043348": "United States of America 2018",
    "gf_24179489": "Argentina 2021",
    "gf_26577816": "Algeria 2023",
    "gf_24587934": "Greece 2021",
    "log_2018_romania_toplita": "Romania 2018",
    "log_2022_cardamoms":       "Cardamom Mts. 2022",
}

# These globals are filled in by load_model(); the compute functions read them.
PRITHVI_MEAN: np.ndarray | None = None
PRITHVI_STD:  np.ndarray | None = None
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("rasterio").setLevel(logging.ERROR)
np.seterr(divide="ignore", invalid="ignore", over="ignore")


# --------------------------------------------------------------------------- #
# Determinism  (notebook Cell 2)
# --------------------------------------------------------------------------- #
def set_global_determinism(seed: int = RANDOM_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------- #
# Model loading  (notebook Cell 3, local checkpoint)
# --------------------------------------------------------------------------- #
def load_model():
    """Load Prithvi-EO-2.0-300M in single-frame inference mode from local files.

    Populates module globals MODEL, PRITHVI_MEAN, PRITHVI_STD and returns MODEL.
    Idempotent: a second call returns the already-loaded model.
    """
    global MODEL, PRITHVI_MEAN, PRITHVI_STD
    if MODEL is not None:
        return MODEL

    for p in (CKPT_PATH, CONFIG_PATH, os.path.join(HERE, "prithvi_mae.py")):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required model file missing: {p}\n"
                "Place Prithvi_EO_V2_300M.pt, config.json and prithvi_mae.py "
                "next to prithvi_common.py."
            )

    from prithvi_mae import PrithviMAE  # local module

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)["pretrained_cfg"]

    PRITHVI_MEAN = np.array(cfg["mean"], dtype=np.float32) / 10000.0
    PRITHVI_STD  = np.array(cfg["std"],  dtype=np.float32) / 10000.0

    valid_keys = {
        "img_size", "patch_size", "num_frames", "in_chans",
        "embed_dim", "depth", "num_heads",
        "decoder_embed_dim", "decoder_depth", "decoder_num_heads",
        "mlp_ratio", "norm_pix_loss", "coords_encoding",
        "coords_scale_learn", "drop_path", "mask_ratio",
    }
    model_cfg = {k: v for k, v in cfg.items() if k in valid_keys}
    model_cfg["num_frames"] = 1  # single-image mode (pre and post passed separately)

    model = PrithviMAE(**model_cfg)
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    # Drop fixed temporal pos-embeddings (pretrained with num_frames=4); they are
    # re-initialised for single-frame inference. All other weights transfer.
    state_dict = {k: v for k, v in state_dict.items() if "pos_embed" not in k}
    model.load_state_dict(state_dict, strict=False)

    MODEL = model.to(DEVICE).eval()
    print(f"Loaded Prithvi-EO-2.0-300M on {DEVICE}.")
    return MODEL


# --------------------------------------------------------------------------- #
# Core computation  (notebook Cell 4, verbatim logic)
# --------------------------------------------------------------------------- #
def load_hls(path: str, normalize: bool = False) -> np.ndarray:
    """Read a 6-band HLS GeoTIFF (0-1 float reflectance). normalize=True applies
    Prithvi pretraining normalisation (for the encoder); False keeps raw
    reflectance (for spectral indices)."""
    with rasterio.open(path) as src:
        nodata = src.nodata
        data = src.read(list(range(1, 7))).astype(np.float32)
        if nodata is not None:
            if np.isneginf(nodata) or np.isposinf(nodata) or np.isnan(nodata):
                data[~np.isfinite(data)] = np.nan
            else:
                data[data == nodata] = np.nan
    data = np.nan_to_num(data, nan=0.0)
    if normalize:
        data = (data - PRITHVI_MEAN[:, None, None]) / PRITHVI_STD[:, None, None]
    return data


def get_nbr(img: np.ndarray) -> np.ndarray:
    num = img[NIR_IDX] - img[SWIR2_IDX]
    den = img[NIR_IDX] + img[SWIR2_IDX] + 1e-10
    return num / den


def compute_dnbr(pre_path: str, post_path: str):
    pre  = load_hls(pre_path,  normalize=False)
    post = load_hls(post_path, normalize=False)
    return get_nbr(pre) - get_nbr(post), pre, post


def l2_normalize_rows(arr, eps: float = EPSILON) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


def select_valid_patch_coords(h, w, num_patches=NUM_PATCHES, seed=RANDOM_SEED):
    """Jittered-grid patch centres (logging / non-fire sites)."""
    rng = np.random.default_rng(seed)
    margin = PATCH_HALF
    if h <= 2 * margin or w <= 2 * margin:
        return []
    max_jitter = PATCH_SIZE // 4
    y_starts = np.arange(margin, h - margin, PATCH_SIZE)
    x_starts = np.arange(margin, w - margin, PATCH_SIZE)
    coords = []
    for y in y_starts:
        for x in x_starts:
            jy = int(rng.integers(-max_jitter, max_jitter + 1))
            jx = int(rng.integers(-max_jitter, max_jitter + 1))
            cy = int(np.clip(y + jy, margin, h - margin - 1))
            cx = int(np.clip(x + jx, margin, w - margin - 1))
            coords.append((cy, cx))
    rng.shuffle(coords)
    return coords[:num_patches]


def select_burn_patch_coords(dnbr, num_patches=NUM_PATCHES, dnbr_lo=0.15, seed=RANDOM_SEED):
    """Greedy spatially-thinned patch centres drawn from burned pixels (dNBR>lo)."""
    rng = np.random.default_rng(seed)
    h, w = dnbr.shape
    margin = PATCH_HALF
    y_idx, x_idx = np.where(dnbr > dnbr_lo)
    valid = ((y_idx >= margin) & (y_idx < h - margin) &
             (x_idx >= margin) & (x_idx < w - margin))
    y_idx, x_idx = y_idx[valid], x_idx[valid]
    if len(y_idx) == 0:
        return []
    order = rng.permutation(len(y_idx))
    y_idx, x_idx = y_idx[order], x_idx[order]
    sy, sx = [], []
    for cy, cx in zip(y_idx.tolist(), x_idx.tolist()):
        if len(sy) >= num_patches:
            break
        if not any(abs(cy - a) < PATCH_SIZE and abs(cx - b) < PATCH_SIZE
                   for a, b in zip(sy, sx)):
            sy.append(cy)
            sx.append(cx)
    return list(zip(sy, sx))


def embed_patch(img_normalized, cy, cx):
    """Single patch -> 1024-d embedding (mean-pooled spatial tokens, CLS excluded)."""
    y1, y2 = cy - PATCH_HALF, cy + PATCH_HALF
    x1, x2 = cx - PATCH_HALF, cx + PATCH_HALF
    patch = img_normalized[:, y1:y2, x1:x2]
    if patch.shape[1] != PATCH_SIZE or patch.shape[2] != PATCH_SIZE:
        return None
    patch = np.ascontiguousarray(patch, dtype=np.float32)
    t = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)            # (1, 6, 64, 64)
    t = F.interpolate(t, size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                      mode="bilinear", align_corners=False)        # (1, 6, 224, 224)
    with torch.no_grad():
        out = MODEL.forward_features(t)
        final_tokens = out[-1] if isinstance(out, (list, tuple)) else out
        vec = final_tokens[:, 1:, :].mean(dim=1).detach().cpu().numpy().flatten()
    if np.isnan(vec).any() or np.isinf(vec).any():
        return None
    return vec.astype(np.float32)


def embed_paired_patches(img_pre, img_post, coords, normalize=True):
    pre_embs, post_embs = [], []
    for cy, cx in coords:
        pre_vec  = embed_patch(img_pre,  cy, cx)
        post_vec = embed_patch(img_post, cy, cx)
        if pre_vec is None or post_vec is None:
            continue
        if normalize:
            pre_vec  = pre_vec  / max(np.linalg.norm(pre_vec),  EPSILON)
            post_vec = post_vec / max(np.linalg.norm(post_vec), EPSILON)
        pre_embs.append(pre_vec)
        post_embs.append(post_vec)
    if not pre_embs:
        return None, None
    return np.vstack(pre_embs), np.vstack(post_embs)


def embed_patches(img, coords, normalize=True):
    """Embed multiple patches from a single (already-normalised) image.
    Returns (n_valid, 1024) or None. Used by the temporal analysis."""
    embs = [embed_patch(img, cy, cx) for cy, cx in coords]
    embs = [v for v in embs if v is not None]
    if not embs:
        return None
    embs = np.vstack(embs)
    if normalize:
        embs = l2_normalize_rows(embs)
    return embs


def _cap_patches(shape, num_patches):
    h, w = shape
    usable_h = h - 2 * PATCH_HALF
    usable_w = w - 2 * PATCH_HALF
    if usable_h <= 0 or usable_w <= 0:
        return 0
    max_non_overlapping = (usable_h // PATCH_SIZE) * (usable_w // PATCH_SIZE)
    return min(num_patches, max(1, max_non_overlapping))


def estimate_site_delta(pre_path, post_path, num_patches=NUM_PATCHES,
                        num_resamples=PREPOST_RESAMPLES, seed=RANDOM_SEED,
                        use_burn_coords=True):
    """Resampled mean delta vector for a pre/post pair (notebook estimate_site_delta).

    Returns dict {delta_mean, delta_std, n_resamples, delta_runs, pre_mean, post_mean}
    where pre_mean/post_mean are the per-patch-L2-normalised embeddings averaged over
    the first valid draw (used for the anisotropy floor in null_analysis.py), or None.
    """
    img_pre_norm  = load_hls(pre_path,  normalize=True)
    img_post_norm = load_hls(post_path, normalize=True)
    h, w = img_pre_norm.shape[1], img_pre_norm.shape[2]
    actual_patches = _cap_patches((h, w), num_patches)

    dnbr = None
    if use_burn_coords:
        dnbr, _, _ = compute_dnbr(pre_path, post_path)

    deltas = []
    rep_pre_emb, rep_post_emb = None, None   # full first-draw matrices (for t-SNE)
    for i in range(num_resamples):
        if use_burn_coords:
            coords = select_burn_patch_coords(dnbr, num_patches=actual_patches, seed=seed + i)
        else:
            coords = select_valid_patch_coords(h, w, num_patches=actual_patches, seed=seed + i)
        if not coords:
            continue
        pre_emb, post_emb = embed_paired_patches(img_pre_norm, img_post_norm, coords, normalize=True)
        if pre_emb is None or post_emb is None:
            continue
        deltas.append(post_emb.mean(0) - pre_emb.mean(0))
        if rep_pre_emb is None:
            rep_pre_emb, rep_post_emb = pre_emb, post_emb

    if not deltas:
        return None
    deltas = np.vstack(deltas)
    return {
        "delta_mean":  deltas.mean(axis=0),
        "delta_std":   deltas.std(axis=0),
        "n_resamples": len(deltas),
        "delta_runs":  deltas,
        "pre_mean":    rep_pre_emb.mean(0),
        "post_mean":   rep_post_emb.mean(0),
        "pre_emb":     rep_pre_emb,    # (n_patches, 1024) first valid draw
        "post_emb":    rep_post_emb,
    }


# --------------------------------------------------------------------------- #
# Site discovery + the 50-site delta pass, with caching
# --------------------------------------------------------------------------- #
DELTA_CACHE       = os.path.join(CACHE_DIR, "site_deltas.npz")        # Fig 2 / null
TEMPORAL_CACHE    = os.path.join(CACHE_DIR, "temporal_sim.csv")       # Fig 3
SELECTIVITY_CACHE = os.path.join(CACHE_DIR, "selectivity.npz")        # Fig 4


def discover_sites(sites_dir=SITES_DIR):
    """Group *_pre.tif / *_post.tif files by site id -> {country, pre, post}."""
    groups = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(sites_dir, "*.tif"))):
        parts = os.path.basename(f).replace(".tif", "").split("_")
        if len(parts) >= 4:
            site_id = f"{parts[0]}_{parts[1]}"
            groups[site_id]["country"] = parts[2]
            groups[site_id][parts[-1].lower()] = f
    return groups


def compute_site_deltas(sites_dir=SITES_DIR, seed=RANDOM_SEED, verbose=True):
    """Run the model over every site and return a list of per-site dicts:
    {id, country, delta, delta_std, pre_mean, post_mean, n_resamples}."""
    load_model()
    set_global_determinism(seed)
    groups = discover_sites(sites_dir)
    try:
        from tqdm import tqdm
        iterator = tqdm(sorted(groups), desc="Sites")
    except Exception:
        iterator = sorted(groups)

    results, failed = [], []
    for site_id in iterator:
        info = groups[site_id]
        if "pre" not in info or "post" not in info:
            failed.append((site_id, "missing pre/post"))
            continue
        stats = estimate_site_delta(info["pre"], info["post"], seed=seed)
        if stats is None:
            failed.append((site_id, "no valid embeddings"))
            continue
        results.append({
            "id": site_id,
            "country": info["country"],
            "delta": stats["delta_mean"],
            "delta_std": stats["delta_std"],
            "pre_mean": stats["pre_mean"],
            "post_mean": stats["post_mean"],
            "n_resamples": stats["n_resamples"],
        })
    if verbose:
        print(f"Computed deltas for {len(results)} sites; {len(failed)} failed.")
        for sid, msg in failed:
            print(f"  - {sid}: {msg}")
    return results


def save_delta_cache(results, path=DELTA_CACHE):
    np.savez_compressed(
        path,
        ids=np.array([r["id"] for r in results]),
        countries=np.array([r["country"] for r in results]),
        delta=np.vstack([r["delta"] for r in results]).astype(np.float32),
        delta_std=np.vstack([r["delta_std"] for r in results]).astype(np.float32),
        pre_mean=np.vstack([r["pre_mean"] for r in results]).astype(np.float32),
        post_mean=np.vstack([r["post_mean"] for r in results]).astype(np.float32),
        n_resamples=np.array([r["n_resamples"] for r in results]),
        meta=np.array([json.dumps({"seed": RANDOM_SEED,
                                    "num_patches": NUM_PATCHES,
                                    "resamples": PREPOST_RESAMPLES})]),
    )
    print(f"Saved delta cache -> {path}")


def load_delta_cache(path=DELTA_CACHE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No delta cache at {path}. Run compute_site_deltas.py first."
        )
    z = np.load(path, allow_pickle=True)
    return {
        "ids": [str(x) for x in z["ids"]],
        "countries": [str(x) for x in z["countries"]],
        "delta": z["delta"],
        "delta_std": z["delta_std"],
        "pre_mean": z["pre_mean"],
        "post_mean": z["post_mean"],
        "n_resamples": z["n_resamples"],
    }
