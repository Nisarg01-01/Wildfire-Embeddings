# Wildfire Embedding Analysis (Shareable Repository)

This repository contains code, notebooks, and data helpers for wildfire embedding analysis using Prithvi-EO-2.0.

## Repository Layout

- `analysis/`: core Python analysis pipeline, model config, and data workspace
- `notebooks/`: exploratory and reproducibility notebooks
- `gee/`: Google Earth Engine scripts for site filtering and logging-site downloads

## Core Analysis Files

- `analysis/prithvi_common.py`: shared module for paths, model loading, embeddings, and cache utilities.
- `analysis/compute_site_deltas.py`: step 1 (expensive model pass), writes site deltas cache.
- `analysis/null_analysis.py`: step 2, computes anisotropy-aware null baselines.
- `analysis/compute_fig34_cache.py`: computes caches used by Figures 3 and 4.
- `analysis/regenerate_figures.py`: regenerates figures from caches.
- `analysis/make_graphical_abstract.py`: builds graphical abstract outputs.

## Quick Start

1. Create a Python 3.10+ environment.
2. Install dependencies listed below.
3. Place the model checkpoint in `analysis/Prithvi_EO_V2_300M.pt` (excluded from Git by default).
4. Populate required datasets under `analysis/data/`.

Run from inside `analysis/`:

```powershell
python compute_site_deltas.py --print-sim
python null_analysis.py
python compute_fig34_cache.py
python regenerate_figures.py
python make_graphical_abstract.py
```

For the core reproducibility path, run this minimum sequence:

```powershell
cd analysis
python compute_site_deltas.py --print-sim
python null_analysis.py
python regenerate_figures.py
```

## Minimal Python Dependencies

- torch
- numpy
- rasterio
- scikit-learn
- matplotlib
- seaborn
- tqdm
- pandas
- geopandas
- pyogrio
- shapely
- pillow

## Data and Large Assets

Large files are intentionally excluded from Git for easy sharing:

- model checkpoints (`*.pt`, `*.pth`)
- large geospatial inputs (`analysis/data/GLOBFIRE_Shapefiles/`)
- generated figures and caches

If you want a "full reproducibility snapshot", publish large artifacts with one of:

- GitHub Releases
- Git LFS
- Zenodo / OSF / Hugging Face datasets

## Notes

- Core Python scripts are kept together in `analysis/` so relative paths in the existing code continue to work without logic changes.
