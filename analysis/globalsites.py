import sys
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import pyogrio
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

# ====================== CONFIGURATION ======================
DATA_DIR = Path(r".\code\data\GLOBFIRE_Shapefiles")
SHP_GLOB = "*.shp"

YEAR_START = 2018
YEAR_END   = 2023

MIN_AREA_HA       = 2500
MIN_DURATION_DAYS = 3

GRID_DEG     = 10
TOP_PER_GRID = 5

PRE_MONTHS    = 2
PRE_GAP_DAYS  = 7
POST_GAP_DAYS = 7
POST_MONTHS   = 2

# ====================== HELPER FUNCTION ======================
def find_col(cols, variants):
    """Find column by name (case-insensitive). Different shapefiles use different column names."""
    cols_lower = {c.lower(): c for c in cols}
    for v in variants:
        if v.lower() in cols_lower:
            return cols_lower[v.lower()]
    return None

# ====================== STEP 1: LOAD SHAPEFILES ======================
shp_files = sorted(DATA_DIR.glob(SHP_GLOB))
print(f"Found {len(shp_files)} shapefiles: {[s.name for s in shp_files]}")

gdfs = []
for shp in shp_files:
    print(f"Processing: {shp.name}")
    try:
        file_year = int(shp.stem[-4:])
    except ValueError:
        continue

    if not (YEAR_START <= file_year <= YEAR_END):
        continue

    df = pyogrio.read_dataframe(shp)
    cols = list(df.columns)

    id_col    = find_col(cols, ["Id", "ID", "id", "id_fire", "fire_id"])
    idate_col = find_col(cols, ["IDate", "initialdat", "initialdate", "initial_date"])
    fdate_col = find_col(cols, ["FDate", "finaldate", "final_date"])

    if not all([id_col, idate_col, fdate_col]):
        print(f"  Skipped: missing required columns")
        continue

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.rename(columns={id_col: 'Id', idate_col: 'IDate', fdate_col: 'FDate'})
    gdf['Id'] = pd.to_numeric(gdf['Id'], errors='coerce').fillna(0).astype(int)
    gdfs.append(gdf[['Id', 'IDate', 'FDate', 'geometry']])

combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
combined_gdf["start_dt"] = pd.to_datetime(combined_gdf["IDate"], utc=True, errors='coerce')
combined_gdf["end_dt"]   = pd.to_datetime(combined_gdf["FDate"], utc=True, errors='coerce')
combined_gdf = combined_gdf.dropna(subset=["start_dt", "end_dt"])

# ====================== STEP 2: CALCULATE AREA & CENTROID ======================
# Centroid is computed once in Equal Earth projection (accurate for non-convex polygons)
# then reprojected back to WGS84 for lat/lon output.
print("Calculating fire areas and centroids...")

proj_gdf = combined_gdf.to_crs("+proj=eqearth +units=m")
combined_gdf["area_ha"] = proj_gdf.geometry.area / 10000.0

centroids_proj = proj_gdf.geometry.centroid
# Fallback: if centroid is outside its own polygon, use representative_point instead
safe_centroids = centroids_proj.copy()
outside_mask = ~proj_gdf.geometry.contains(centroids_proj)
if outside_mask.any():
    print(f"  {outside_mask.sum()} concave polygons: falling back to representative_point")
    safe_centroids[outside_mask] = proj_gdf.geometry[outside_mask].representative_point()

safe_centroids_wgs84 = safe_centroids.set_crs(proj_gdf.crs).to_crs("EPSG:4326")
combined_gdf["attr_InitialLongitude"] = safe_centroids_wgs84.x
combined_gdf["attr_InitialLatitude"]  = safe_centroids_wgs84.y

# ====================== STEP 3: FILTER BY SIZE, DURATION & YEAR ======================
print(f"Before filtering: {len(combined_gdf)} fires")

combined_gdf["duration_days"] = (
    (combined_gdf["end_dt"] - combined_gdf["start_dt"]).dt.total_seconds() / 86400.0
)
combined_gdf = combined_gdf[
    (combined_gdf["area_ha"] >= MIN_AREA_HA) &
    (combined_gdf["duration_days"] >= MIN_DURATION_DAYS) &
    (combined_gdf["start_dt"].dt.year >= YEAR_START) &
    (combined_gdf["start_dt"].dt.year <= YEAR_END)
].copy()

print(f"After filtering (>={MIN_AREA_HA} ha, >={MIN_DURATION_DAYS} days, {YEAR_START}-{YEAR_END}): {len(combined_gdf)} fires")

# ====================== STEP 4: IDENTIFY COUNTRIES ======================
# Two-stage spatial join: within-polygon first, nearest-neighbour fallback for
# coastal/border centroids that fall outside all polygons.
print("Identifying fire countries using Natural Earth boundary data...")

url = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
try:
    world = gpd.read_file(url)
except Exception as e:
    print(f"Could not fetch 50m world map: {e}. Falling back to 110m.")
    url_low = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    try:
        world = gpd.read_file(url_low)
    except Exception as e2:
        print(f"Could not fetch 110m world map: {e2}. Country assignment skipped.")
        world = None

if world is None:
    combined_gdf["country"] = "Unknown"
else:
    world.columns = [c.lower() for c in world.columns]
    world = world[['name', 'geometry']]

    points_gdf = gpd.GeoDataFrame(
        combined_gdf,
        geometry=gpd.points_from_xy(combined_gdf.attr_InitialLongitude, combined_gdf.attr_InitialLatitude),
        crs="EPSG:4326"
    )

    joined = gpd.sjoin(points_gdf, world, how="left", predicate="within")
    unknowns = joined[joined["name"].isna()]
    if not unknowns.empty:
        print(f"Resolving {len(unknowns)} coastal/border points using nearest-neighbour...")
        nearest = gpd.sjoin_nearest(
            unknowns.drop(columns=['name', 'index_right']), world, how="left", max_distance=0.5
        )
        joined.loc[unknowns.index, "name"] = nearest["name"]

    combined_gdf["country"] = (
        joined["name"].fillna("Unknown").str.replace(' ', '-').str.replace('_', '-')
    )

# ====================== STEP 5: GRID-BASED SAMPLING ======================
print("Applying grid-based sampling...")

lon_bin = np.floor(combined_gdf["attr_InitialLongitude"] / GRID_DEG).astype(int)
lat_bin = np.floor(combined_gdf["attr_InitialLatitude"]  / GRID_DEG).astype(int)
combined_gdf["grid_id"] = ["lat_%d_lon_%d" % (la, lo) for la, lo in zip(lat_bin, lon_bin)]

sampled_fires = []
for grid_id in combined_gdf["grid_id"].unique():
    grid_fires = combined_gdf[combined_gdf["grid_id"] == grid_id]
    grid_fires = grid_fires.sort_values("area_ha", ascending=False).head(TOP_PER_GRID)
    sampled_fires.append(grid_fires)

final_df = gpd.GeoDataFrame(pd.concat(sampled_fires, ignore_index=True), crs="EPSG:4326")

print(f"Grid sampling complete: {len(final_df)} fires from {len(sampled_fires)} grid cells")
print(f"Countries represented: {final_df['country'].nunique()}")

# ====================== STEP 6: SATELLITE TIME WINDOWS & EXPORT ======================
print("Calculating satellite observation windows...")

final_df["attr_IncidentName"] = "gf_" + final_df["Id"].astype(int).astype(str)

final_df["pre_start"]  = (final_df["start_dt"] - pd.DateOffset(months=PRE_MONTHS)).dt.strftime("%Y-%m-%d")
final_df["pre_end"]    = (final_df["start_dt"] - pd.Timedelta(days=PRE_GAP_DAYS)).dt.strftime("%Y-%m-%d")
final_df["post_start"] = (final_df["end_dt"]   + pd.Timedelta(days=POST_GAP_DAYS)).dt.strftime("%Y-%m-%d")
final_df["post_end"]   = (final_df["end_dt"]   + pd.DateOffset(months=POST_MONTHS)).dt.strftime("%Y-%m-%d")

out = final_df[[
    "attr_IncidentName",
    "country",
    "grid_id",
    "attr_InitialLongitude",
    "attr_InitialLatitude",
    "pre_start", "pre_end",
    "post_start", "post_end",
    "area_ha",
    "IDate", "FDate"
]]

out.to_csv(r".\code\data\Global_fire_incidents_Int.csv", index=False)
print(f"Export complete: {len(out)} fires saved to Global_fire_incidents_Int.csv")
print(f"  Grid cells : {out['grid_id'].nunique()}")
print(f"  Countries  : {out['country'].nunique()}")
