import shapefile
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

sf = shapefile.Reader("data/taxi_zones.shp")

fields_name = [field[0] for field in sf.fields[1:]]
attributes = sf.records()
shp_attr = [dict(zip(fields_name, attr)) for attr in attributes]

polygons = [Polygon(s.points) for s in sf.shapes()]

df_zone = pd.DataFrame(shp_attr)
df_zone["geometry"] = polygons
gdf = gpd.GeoDataFrame(df_zone, geometry="geometry")

gdf["lon"] = gdf.geometry.centroid.x
gdf["lat"] = gdf.geometry.centroid.y

gdf = gdf[["LocationID", "zone", "borough", "lat", "lon"]]

import numpy as np

GRID_H, GRID_W = 32, 32

lon_min, lon_max = gdf["lon"].min(), gdf["lon"].max()
lat_min, lat_max = gdf["lat"].min(), gdf["lat"].max()

def latlon_to_grid(lat, lon):
    gx = int((lon - lon_min) / (lon_max - lon_min) * (GRID_W - 1))
    gy = int((lat_max - lat) / (lat_max - lat_min) * (GRID_H - 1))
    return gx, gy

loc_to_xy = {}
for _, row in gdf.iterrows():
    x, y = latlon_to_grid(row["lat"], row["lon"])
    loc_to_xy[int(row["LocationID"])] = (x, y)
    
