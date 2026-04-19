import shapefile
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

# 載入 shapefile
sf = shapefile.Reader("data/taxi_zones.shp")

# 取得欄位名稱
fields_name = [field[0] for field in sf.fields[1:]]
attributes = sf.records()
shp_attr = [dict(zip(fields_name, attr)) for attr in attributes]

# 每個 polygon 的 geometry
polygons = [Polygon(s.points) for s in sf.shapes()]

# 建立 DataFrame
df_zone = pd.DataFrame(shp_attr)
df_zone["geometry"] = polygons
gdf = gpd.GeoDataFrame(df_zone, geometry="geometry")

# 計算每個 LocationID 的中心點經緯度
gdf["lon"] = gdf.geometry.centroid.x
gdf["lat"] = gdf.geometry.centroid.y

gdf = gdf[["LocationID", "zone", "borough", "lat", "lon"]]

import numpy as np

GRID_H, GRID_W = 32, 32

# 將經緯度轉換成 0~1 區間（normalize）
lon_min, lon_max = gdf["lon"].min(), gdf["lon"].max()
lat_min, lat_max = gdf["lat"].min(), gdf["lat"].max()

def latlon_to_grid(lat, lon):
    gx = int((lon - lon_min) / (lon_max - lon_min) * (GRID_W - 1))
    gy = int((lat_max - lat) / (lat_max - lat_min) * (GRID_H - 1))  # 注意 Y 是反過來的
    return gx, gy

# 建立對應表：LocationID → (x, y)
loc_to_xy = {}
for _, row in gdf.iterrows():
    x, y = latlon_to_grid(row["lat"], row["lon"])
    loc_to_xy[int(row["LocationID"])] = (x, y)
    