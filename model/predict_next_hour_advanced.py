from pathlib import Path#處理路徑
import pandas as pd#資料表處理
import numpy as np
import xgboost as xgb
import folium#地圖
from folium.plugins import HeatMap
from pyproj import Transformer

# === 資料夾設定 ===
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

model_path = MODEL_DIR / "xgb_demand_poisson.model"
hourly_path = DATA_DIR / "test_hourly.parquet"
centroid_path = DATA_DIR / "taxi_zone_centroids.csv"

print("模型：", model_path)
print("每小時資料：", hourly_path)
print("centroid：", centroid_path)

# 1. 載入模型
model = xgb.Booster()
model.load_model(str(model_path))
print("✅ 模型載入完成")

# 2. 讀取歷史每小時資料
df = pd.read_parquet(hourly_path)
df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])

last_hour = df["pickup_hour"].max()
next_hour = last_hour + pd.Timedelta(hours=1)
print("最後資料時間：", last_hour)
print("預測時間（下一小時）：", next_hour)

# 3. 為「下一小時」建立特徵
rows = []
for loc_id, g in df.groupby("PULocationID"):
    g = g.sort_values("pickup_hour")
    g = g[g["pickup_hour"] <= last_hour]

    y = g["rides"].values
    if len(y) < 24:
        continue  # 不足 24 小時，跳過

    lag_1 = float(y[-1])
    lag_24 = float(y[-24])
    ma_3 = float(np.mean(y[-3:]))
    ma_24 = float(np.mean(y[-24:]))

    dow = next_hour.dayofweek
    is_weekend = 1 if dow >= 5 else 0
    hour = next_hour.hour

    rows.append({
        "PULocationID": loc_id,
        "hour": hour,
        "dow": dow,
        "is_weekend": is_weekend,
        "lag_1": lag_1,
        "lag_24": lag_24,
        "ma_3": ma_3,
        "ma_24": ma_24,
        "predict_hour": next_hour
    })

df_feat = pd.DataFrame(rows)
print("可預測的地區數量：", len(df_feat))
if df_feat.empty:
    raise RuntimeError("沒有任何地區有足夠資料可預測")

feature_cols = [
    "PULocationID",
    "hour",
    "dow",
    "is_weekend",
    "lag_1",
    "lag_24",
    "ma_3",
    "ma_24",
]

dtest = xgb.DMatrix(df_feat[feature_cols])
pred = model.predict(dtest, validate_features=False)
df_feat["pred_rides"] = pred

# 4. 輸出預測 CSV 到 outputs/
csv_path = OUT_DIR / "pred_next_hour_advanced.csv"
df_feat.to_csv(csv_path, index=False, encoding="utf-8-sig")
print("✅ 下一小時預測輸出：", csv_path)

# 5. 合併 centroid + 座標轉換 → 畫熱力圖
df_cent = pd.read_csv(centroid_path)

# EPSG:2263 → WGS84(EPSG:4326)
transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)

df_merged = df_feat.merge(
    df_cent[["LocationID", "lat", "lon"]],
    left_on="PULocationID",
    right_on="LocationID",
    how="left"
)

lons, lats = transformer.transform(df_merged["lon"].values, df_merged["lat"].values)
df_merged["lat_wgs"] = lats
df_merged["lon_wgs"] = lons

center_lat = df_merged["lat_wgs"].mean()
center_lon = df_merged["lon_wgs"].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

heat_data = df_merged[["lat_wgs", "lon_wgs", "pred_rides"]].dropna().values.tolist()
HeatMap(heat_data, radius=15, blur=10).add_to(m)

html_path = OUT_DIR / "pred_next_hour_advanced_heatmap.html"
m.save(html_path)
print("✅ 熱力圖輸出：", html_path)
