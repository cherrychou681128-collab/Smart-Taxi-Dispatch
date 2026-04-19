from pathlib import Path
import pandas as pd
import numpy as np
from pyproj import Transformer

# ===== Min-Max Norm（含全相等保護：全 0 → 全 0，符合「補 0」公平性）=====
def minmax_norm(s: pd.Series) -> pd.Series:
    s = s.fillna(0).astype(float)
    mn, mx = float(s.min()), float(s.max())
    if mx - mn < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

# ===== 用最近 centroid 把 311 點位配到 Taxi Zone =====
def nearest_zone_index(lat_arr, lon_arr, zones_lat, zones_lon):
    """
    回傳每個 311 點對應到 zones 的 index（最近 centroid）
    平面近似距離（NYC 範圍足夠）
    """
    lat_arr = np.asarray(lat_arr, dtype=float)
    lon_arr = np.asarray(lon_arr, dtype=float)
    zones_lat = np.asarray(zones_lat, dtype=float)
    zones_lon = np.asarray(zones_lon, dtype=float)

    cos_lat = np.cos(np.deg2rad(lat_arr))
    best = np.empty(len(lat_arr), dtype=int)

    batch = 8000
    for i in range(0, len(lat_arr), batch):
        sl = slice(i, min(i + batch, len(lat_arr)))
        dlat = lat_arr[sl, None] - zones_lat[None, :]
        dlon = (lon_arr[sl, None] - zones_lon[None, :]) * cos_lat[sl, None]
        dist2 = dlat * dlat + dlon * dlon
        best[sl] = dist2.argmin(axis=1)

    return best

def main():
    # =========================
    # ✅ 路徑：符合你現在專案結構
    # =========================
    BASE = Path(__file__).resolve().parent
    path_311  = BASE / "獎懲機制" / "nyc_311_2025_07.csv"
    path_cent = BASE / "data" / "taxi_zone_centroids.csv"
    out_path  = BASE / "outputs" / "zone_reward.csv"
    out_path.parent.mkdir(exist_ok=True)

    if not path_311.exists():
        raise FileNotFoundError(f"找不到 311 檔案：{path_311}")
    if not path_cent.exists():
        raise FileNotFoundError(f"找不到 centroid 檔案：{path_cent}")

    # =========================
    # 讀檔
    # =========================
    df311 = pd.read_csv(path_311)
    dfcent = pd.read_csv(path_cent)

    # =========================
    # centroid：EPSG:2263 → WGS84
    # taxi_zone_centroids.csv 欄位是 lat/lon 但其實是 2263 的座標 (X/Y)
    # =========================
    transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
    lon_wgs, lat_wgs = transformer.transform(dfcent["lon"].values, dfcent["lat"].values)
    dfcent["lon_wgs"] = np.round(lon_wgs, 6)
    dfcent["lat_wgs"] = np.round(lat_wgs, 6)

    # =========================
    # 311：時間處理（取「最新一小時」當作當下狀態）
    # 欄位名依你檔案：created_date / latitude / longitude / complaint_type / descriptor
    # =========================
    df311["created_date"] = pd.to_datetime(df311["created_date"], errors="coerce")
    df311 = df311.dropna(subset=["created_date", "latitude", "longitude"]).copy()
    df311["hour_ts"] = df311["created_date"].dt.floor("H")

    latest_hour = df311["hour_ts"].max()
    df311 = df311[df311["hour_ts"] == latest_hour].copy()

    # =========================
    # 311 分類：需求面 D vs 阻礙面 C
    # 你組員的核心示例：
    # - 需求面：半夜開派對、音樂太大聲 → Noise 類、Loud Music/Party
    # - 阻礙面：違規停車、路霸擋路 → Illegal Parking、Blocked Driveway
    # =========================
    ct = df311["complaint_type"].astype(str).str.lower()
    desc = df311["descriptor"].astype(str).str.lower()

    # 需求面（正向）
    demand_mask = (
        ct.str.startswith("noise") |
        desc.str.contains(r"party|loud music|music|loud", regex=True)
    )

    # 阻礙面（負向）
    constraint_mask = (
        ct.str.contains(r"illegal parking|blocked driveway|street condition|traffic", regex=True) |
        desc.str.contains(r"blocked|no access|double parked|obstruction|road", regex=True)
    )

    df311["DC"] = np.where(demand_mask, "D", np.where(constraint_mask, "C", ""))
    df311 = df311[df311["DC"] != ""].copy()

    # =========================
    # 將 311 點位配到最近 Taxi Zone
    # =========================
    zones_lat = dfcent["lat_wgs"].to_numpy()
    zones_lon = dfcent["lon_wgs"].to_numpy()

    idx = nearest_zone_index(
        df311["latitude"].to_numpy(),
        df311["longitude"].to_numpy(),
        zones_lat, zones_lon
    )

    df311["PULocationID"] = dfcent["LocationID"].iloc[idx].to_numpy()

    # =========================
    # 彙整：Taxi Zone × (D/C)
    # =========================
    g = df311.groupby(["PULocationID", "DC"]).size().unstack(fill_value=0)
    if "D" not in g.columns: g["D"] = 0
    if "C" not in g.columns: g["C"] = 0
    g = g.reset_index()[["PULocationID", "D", "C"]]

    # =========================
    # 公平性：沒報案的 Zone 補 0（疑責從無）
    # =========================
    all_zones = pd.DataFrame({"PULocationID": dfcent["LocationID"].astype(int)})
    z = all_zones.merge(g, on="PULocationID", how="left")
    z["D"] = z["D"].fillna(0).astype(int)
    z["C"] = z["C"].fillna(0).astype(int)

    # =========================
    # DRS：alpha*Norm(D) - beta*Norm(C)
    # =========================
    alpha, beta = 0.7, 0.3
    z["D_norm"] = minmax_norm(z["D"])
    z["C_norm"] = minmax_norm(z["C"])
    z["DRS"] = alpha * z["D_norm"] - beta * z["C_norm"]

    # 給地圖好用的 0~100 分（DRS 再 norm 一次）
    z["final_score"] = (minmax_norm(z["DRS"]) * 100).round(2)

    # 記錄本次使用的時段（方便你報告）
    z["used_hour"] = str(latest_hour)

    # 輸出
    z.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("OK ->", out_path)
    print("used hour:", latest_hour)

if __name__ == "__main__":
    main()
