import os
import numpy as np
import pandas as pd
import shapefile
from datetime import datetime
from typing import Dict, Tuple

# ========== 基本設定 ==========
GRID_H, GRID_W = 32, 32
SEQ_OPTIONS = [24, 72, 168]  # 對應 1日、3日、7日序列
ZONE_COL = "PULocationID"
TIME_COL = "pickup_hour"
TARGET_COL = "rides"

# ========== 載入 shapefile 並建立 LocationID → (x,y) 對應表 ==========
def build_loc_to_xy(shp_path: str, shx_path: str, dbf_path: str) -> Dict[int, Tuple[int, int]]:
    sf = shapefile.Reader(shp=shp_path, shx=shx_path, dbf=dbf_path)
    fields_name = [field[0] for field in sf.fields[1:]]
    attributes = sf.records()
    shp_attr = [dict(zip(fields_name, attr)) for attr in attributes]

    rows = []
    for attr, shape_rec in zip(shp_attr, sf.shapes()):
        xmin, ymin, xmax, ymax = shape_rec.bbox
        lon = (xmin + xmax) / 2
        lat = (ymin + ymax) / 2
        rows.append({
            "LocationID": int(attr["LocationID"]),
            "lat": lat,
            "lon": lon
        })

    gdf = pd.DataFrame(rows)
    lon_min, lon_max = gdf["lon"].min(), gdf["lon"].max()
    lat_min, lat_max = gdf["lat"].min(), gdf["lat"].max()

    def latlon_to_grid(lat, lon):
        # 使用 clip 確保座標不會超出 0~31 範圍
        gx = int(np.clip((lon - lon_min) / (lon_max - lon_min) * (GRID_W - 1), 0, GRID_W - 1))
        gy = int(np.clip((lat_max - lat) / (lat_max - lat_min) * (GRID_H - 1), 0, GRID_H - 1))
        return gx, gy

    loc_to_xy = {}
    for _, row in gdf.iterrows():
        x, y = latlon_to_grid(row["lat"], row["lon"])
        loc_to_xy[int(row["LocationID"])] = (y, x) # 注意：矩陣索引通常是 (row, col) 即 (y, x)

    return loc_to_xy

# ========== 建立 Grid 資料並儲存為 .npz ==========
def hourly_df_to_grid_npz(df: pd.DataFrame, loc_to_xy: Dict[int, Tuple[int, int]], prefix: str):
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    
    # --- 優化：時間補洞 ---
    # 建立該資料段完整的時間範圍，缺失的小時補 0
    full_range = pd.date_range(start=df[TIME_COL].min(), end=df[TIME_COL].max(), freq='h')
    ts_all = full_range.tolist()
    
    # 預先將 DataFrame 轉換為 dict 提高查詢速度
    # 使用 pivot_table 解決多個區域映射到同一 Grid 的問題（先按時間與座標聚合）
    df['grid_pos'] = df[ZONE_COL].map(loc_to_xy)
    
    df = df.dropna(subset=['grid_pos']) # 移除沒有座標對應的區域
    
    # 建立時間步對應的 Grid 字典
    ts_to_grid = {}
    for t, group in df.groupby(TIME_COL):
        grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        for _, row in group.iterrows():
            y, x = row['grid_pos']
            grid[y, x] += row[TARGET_COL] # 使用 += 防止數值覆蓋
        ts_to_grid[t] = grid

    # 確保 ts_to_grid 包含所有時間點（沒資料的補全零矩陣）
    empty_grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    
    for SEQ_LEN in SEQ_OPTIONS:
        X_list, y_list = [], []
        
        # 滑動視窗
        for i in range(len(ts_all) - SEQ_LEN):
            t_seq = ts_all[i : i + SEQ_LEN]
            t_next = ts_all[i + SEQ_LEN]
            
            # 從字典拿 Grid，拿不到就給空矩陣
            x_seq = np.stack([ts_to_grid.get(t, empty_grid) for t in t_seq], axis=0) # (T, H, W)
            target = ts_to_grid.get(t_next, empty_grid) # (H, W)
            
            # 增加 Channel 維度並存入 List
            X_list.append(x_seq[None, :, :, :]) # (1, T, H, W)
            y_list.append(target[None, :, :])    # (1, H, W)
        
        if not X_list:
            print(f"數據量不足以產生 SEQ_LEN={SEQ_LEN} 的序列")
            continue

        X = np.stack(X_list).astype(np.float32) # (N, 1, T, H, W)
        y = np.stack(y_list).astype(np.float32) # (N, 1, H, W)
        
        # 調整維度符合 ConvLSTM 常用格式 (N, T, C, H, W)
        # 假設 C=1 (單一特徵：乘車人數)
        X = np.transpose(X, (0, 2, 1, 3, 4)) 
        
        out_path = f"{prefix}_t{SEQ_LEN}.npz"
        np.savez_compressed(out_path, X=X, y=y)
        print(f"[SAVED] {out_path} : X {X.shape} | y {y.shape}")

# ========== 主流程 ==========
if __name__ == "__main__":
    # 設定檔案路徑
    base_path = "data"
    shp_file = os.path.join(base_path, "taxi_zones.shp")
    shx_file = os.path.join(base_path, "taxi_zones.shx")
    dbf_file = os.path.join(base_path, "taxi_zones.dbf")
    
    if not os.path.exists(shp_file):
        print("錯誤：找不到 Shapefile，請檢查路徑。")
    else:
        # 1. 建立對照表
        loc_mapping = build_loc_to_xy(shp_file, shx_file, dbf_file)

        # 2. 處理資料集
        for split in ["train", "valid", "test"]:
            file_path = os.path.join(base_path, f"{split}_hourly.parquet")
            if os.path.exists(file_path):
                print(f"\n--- 開始處理 {split} 資料集 ---")
                data_df = pd.read_parquet(file_path)
                hourly_df_to_grid_npz(data_df, loc_mapping, prefix=split)
            else:
                print(f"跳過 {split}：檔案不存在。")