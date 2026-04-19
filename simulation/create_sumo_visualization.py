import pandas as pd
import sumolib
from pyproj import Transformer
from pathlib import Path

# ===== 設定區 (檔名要對) =====
NET_FILE = "nyc.net.xml"               # 你的地圖檔
CENTROID_FILE = "taxi_zone_centroids.csv" # 座標檔
PRED_FILE = "pred_next_hour_advanced.csv" # 預測結果檔
OUTPUT_ADD_FILE = "visualization.add.xml" # 輸出的 SUMO 外掛檔
# ===========================

def get_color(score):
    """依照分數決定顏色 (R,G,B)"""
    # 這裡依照你的模型分數調整，假設分數 0~1 或 0~100
    # 這裡假設是 0.0 ~ 1.0 的機率，如果你的分數很大，請自己調整閾值
    if score > 0.5:
        return "1,0,0"       # 紅色 (高需求)
    elif score > 0.2:
        return "1,0.5,0"     # 橘色
    elif score > 0.05:
        return "1,1,0"       # 黃色
    else:
        return "0,1,0"       # 綠色 (低需求)

def main():
    print(f"1. 讀取路網 {NET_FILE} ...")
    try:
        net = sumolib.net.readNet(NET_FILE)
    except Exception as e:
        print(f"錯誤：找不到 {NET_FILE}，請確認檔案在同一個資料夾。")
        return

    print("2. 讀取 CSV 資料...")
    try:
        df_cent = pd.read_csv(CENTROID_FILE)
        df_pred = pd.read_csv(PRED_FILE)
    except Exception as e:
        print(f"錯誤：讀取 CSV 失敗 ({e})，請確認檔名是否正確。")
        return

    # 合併資料
    # 確保兩邊都有 LocationID (或 PULocationID)
    if 'PULocationID' in df_pred.columns:
        left_key = 'PULocationID'
    else:
        left_key = 'LocationID' # 依你的欄位名稱調整

    df = df_pred.merge(df_cent, left_on=left_key, right_on="LocationID", how="inner")
    
    print(f"   合併後共有 {len(df)} 筆資料")

    print("3. 座標轉換 & 產生 XML...")
    # 建立轉換器：紐約長島座標 (EPSG:2263) -> 經緯度 (EPSG:4326)
    transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)

    with open(OUTPUT_ADD_FILE, "w", encoding="utf-8") as f:
        f.write('<additional>\n')
        
        for idx, row in df.iterrows():
            try:
                # 1. 轉經緯度
                lon, lat = transformer.transform(row['lon'], row['lat'])
                
                # 2. 轉 SUMO 座標 (x,y)
                x, y = net.convertLonLat2XY(lon, lat)
                
                # 3. 取得分數
                score = row.get('pred_rides', 0) # 假設預測欄位叫 pred_rides
                
                # 4. 寫入 XML
                color = get_color(score)
                radius = 30 + (score * 50) # 半徑隨分數變大
                
                f.write(f'    <poi id="zone_{int(row["LocationID"])}" '
                        f'type="taxi_zone" '
                        f'color="{color}" '
                        f'x="{x:.2f}" y="{y:.2f}" '
                        f'width="{radius:.2f}" height="{radius:.2f}" '
                        f'layer="100"/>\n')
            except Exception:
                continue
                
        f.write('</additional>\n')
    
    print(f"✅ 成功！已產生 {OUTPUT_ADD_FILE}")
    print("請進行第三步：開啟 SUMO")

if __name__ == "__main__":
    main()