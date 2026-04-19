import pandas as pd
import sumolib
import random
from pyproj import Transformer

# ===== 設定區 =====
NET_FILE = "nyc.net.xml"                 # 你的地圖檔
CENTROID_FILE = "taxi_zone_centroids.csv" # 座標檔
PRED_FILE = "pred_next_hour_advanced.csv" # 預測結果檔
OUTPUT_TRIPS = "trips.rou.xml"            # 輸出的車流檔
# =================

def main():
    print(f"1. 讀取路網 {NET_FILE} ...")
    try:
        net = sumolib.net.readNet(NET_FILE)
    except Exception as e:
        print(f"錯誤：讀取路網失敗，請確認檔案存在。({e})")
        return

    # 取得地圖上所有道路的 ID (用來當作隨機目的地)
    all_edge_ids = [e.getID() for e in net.getEdges()]
    print(f"   地圖共有 {len(all_edge_ids)} 條道路")

    print("2. 讀取 CSV 資料...")
    df_cent = pd.read_csv(CENTROID_FILE)
    df_pred = pd.read_csv(PRED_FILE)

    # 合併資料
    if 'PULocationID' in df_pred.columns:
        left_key = 'PULocationID'
    else:
        left_key = 'LocationID'
    
    df = df_pred.merge(df_cent, left_on=left_key, right_on="LocationID", how="inner")
    
    # 建立座標轉換器 (如果需要的話)
    # 這裡假設 CSV 裡已經是經緯度 (lon, lat)
    # 如果不是，請自行調整 Transformer
    
    print("3. 產生車流檔案...")
    vehicle_count = 0
    
    with open(OUTPUT_TRIPS, "w", encoding="utf-8") as f:
        f.write('<routes>\n')
        # 定義車子的樣子 (黃色計程車)
        f.write('    <vType id="taxi" accel="2.6" decel="4.5" sigma="0.5" length="4.5" minGap="2.5" maxSpeed="50" color="1,0.8,0"/>\n')

        for idx, row in df.iterrows():
            try:
                # 1. 取得該區的預測叫車量
                demand = int(row.get('pred_rides', 0))
                if demand <= 0:
                    continue

                # 為了避免畫面爆炸，如果數量太大，可以除以 10 (例如 demand // 10)
                # 這裡我們先全畫，如果太卡再調整
                
                # 2. 找到該區域中心點最近的「道路」
                lon, lat = row['lon'], row['lat']
                x, y = net.convertLonLat2XY(lon, lat)
                
                # 搜尋半徑 200 公尺內的最近道路
                edges = net.getNeighboringEdges(x, y, 200)
                
                if len(edges) == 0:
                    # 找不到路 (可能該區在海裡或地圖外)，跳過
                    continue
                
                # 取最近的一條路作為起點
                start_edge = edges[0][0].getID()

                # 3. 產生車輛
                for i in range(demand):
                    veh_id = f"veh_{int(row['LocationID'])}_{i}"
                    # 隨機目的地
                    end_edge = random.choice(all_edge_ids)
                    # 隨機出發時間 (0 ~ 3600秒之間)
                    depart_time = random.randint(0, 3600)
                    
                    # 寫入行程
                    f.write(f'    <trip id="{veh_id}" type="taxi" depart="{depart_time}" from="{start_edge}" to="{end_edge}"/>\n')
                    vehicle_count += 1

            except Exception as e:
                continue

        f.write('</routes>\n')

    print(f"✅ 成功！已產生 {OUTPUT_TRIPS}")
    print(f"   總共生成了 {vehicle_count} 輛計程車")
    print("請進行下一步：執行 SUMO")

if __name__ == "__main__":
    main()