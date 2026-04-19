import pandas as pd
import sumolib
from pyproj import Transformer
from pathlib import Path

NET_FILE = "nyc.net.xml"
CENTROID_FILE = "taxi_zone_centroids.csv"
PRED_FILE = "pred_next_hour_advanced.csv"
OUTPUT_ADD_FILE = "visualization.add.xml"

def get_color(score):
    """依照分數決定顏色 (R,G,B)"""
    if score > 0.5:
        return "1,0,0"
    elif score > 0.2:
        return "1,0.5,0"
    elif score > 0.05:
        return "1,1,0"
    else:
        return "0,1,0"
def main():
    try:
        net = sumolib.net.readNet(NET_FILE)
    except Exception as e:
        print(f"error:can't find{NET_FILE}")
        return

    try:
        df_cent = pd.read_csv(CENTROID_FILE)
        df_pred = pd.read_csv(PRED_FILE)
    except Exception as e:
        print(f"error:failed")
        return

    if 'PULocationID' in df_pred.columns:
        left_key = 'PULocationID'
    else:
        left_key = 'LocationID'

    df = df_pred.merge(df_cent, left_on=left_key, right_on="LocationID", how="inner")
    
    print(f" 合併後共有 {len(df)} 筆資料")

    transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)

    with open(OUTPUT_ADD_FILE, "w", encoding="utf-8") as f:
        f.write('<additional>\n')
        
        for idx, row in df.iterrows():
            try:
                lon, lat = transformer.transform(row['lon'], row['lat'])
                
                x, y = net.convertLonLat2XY(lon, lat)
                
                score = row.get('pred_rides', 0)
                
                color = get_color(score)
                radius = 30 + (score * 50)
                
                f.write(f'    <poi id="zone_{int(row["LocationID"])}" '
                        f'type="taxi_zone" '
                        f'color="{color}" '
                        f'x="{x:.2f}" y="{y:.2f}" '
                        f'width="{radius:.2f}" height="{radius:.2f}" '
                        f'layer="100"/>\n')
            except Exception:
                continue
                
        f.write('</additional>\n')
    
    print(f"已產生 {OUTPUT_ADD_FILE}")

if __name__ == "__main__":
    main()
