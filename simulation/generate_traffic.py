import pandas as pd
import sumolib
import random
from pyproj import Transformer

NET_FILE = "nyc.net.xml"
CENTROID_FILE = "taxi_zone_centroids.csv"
PRED_FILE = "pred_next_hour_advanced.csv"
OUTPUT_TRIPS = "trips.rou.xml"

def main():
    try:
        net = sumolib.net.readNet(NET_FILE)
    except Exception as e:
        print(f"error:failed")
        return

    all_edge_ids = [e.getID() for e in net.getEdges()]
    print(f"地圖共有 {len(all_edge_ids)} 條道路")

    df_cent = pd.read_csv(CENTROID_FILE)
    df_pred = pd.read_csv(PRED_FILE)

    if 'PULocationID' in df_pred.columns:
        left_key = 'PULocationID'
    else:
        left_key = 'LocationID'
    
    df = df_pred.merge(df_cent, left_on=left_key, right_on="LocationID", how="inner")
    
    vehicle_count = 0
    
    with open(OUTPUT_TRIPS, "w", encoding="utf-8") as f:
        f.write('<routes>\n')
        f.write('    <vType id="taxi" accel="2.6" decel="4.5" sigma="0.5" length="4.5" minGap="2.5" maxSpeed="50" color="1,0.8,0"/>\n')

        for idx, row in df.iterrows():
            try:
                demand = int(row.get('pred_rides', 0))
                if demand <= 0:
                    continue

                lon, lat = row['lon'], row['lat']
                x, y = net.convertLonLat2XY(lon, lat)
                
                edges = net.getNeighboringEdges(x, y, 200)
                
                if len(edges) == 0:
                    continue

                start_edge = edges[0][0].getID()

                for i in range(demand):
                    veh_id = f"veh_{int(row['LocationID'])}_{i}"
                    end_edge = random.choice(all_edge_ids)
                    depart_time = random.randint(0, 3600)
                    
                    f.write(f'    <trip id="{veh_id}" type="taxi" depart="{depart_time}" from="{start_edge}" to="{end_edge}"/>\n')
                    vehicle_count += 1

            except Exception as e:
                continue

        f.write('</routes>\n')

    print(f"已產生 {OUTPUT_TRIPS}")
    print(f"共生成了 {vehicle_count} 輛計程車")

if __name__ == "__main__":
    main()
