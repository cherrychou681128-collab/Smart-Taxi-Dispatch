import json
import xml.etree.ElementTree as ET
from collections import defaultdict
import sumolib

def fcd_to_demo_json(net_xml, fcd_xml, out_json, max_vehicles=80, max_points_per_vehicle=2500):
    # 讀路網，用來把 x,y 轉 lon,lat
    net = sumolib.net.readNet(net_xml)

    vehicles = defaultdict(list)
    seen_order = []  # preserve first-seen order to cap max_vehicles

    ctx = ET.iterparse(fcd_xml, events=("start", "end"))
    _, root = next(ctx)

    current_t = 0.0

    for event, elem in ctx:
        if event == "start" and elem.tag == "timestep":
            current_t = float(elem.attrib.get("time", "0"))

        elif event == "end" and elem.tag == "vehicle":
            vid = elem.attrib.get("id")
            if not vid:
                elem.clear()
                continue

            # 限制車輛數，避免 JSON 爆掉
            if vid not in vehicles:
                if len(seen_order) >= max_vehicles:
                    elem.clear()
                    continue
                seen_order.append(vid)

            x = elem.attrib.get("x")
            y = elem.attrib.get("y")
            speed = elem.attrib.get("speed")  # m/s
            angle = elem.attrib.get("angle")  # deg

            if x is None or y is None:
                elem.clear()
                continue

            x = float(x); y = float(y)
            lon, lat = net.convertXY2LonLat(x, y)

            p = {
                "t": current_t,
                "lat": float(lat),
                "lng": float(lon),
                "speed_mps": float(speed) if speed is not None else 0.0,
                "angle": float(angle) if angle is not None else 0.0,
            }

            vehicles[vid].append(p)

            # 限制點數
            if max_points_per_vehicle and len(vehicles[vid]) > max_points_per_vehicle:
                vehicles[vid] = vehicles[vid][-max_points_per_vehicle:]

            elem.clear()

        elif event == "end" and elem.tag == "timestep":
            root.clear()

    out = {"vehicles": {vid: {"points": vehicles[vid]} for vid in seen_order}}

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"✅ wrote {out_json}  vehicles={len(seen_order)}")

if __name__ == "__main__":
    # 例：python tools/fcd_to_demo_json.py
    fcd_to_demo_json(
        net_xml="sumo/nyc.net.xml",
        fcd_xml="sumo/fcd.xml",
        out_json="src/public/sumo_traces/demo.json",  # 你可改成你的路徑
        max_vehicles=80,
        max_points_per_vehicle=2500,
    )
