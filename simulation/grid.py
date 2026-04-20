import numpy as np
import xml.etree.ElementTree as ET
import random

NPZ_PATH = "plot_data_hybrid_8x8.npz"
OUTPUT_XML = "grid.xml"

data = np.load(NPZ_PATH)
y_pred = data["y_pred"]

N = y_pred.shape[0]

root = ET.Element("trips")

trip_id = 0

for t in range(N):
    for x in range(8):
        for y in range(8):
            demand = int(y_pred[t, 0, x, y])

            for i in range(demand):
                trip = ET.SubElement(root, "trip")

                trip.set("id", f"trip_{trip_id}")
                trip.set("depart", str(t))

                from_edge = f"edge_{x}_{y}"
                to_edge = f"edge_{random.randint(0,7)}_{random.randint(0,7)}"

                trip.set("from", from_edge)
                trip.set("to", to_edge)

                trip_id += 1

tree = ET.ElementTree(root)
tree.write(OUTPUT_XML)

print("grid.xml 產生完成")
