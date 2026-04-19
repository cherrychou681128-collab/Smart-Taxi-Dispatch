import os
import sys
import random
import sumolib

NUM_TAXIS = 20
NUM_PASSENGERS = 1000
SIM_TIME = 3600

sys.path.append(r"C:\Program Files (x86)\Eclipse\Sumo\tools")
net = sumolib.net.readNet('grid.net.xml')
edges = [e.getID() for e in net.getEdges() if not e.getFunction() == 'internal']

with open("experiment_demand.xml", "w", encoding="utf-8") as f:
    f.write('<routes>\n')
    
    f.write('    <vType id="taxi" vClass="taxi" modes="taxi" personCapacity="4">\n')
    f.write('        <param key="has.taxi.device" value="true"/>\n')
    f.write('        <param key="device.taxi.idle-algorithm" value="randomCircling"/>\n')
    f.write('    </vType>\n\n')

    for i in range(NUM_TAXIS):
        start_edge = random.choice(edges)
        f.write(f'    <vehicle id="taxi_{i}" type="taxi" depart="0.00" departPos="random" line="taxi">\n')
        f.write(f'        <route edges="{start_edge}"/>\n')
        f.write('    </vehicle>\n\n')

    passengers = []
    for i in range(NUM_PASSENGERS):
        depart_time = random.randint(0, SIM_TIME)
        start_edge = random.choice(edges)
        end_edge = random.choice(edges)
        while start_edge == end_edge: 
            end_edge = random.choice(edges)
        passengers.append((depart_time, i, start_edge, end_edge))
    
    passengers.sort()

    for p in passengers:
        depart_time, i, start_edge, end_edge = p
        f.write(f'    <person id="person_{i}" depart="{depart_time}">\n')
        f.write(f'        <ride from="{start_edge}" to="{end_edge}" lines="taxi"/>\n')
        f.write('    </person>\n')

    f.write('</routes>\n')

print("成功產生固定實驗數據")
