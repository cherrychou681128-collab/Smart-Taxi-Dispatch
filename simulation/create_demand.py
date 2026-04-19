import os
import sys
import random
import sumolib

# ==========================================
# 這裡就是你的「固定變因」控制台！
NUM_TAXIS = 20         # 計程車數量：20 台
NUM_PASSENGERS = 1000   # 乘客需求：1000 人
SIM_TIME = 3600        # 模擬時間：3600 秒 (1小時)
# ==========================================

# 1. 讀取你剛剛生成的網格地圖
sys.path.append(r"C:\Program Files (x86)\Eclipse\Sumo\tools")
net = sumolib.net.readNet('grid.net.xml')
# 抓出所有正常的道路 (排除路口內部)
edges = [e.getID() for e in net.getEdges() if not e.getFunction() == 'internal']

with open("experiment_demand.xml", "w", encoding="utf-8") as f:
    f.write('<routes>\n')
    
    # 2. 定義「計程車」職業
    f.write('    <vType id="taxi" vClass="taxi" modes="taxi" personCapacity="4">\n')
    f.write('        <param key="has.taxi.device" value="true"/>\n')
    f.write('        <param key="device.taxi.idle-algorithm" value="randomCircling"/>\n')
    f.write('    </vType>\n\n')

    # 3. 生成計程車隊 
    for i in range(NUM_TAXIS):
        start_edge = random.choice(edges)
        f.write(f'    <vehicle id="taxi_{i}" type="taxi" depart="0.00" departPos="random" line="taxi">\n')
        f.write(f'        <route edges="{start_edge}"/>\n')
        f.write('    </vehicle>\n\n')

    # 4. 生成乘客需求 
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
        # 這裡是最乾淨安全的寫法，保證不會報錯！
        f.write(f'    <person id="person_{i}" depart="{depart_time}">\n')
        f.write(f'        <ride from="{start_edge}" to="{end_edge}" lines="taxi"/>\n')
        f.write('    </person>\n')

    f.write('</routes>\n')

print("✅ 成功產生固定實驗數據：experiment_demand.xml (絕對安全版)")