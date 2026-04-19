import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

# =========================
# 路徑設定
# =========================
NPZ_PATH = r"plot_data_hybrid_8x8.npz"
XML_PATH = r"trips.xml"

# =========================
# 美化 XML 輸出
# =========================
def prettify(elem):
    rough_string = ET.tostring(elem, encoding="utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ", encoding="utf-8")

# =========================
# 檢查檔案
# =========================
if not os.path.exists(NPZ_PATH):
    print(f"錯誤：找不到檔案 {NPZ_PATH}")
    exit()

# =========================
# 載入 npz
# =========================
data = np.load(NPZ_PATH)

print("[INFO] npz keys:", data.files)

if "y_true" not in data or "y_pred" not in data:
    print("錯誤：npz 內必須包含 y_true 和 y_pred")
    exit()

y_true = data["y_true"]
y_pred = data["y_pred"]

print("[INFO] y_true shape:", y_true.shape)
print("[INFO] y_pred shape:", y_pred.shape)

# 預期 shape = (N, 1, 8, 8)
if y_true.ndim != 4 or y_pred.ndim != 4:
    print("錯誤：資料維度不對，預期為 (N,1,8,8)")
    exit()

N, C, H, W = y_true.shape
if C != 1 or H != 8 or W != 8:
    print(f"錯誤：y_true shape={y_true.shape}，預期為 (N,1,8,8)")
    exit()

# =========================
# 建立 XML
# =========================
root = ET.Element("demandData")
root.set("gridSize", "8x8")
root.set("samples", str(N))

for t in range(N):
    time_elem = ET.SubElement(root, "timeStep")
    time_elem.set("index", str(t))

    for gx in range(H):
        for gy in range(W):
            cell_elem = ET.SubElement(time_elem, "cell")
            cell_elem.set("x", str(gx))
            cell_elem.set("y", str(gy))
            cell_elem.set("y_true", str(float(y_true[t, 0, gx, gy])))
            cell_elem.set("y_pred", str(float(y_pred[t, 0, gx, gy])))

# =========================
# 寫入 XML
# =========================
xml_bytes = prettify(root)

with open(XML_PATH, "wb") as f:
    f.write(xml_bytes)

print(f"[SUCCESS] XML 已輸出：{XML_PATH}")