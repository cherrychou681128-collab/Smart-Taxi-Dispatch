import os
import numpy as np
import pandas as pd

NPZ_PATH = r"plot_data_hybrid_8x8.npz"
CSV_PATH = r"demand_8x8.csv"

if not os.path.exists(NPZ_PATH):
    print(f"錯誤：找不到檔案 {NPZ_PATH}")
    exit()

data = np.load(NPZ_PATH)

print("[INFO] npz keys:", data.files)

if "y_true" not in data or "y_pred" not in data:
    print("錯誤：npz 檔案中必須包含 y_true 與 y_pred")
    exit()

y_true = data["y_true"]
y_pred = data["y_pred"]

print("[INFO] y_true shape:", y_true.shape)
print("[INFO] y_pred shape:", y_pred.shape)

if y_true.ndim != 4 or y_pred.ndim != 4:
    print("錯誤：y_true / y_pred 維度不對，預期為 4 維 (N,1,8,8)")
    exit()

N, C, H, W = y_true.shape

if C != 1 or H != 8 or W != 8:
    print(f"錯誤：y_true shape={y_true.shape}，預期為 (N,1,8,8)")
    exit()

N2, C2, H2, W2 = y_pred.shape
if (N, C, H, W) != (N2, C2, H2, W2):
    print("錯誤：y_true 與 y_pred shape 不一致")
    exit()

rows = []

for t in range(N):
    for x in range(H):
        for y in range(W):
            rows.append({
                "time_idx": t,
                "grid_x": x,
                "grid_y": y,
                "y_true": float(y_true[t, 0, x, y]),
                "y_pred": float(y_pred[t, 0, x, y]),
            })

df = pd.DataFrame(rows)

df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

print(f"[SUCCESS] 已輸出 CSV：{CSV_PATH}")
print(df.head(10))
print(f"[INFO] 總筆數: {len(df)}")
