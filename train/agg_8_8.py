import os
import torch
import numpy as np
import xgboost as xgb
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =============================
# 1. ConvLSTM 模型架構
# =============================
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=p)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

class ConvLSTM(nn.Module):
    def __init__(self, in_ch=1, hid_ch=32, layers=2):
        super().__init__()
        self.hid_ch = hid_ch
        self.layers = nn.ModuleList([
            ConvLSTMCell(in_ch if i == 0 else hid_ch, hid_ch)
            for i in range(layers)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(hid_ch, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        h = [torch.zeros(B, self.hid_ch, H, W, device=x.device) for _ in self.layers]
        c = [torch.zeros(B, self.hid_ch, H, W, device=x.device) for _ in self.layers]

        for t in range(T):
            inp = x[:, t]
            for i, cell in enumerate(self.layers):
                h[i], c[i] = cell(inp, h[i], c[i])
                inp = h[i]

        return self.head(h[-1])

# =============================
# 2. 核心功能函式
# =============================
def create_xgb_data(X_grid, y_grid, coords):
    N, T, C, H, W = X_grid.shape
    feats, targets = [], []
    for n in range(N):
        hour = n % 24
        for (y, x) in coords:
            history = X_grid[n, :, 0, y, x]
            f = np.concatenate([
                history,
                [np.mean(history), np.std(history), np.max(history)],
                [y, x, hour]
            ])
            feats.append(f)
            targets.append(y_grid[n, 0, y, x])
    return np.array(feats), np.array(targets)

def aggregate_32_to_8_batch(arr):
    """
    arr: (N, 1, 32, 32)
    return: (N, 1, 8, 8)
    """
    if arr.ndim != 4:
        raise ValueError(f"輸入維度錯誤: {arr.shape}，預期是 (N,1,32,32)")

    N, C, H, W = arr.shape
    if C != 1 or H != 32 or W != 32:
        raise ValueError(f"輸入 shape 錯誤: {arr.shape}，預期是 (N,1,32,32)")

    # 每個 4x4 block 做加總
    out = arr.reshape(N, C, 8, 4, 8, 4).sum(axis=(3, 5))
    return out

# =============================
# 3. 主程式
# =============================
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "best_t24convlstm/best.pt"
    TRAIN_NPZ = "train_t24.npz"
    TEST_NPZ = "test_t24.npz"

    SAVE_RESULT_PATH = "plot_data_hybrid_32x32.npz"   # 原始融合結果
    SAVE_AGG_PATH    = "plot_data_hybrid_8x8.npz"     # 聚合後結果

    coords = [(y, x) for y in range(32) for x in range(32)]

    print("[1/5] Loading Data...")
    train_data = np.load(TRAIN_NPZ)
    test_data = np.load(TEST_NPZ)
    X_test_np, y_test_np = test_data["X"], test_data["y"]

    print(f"[INFO] X_test shape: {X_test_np.shape}")
    print(f"[INFO] y_test shape: {y_test_np.shape}")

    # --- B. ConvLSTM 預測 ---
    print("[2/5] ConvLSTM Predicting...")
    model = ConvLSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        X_test_torch = torch.from_numpy(X_test_np).float().to(DEVICE)
        pred_conv = model(X_test_torch).cpu().numpy()

    print(f"[INFO] pred_conv shape: {pred_conv.shape}")

    # --- C. XGBoost 訓練 ---
    print("[3/5] Preparing XGBoost Features...")
    # 若記憶體夠，可把 [:500] 拿掉
    X_xgb_train, y_xgb_train = create_xgb_data(train_data["X"][:500], train_data["y"][:500], coords)
    X_xgb_test, y_xgb_test = create_xgb_data(X_test_np, y_test_np, coords)

    print("[4/5] Training XGBoost...")
    xgb_reg = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1
    )
    xgb_reg.fit(X_xgb_train, y_xgb_train)

    pred_xgb_flat = xgb_reg.predict(X_xgb_test)
    pred_xgb_grid = pred_xgb_flat.reshape(len(X_test_np), 1, 32, 32)

    print(f"[INFO] pred_xgb_grid shape: {pred_xgb_grid.shape}")

    # --- D. 權重融合 ---
    print("[5/5] Fusing, Aggregating & Saving Results...")
    ALPHA_WEIGHT = 0.7
    final_pred = (ALPHA_WEIGHT * pred_conv) + ((1 - ALPHA_WEIGHT) * pred_xgb_grid)

    # 避免負值
    final_pred = np.maximum(final_pred, 0)

    # ===== 先存原始 32x32 融合結果 =====
    np.savez(SAVE_RESULT_PATH, y_true=y_test_np, y_pred=final_pred)
    print(f"[DONE] 原始 32x32 融合結果已儲存至 {SAVE_RESULT_PATH}")

    # ===== 聚合成 8x8 =====
    y_true_agg = aggregate_32_to_8_batch(y_test_np)
    y_pred_agg = aggregate_32_to_8_batch(final_pred)

    print(f"[INFO] y_true_agg shape: {y_true_agg.shape}")
    print(f"[INFO] y_pred_agg shape: {y_pred_agg.shape}")

    # ===== 存聚合後結果 =====
    np.savez(SAVE_AGG_PATH, y_true=y_true_agg, y_pred=y_pred_agg)
    print(f"[DONE] 聚合後 8x8 結果已儲存至 {SAVE_AGG_PATH}")

    # ===== 數據指標計算：32x32 =====
    rmse_conv = np.sqrt(mean_squared_error(y_test_np.flatten(), pred_conv.flatten()))
    mae_conv = mean_absolute_error(y_test_np.flatten(), pred_conv.flatten())

    rmse_hybrid = np.sqrt(mean_squared_error(y_test_np.flatten(), final_pred.flatten()))
    mae_hybrid = mean_absolute_error(y_test_np.flatten(), final_pred.flatten())

    # ===== 數據指標計算：8x8 聚合後 =====
    rmse_hybrid_agg = np.sqrt(mean_squared_error(y_true_agg.flatten(), y_pred_agg.flatten()))
    mae_hybrid_agg = mean_absolute_error(y_true_agg.flatten(), y_pred_agg.flatten())

    print("\n======= Evaluation (32x32) =======")
    print(f"ConvLSTM RMSE : {rmse_conv:.4f}")
    print(f"ConvLSTM MAE  : {mae_conv:.4f}")
    print(f"Hybrid RMSE   : {rmse_hybrid:.4f}")
    print(f"Hybrid MAE    : {mae_hybrid:.4f}")
    print(f"Improvement   : {((rmse_conv - rmse_hybrid) / rmse_conv) * 100:.2f}%")
    print("==================================")

    print("\n======= Evaluation (8x8 Aggregated) =======")
    print(f"Hybrid RMSE   : {rmse_hybrid_agg:.4f}")
    print(f"Hybrid MAE    : {mae_hybrid_agg:.4f}")
    print("===========================================")