import os
import torch
import numpy as np
import xgboost as xgb
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
        self.layers = nn.ModuleList([ConvLSTMCell(in_ch if i==0 else hid_ch, hid_ch) for i in range(layers)])
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

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "best_t24convlstm/best.pt" 
    TRAIN_NPZ = "train_t24.npz"
    TEST_NPZ = "test_t24.npz"
    SAVE_RESULT_PATH = "plot_data.npz"

    coords = [(y, x) for y in range(32) for x in range(32)] 

    print("[1/5] Loading Data...")
    train_data = np.load(TRAIN_NPZ)
    test_data = np.load(TEST_NPZ)
    X_test_np, y_test_np = test_data["X"], test_data["y"]

    print("[2/5] ConvLSTM Predicting...")
    model = ConvLSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        X_test_torch = torch.from_numpy(X_test_np).float().to(DEVICE)
        pred_conv = model(X_test_torch).cpu().numpy()

    print("[3/5] Preparing XGBoost Features...")
    X_xgb_train, y_xgb_train = create_xgb_data(train_data["X"][:500], train_data["y"][:500], coords)
    X_xgb_test, y_xgb_test = create_xgb_data(X_test_np, y_test_np, coords)

    print("[4/5] Training XGBoost...")
    xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1)
    xgb_reg.fit(X_xgb_train, y_xgb_train)
    pred_xgb_flat = xgb_reg.predict(X_xgb_test)
    pred_xgb_grid = pred_xgb_flat.reshape(len(X_test_np), 1, 32, 32)

    print("[5/5] Fusing & Saving Results...")
    ALPHA_WEIGHT = 0.7
    final_pred = (ALPHA_WEIGHT * pred_conv) + ((1 - ALPHA_WEIGHT) * pred_xgb_grid)
    
    final_pred = np.maximum(final_pred, 0)
    np.savez(SAVE_RESULT_PATH, y_true=y_test_np, y_pred=final_pred)
    print(f"[DONE] 融合結果已儲存至 {SAVE_RESULT_PATH}")

    rmse_conv = np.sqrt(mean_squared_error(y_test_np.flatten(), pred_conv.flatten()))
    rmse_hybrid = np.sqrt(mean_squared_error(y_test_np.flatten(), final_pred.flatten()))

    print(f"\n======= Evaluation =======")
    print(f"ConvLSTM RMSE : {rmse_conv:.4f}")
    print(f"Hybrid RMSE   : {rmse_hybrid:.4f}")
    print(f"Improvement   : {((rmse_conv - rmse_hybrid)/rmse_conv)*100:.2f}%")
    print(f"==========================")

