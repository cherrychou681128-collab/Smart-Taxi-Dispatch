import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_t24convlstm/best.pt"
NPZ_PATH = "test_t24.npz"
SAVE_NPZ_PATH = "plot_data.npz"

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        self.hid_ch = hid_ch
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

if not os.path.exists(NPZ_PATH):
    print(f"錯誤：找不到資料檔案 {NPZ_PATH}")
else:
    data = np.load(NPZ_PATH)
    X_raw = data["X"].astype(np.float32)
    Y_raw = data["y"].astype(np.float32)
    
    X = torch.from_numpy(X_raw).to(DEVICE)
    Y = torch.from_numpy(Y_raw).to(DEVICE)
    print(f"[INFO] 載入資料成功: X={X.shape}, Y={Y.shape}")

    model = ConvLSTM().to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤：找不到權重檔案 {MODEL_PATH}")
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"[INFO] 成功載入模型權重: {MODEL_PATH}")

        with torch.no_grad():
            pred = model(X)

        y_true_np = Y.cpu().numpy()
        y_pred_np = pred.cpu().numpy()

        rmse = np.sqrt(mean_squared_error(y_true_np.reshape(-1), y_pred_np.reshape(-1)))
        mae = mean_absolute_error(y_true_np.reshape(-1), y_pred_np.reshape(-1))

        print(f"\n======= [ConvLSTM Evaluation] =======")
        print(f"測試樣本數: {len(Y)}")
        print(f"RMSE = {rmse:.4f}")
        print(f"MAE  = {mae:.4f}")
        print("=====================================")

        print(f"[INFO] 正在將結果儲存至 {SAVE_NPZ_PATH}...")
        np.savez(SAVE_NPZ_PATH, y_true=y_true_np, y_pred=y_pred_np)
        print("[SUCCESS] 檔案已產生。")
