import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DS_PATH = "train_t24.npz"
VAL_PATH = "valid_t24.npz"
CKPT_DIR = "best_t24convlstm"

BATCH = 16
EPOCHS = 100
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PEAK_Q = 0.90
ALPHA = 10.0
P_POWER = 1.2
EPS = 1e-6

BETA  = 0.6
GAMMA = 0.5

os.makedirs(CKPT_DIR, exist_ok=True)

class NpzGridDataset(Dataset):
    def __init__(self, npz_path):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"找不到檔案: {npz_path}")
        d = np.load(npz_path)
        self.X = d["X"].astype(np.float32)
        self.y = d["y"].astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=p)

    def forward(self, x, h, c):
        comb = torch.cat([x, h], dim=1)
        gates = self.conv(comb)
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

def compute_thr(npz_path, q):
    y = np.load(npz_path)["y"].reshape(-1)
    nz = y[y > 0]
    return float(np.quantile(nz, q)) if len(nz) else 1.0

def peak_weight(y, thr):
    ratio = torch.clamp(y / (thr + EPS), min=1.0)
    return 1.0 + ALPHA * (ratio ** P_POWER - 1.0)

def hybrid_loss(pred, y, w):
    mse_all  = ((pred - y) ** 2).mean()
    mse_peak = (((pred - y) ** 2) * w).mean()
    return BETA * mse_peak + (1 - BETA) * mse_all

@torch.no_grad()
def eval_metrics(model, loader, thr):
    model.eval()
    all_preds, all_ys = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        all_preds.append(pred.cpu())
        all_ys.append(y.cpu())

    preds = torch.cat(all_preds)
    ys = torch.cat(all_ys)

    mae_all = torch.abs(preds - ys).mean().item()
    rmse_all = torch.sqrt(((preds - ys) ** 2).mean()).item()

    mask = ys >= thr
    if mask.any():
        mae_peak = torch.abs(preds[mask] - ys[mask]).mean().item()
        rmse_peak = torch.sqrt(((preds[mask] - ys[mask]) ** 2).mean()).item()
    else:
        mae_peak, rmse_peak = mae_all, rmse_all

    return mae_all, rmse_all, mae_peak, rmse_peak

def main():
    train_ld = DataLoader(NpzGridDataset(DS_PATH), BATCH, shuffle=True)
    valid_ld = DataLoader(NpzGridDataset(VAL_PATH), BATCH, shuffle=False)

    thr = compute_thr(DS_PATH, PEAK_Q)
    print(f"[INFO] 裝置: {DEVICE} | Peak 門檻: {thr:.2f}")

    model = ConvLSTM().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_score = float("inf")
    patience, bad = 12, 0

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0
        for x, y in train_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            w = peak_weight(y, thr)
            loss = hybrid_loss(pred, y, w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += loss.item()

        m_all, r_all, m_peak, r_peak = eval_metrics(model, valid_ld, thr)
        
        score = (r_all * 0.4) + (r_peak * 0.6) 

        print(
            f"[E{ep:03d}] Loss:{loss_sum/len(train_ld):.3f} | "
            f"RMSE(All/Peak):{r_all:.3f}/{r_peak:.2f} | "
            f"MAE(All/Peak):{m_all:.3f}/{m_peak:.2f} | "
            f"Score:{score:.3f} | {time.time()-t0:.1f}s"
        )

        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), f"{CKPT_DIR}/best.pt")
            print(f" >>> [SAVED] Best Score: {best_score:.4f}")
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("--- Early Stopping ---")
                break

if __name__ == "__main__":
    main()
