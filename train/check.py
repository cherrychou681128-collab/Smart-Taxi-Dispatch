import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os

def check_data_leakage(train_path, test_path):
    print("===== 1. 數據檔案完整性檢查 =====")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(" 錯誤：找不到 .npz 檔案，請確認路徑。")
        return

    train_data = np.load(train_path)
    test_data = np.load(test_path)
    
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']

    print(f"訓練集形狀: X={X_train.shape}, y={y_train.shape}")
    print(f"測試集形狀: X={X_test.shape}, y={y_test.shape}")

    print("\n===== 2. 時間洩漏檢查 (Look-ahead Bias) =====")
    # 檢查 X 的最後一個時間步是否與 y 相同
    # 假設 X 形狀為 (N, T, C, H, W)
    last_timestep_X = X_train[0, -1, 0] 
    target_y = y_train[0, 0]
    
    if np.array_equal(last_timestep_X, target_y):
        print(" 警告：偵測到嚴重洩漏！X 的最後一個時間步與 y 完全一致。")
        print("這代表模型在預測時直接看到了當下的答案。")
    else:
        print(" 通過：輸入特徵與預測目標無直接重疊。")

    print("\n===== 3. 訓練/測試集交叉污染檢查 =====")
    # 檢查測試集的第一筆資料是否出現在訓練集中
    # 這能防止資料切分時發生隨機洗牌（Shuffle）導致的時間軸錯亂
    found_overlap = False
    for i in range(min(100, len(train_data['X']))): # 抽樣檢查
        if np.array_equal(test_data['X'][0], train_data['X'][i]):
            found_overlap = True
            break
    
    if found_overlap:
        print(" 警告：測試集樣本出現在訓練集中！")
    else:
        print(" 通過：訓練集與測試集樣本無明顯重疊。")

def check_feature_importance(xgb_model, feature_names=None):
    print("\n===== 4. XGBoost 特徵貢獻度分析 =====")
    # 如果某個特徵權重異常高（例如 0.99），通常代表該特徵有洩漏疑慮
    importance = xgb_model.get_fscore()
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print("前 5 大關鍵特徵：")
    for feat, score in importance[:5]:
        print(f"特徵 {feat}: 貢獻度 {score}")
    
    # 繪圖
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(xgb_model, max_num_features=10)
    plt.title("XGBoost Feature Importance")
    plt.show()

def check_prediction_distribution(y_true, y_pred):
    print("\n===== 5. 預測數值合理性分佈 =====")
    y_true_flat = y_true.flatten()
    y_pred_all = y_pred.flatten()

    print(f"真實值區間: [{y_true_flat.min():.2f}, {y_true_flat.max():.2f}]")
    print(f"預測值區間: [{y_pred_all.min():.2f}, {y_pred_all.max():.2f}]")
    
    if y_pred_all.min() < -1: # 容許微小的浮點數誤差
        print(" 警告：預測值出現負數，這在需求預測中不符合物理邏輯。")
    
    # 繪製殘差分佈
    plt.figure(figsize=(8, 5))
    sns.histplot(y_pred_all - y_true_flat, kde=True, bins=50)
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Residual Distribution (Prediction - Truth)")
    plt.xlabel("Error")
    plt.show()

# --- 執行檢查 ---
if __name__ == "__main__":
    # 請根據你的實際檔案路徑修改
    TRAIN_PATH = "train_t24.npz"
    TEST_PATH = "test_t24.npz"
    
    check_data_leakage(TRAIN_PATH, TEST_PATH)
    
    # 如果你已經有跑完的結果，可以傳入進行分佈檢查：
    # check_prediction_distribution(y_test_np, final_pred)
    # 如果有訓練好的 xgb_model：
    # check_feature_importance(xgb_reg)