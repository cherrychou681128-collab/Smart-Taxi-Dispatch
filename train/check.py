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

    print("\n===== 2.時間洩漏檢查 =====")
    last_timestep_X = X_train[0, -1, 0] 
    target_y = y_train[0, 0]
    
    if np.array_equal(last_timestep_X, target_y):
        print(" 警告：偵測到嚴重洩漏！X 的最後一個時間步與 y 完全一致。")
        print("這代表模型在預測時直接看到了當下的答案。")
    else:
        print(" 通過：輸入特徵與預測目標無直接重疊。")

    print("\n===== 3. 訓練/測試集交叉污染檢查 =====")
    found_overlap = False
    for i in range(min(100, len(train_data['X']))):
        if np.array_equal(test_data['X'][0], train_data['X'][i]):
            found_overlap = True
            break
    
    if found_overlap:
        print(" 警告：測試集樣本出現在訓練集中！")
    else:
        print(" 通過：訓練集與測試集樣本無明顯重疊。")

def check_feature_importance(xgb_model, feature_names=None):
    print("\n===== 4. XGBoost 特徵貢獻度分析 =====")
    importance = xgb_model.get_fscore()
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print("前 5 大關鍵特徵：")
    for feat, score in importance[:5]:
        print(f"特徵 {feat}: 貢獻度 {score}")
    
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
    
    if y_pred_all.min() < -1:
        print(" 警告：預測值出現負數，這在需求預測中不符合物理邏輯。")
    
    plt.figure(figsize=(8, 5))
    sns.histplot(y_pred_all - y_true_flat, kde=True, bins=50)
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Residual Distribution (Prediction - Truth)")
    plt.xlabel("Error")
    plt.show()

if __name__ == "__main__":
    TRAIN_PATH = "train_t24.npz"
    TEST_PATH = "test_t24.npz"
    
    check_data_leakage(TRAIN_PATH, TEST_PATH)
