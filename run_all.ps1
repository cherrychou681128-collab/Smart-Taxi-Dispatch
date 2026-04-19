$PY = ".\.venv_cuda\Scripts\python.exe"
# powershell -ExecutionPolicy Bypass -File .\run_all.ps1
function RunPy {
  param(
    [Parameter(Mandatory=$true)][string]$Script,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ArgsList
  )
  $cmdShow = "$Script " + ($ArgsList -join " ")
  Write-Host "[RUN] $PY $cmdShow"
  & $PY $Script @ArgsList
  if ($LASTEXITCODE -ne 0) {
    throw "Python failed (exit=$LASTEXITCODE): $cmdShow"
  }
}


# -------------------------
# 1/12 XGB baseline（一次定義 residual）
# -------------------------
Write-Host "[1/12] predict_xgb_baseline_to_csv.py"
RunPy ".\train\predict_xgb_baseline_to_csv.py"

# -------------------------
# 2/12  唯一 residual 定義 
# -------------------------
Write-Host "[2/12] make_residual_from_xgb.py"
RunPy ".\train\make_residual_from_xgb.py"
# -------------------------
# 3/12 Residual dataset（給 ConvLSTM）
# -------------------------
Write-Host "[3/12] build_convlstm_residual_dataset.py"
RunPy ".\train\build_convlstm_dataset.py"
# -------------------------
# 4/12 Residual dataset（給 ConvLSTM）
# -------------------------
Write-Host "[4/12] fit_residual_scaler_per_zone.py"
RunPy ".\train\fit_residual_scaler_per_zone.py"
# -------------------------
# 5/12 Residual dataset（給 ConvLSTM）
# -------------------------
Write-Host "[5/12] build_convlstm_residual_dataset_perzone_scaled.py"
RunPy ".\train\build_convlstm_residual_dataset_perzone_scaled.py"
# -------------------------
# 6/12 ConvLSTM
# -------------------------
Write-Host "[6/12] ConvLSTM"
RunPy ".\train\train_convlstm_residual_map_masked_huber.py" 
# -------------------------
# 7/12 ConvLSTM
# -------------------------
Write-Host "[7/12]  predict_convlstm_to_zonecsv.py"
RunPy ".\train\predict_convlstm_to_zonecsv.py"

# -------------------------
# 8/12  alpha_and_blend
# -------------------------
Write-Host "[8/12] alpha_and_blend"
RunPy ".\train\optimize_alpha_and_blend.py"
# -------------------------
# 9/12  融合與評估
# -------------------------
Write-Host "[9/12]eval_xgb_plus_convlstm_residual.py"
RunPy ".\train\eval_xgb_plus_convlstm_residual.py"
# -------------------------
# 10/12  Transformer資料處理
# -------------------------
Write-Host "[10/12] build dataset"
RunPy ".\train\build_transformer_fusion_dataset.py"
# -------------------------
# 11/12  Transformer訓練
# -------------------------
Write-Host "[11/12]train\train_transformer"
RunPy ".\train\train_fusion_transformer.py"
# -------------------------
# 12/12 eval
# -------------------------
Write-Host "[12/12]評估"
RunPy ".\train\eval_all.py"









# # -------------------------
# # 5/9 Predict residuals
# # -------------------------
# Write-Host "[5/9] Predict residuals using ConvLSTM"
# RunPy ".\train\predict_convlstm_residual_to_zonecsv.py"

# # -------------------------
# # 6/9 (略) 你可以補前處理或其他任務
# # -------------------------

# # -------------------------
# # 7/9 Optimize alpha + blend
# # -------------------------
# Write-Host "[7/9] Optimize alpha + blend (using *_inv.csv)"
# RunPy ".\train\optimize_alpha_and_blend.py" 