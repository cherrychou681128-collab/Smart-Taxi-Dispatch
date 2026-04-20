$PY = ".\.venv_cuda\Scripts\python.exe"

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

Write-Host "[1/12] predict_xgb_baseline_to_csv.py"
RunPy ".\train\predict_xgb_baseline_to_csv.py"

Write-Host "[2/12] make_residual_from_xgb.py"
RunPy ".\train\make_residual_from_xgb.py"

Write-Host "[3/12] build_convlstm_residual_dataset.py"
RunPy ".\train\build_convlstm_dataset.py"

Write-Host "[4/12] fit_residual_scaler_per_zone.py"
RunPy ".\train\fit_residual_scaler_per_zone.py"

Write-Host "[5/12] build_convlstm_residual_dataset_perzone_scaled.py"
RunPy ".\train\build_convlstm_residual_dataset_perzone_scaled.py"

Write-Host "[6/12] ConvLSTM"
RunPy ".\train\train_convlstm_residual_map_masked_huber.py" 

Write-Host "[7/12]  predict_convlstm_to_zonecsv.py"
RunPy ".\train\predict_convlstm_to_zonecsv.py"

Write-Host "[8/12] alpha_and_blend"
RunPy ".\train\optimize_alpha_and_blend.py"

Write-Host "[9/12]eval_xgb_plus_convlstm_residual.py"
RunPy ".\train\eval_xgb_plus_convlstm_residual.py"

Write-Host "[10/12] build dataset"
RunPy ".\train\build_transformer_fusion_dataset.py"

Write-Host "[11/12]train\train_transformer"
RunPy ".\train\train_fusion_transformer.py"

Write-Host "[12/12]評估"
RunPy ".\train\eval_all.py"
