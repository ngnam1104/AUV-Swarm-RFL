param(
    [string]$WorkspaceRoot = "D:\Documents\HUST\2022-2026\Research_Thesis\AUV-Swarm-RFL",
    [int]$Episodes = 50,
    [int]$M = 9
)

$ErrorActionPreference = "Stop"
Set-Location $WorkspaceRoot

$python = Join-Path $WorkspaceRoot ".venv\Scripts\python.exe"
$pythonUnbuffered = "`"$python`" -u"
if (-not (Test-Path $python)) {
    throw "Python executable not found at $python"
}

$logDir = Join-Path $WorkspaceRoot "results\logs"
if (-not (Test-Path $logDir)) {
    New-Item -Path $logDir -ItemType Directory | Out-Null
}

$pipelineLog = Join-Path $logDir "baselines_two_stage_pipeline.log"
$mediumStepLog = Join-Path $logDir "baselines_step_log_medium.txt"
$fullStepLog = Join-Path $logDir "baselines_step_log_full.txt"

function Run-Step {
    param(
        [string]$Name,
        [string]$Command,
        [string]$LogFile
    )

    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$stamp] START: $Name" -ForegroundColor Cyan
    Write-Host "[$stamp] CMD  : $Command" -ForegroundColor DarkGray
    "[$stamp] START: $Name" | Out-File -FilePath $LogFile -Append -Encoding utf8
    "[$stamp] CMD  : $Command" | Out-File -FilePath $LogFile -Append -Encoding utf8

    # [ĐÃ SỬA]: Bắt từng dòng output, ép in ra terminal (Write-Host) và ghi vào file log
    & cmd /c $Command 2>&1 | ForEach-Object {
        Write-Host $_
        $_ | Out-File -FilePath $LogFile -Append -Encoding utf8
    }
    $exitCode = $LASTEXITCODE

    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    if ($exitCode -ne 0) {
        Write-Host "[$stamp] FAIL : $Name (exit=$exitCode)" -ForegroundColor Red
        "[$stamp] FAIL : $Name (exit=$exitCode)" | Out-File -FilePath $LogFile -Append -Encoding utf8
        throw "Step failed: $Name (exit=$exitCode)"
    }

    Write-Host "[$stamp] DONE : $Name" -ForegroundColor Green
    "[$stamp] DONE : $Name" | Out-File -FilePath $LogFile -Append -Encoding utf8
}

"==== Two-stage baseline pipeline started at $(Get-Date -Format "yyyy-MM-dd HH:mm:ss") ====" | Out-File -FilePath $pipelineLog -Append -Encoding utf8
"==== Medium step-log: $mediumStepLog ====" | Out-File -FilePath $pipelineLog -Append -Encoding utf8
"==== Full step-log  : $fullStepLog ====" | Out-File -FilePath $pipelineLog -Append -Encoding utf8

# --- BƯỚC 1: Thí nghiệm Beta Sensitivity (Figure 1, 2, 3) ---
Run-Step -Name "Experiment: Beta Sensitivity (Fig 1, 2, 3)" `
    -Command "$pythonUnbuffered scripts\eval_beta_sensitivity.py --rounds 1000 --m-values 9 16 25 --out-dir results\beta_sensitivity" `
    -LogFile $pipelineLog

# --- BƯỚC 2: Huấn luyện Baseline Medium (Figure 7 - 100 rounds) ---
Run-Step -Name "Train baselines medium ($Episodes x 100)" `
    -Command "$pythonUnbuffered scripts\train_baselines.py --episodes $Episodes --m $M --max-fl-rounds 100 --algorithms ppo ddpg greedy random --print-every-steps 1 --out-dir results\fig_7_medium --step-log-file results\logs\baselines_step_log_medium.txt" `
    -LogFile $pipelineLog

Run-Step -Name "Plot Figure 7 medium" `
    -Command "$pythonUnbuffered scripts\plot_fig_7.py --input-dir results\fig_7_medium --sigma 2.0 --out-path results\fig_7_medium\figure7_medium.png" `
    -LogFile $pipelineLog

# --- BƯỚC 3: Huấn luyện Baseline Full (Figure 7 - 1000 rounds) ---
Run-Step -Name "Train baselines full ($Episodes x 1000)" `
    -Command "$pythonUnbuffered scripts\train_baselines.py --episodes $Episodes --m $M --max-fl-rounds 1000 --algorithms ppo ddpg greedy random --print-every-steps 1 --out-dir results\fig_7_full --step-log-file results\logs\baselines_step_log_full.txt" `
    -LogFile $pipelineLog

Run-Step -Name "Plot Figure 7 full" `
    -Command "$pythonUnbuffered scripts\plot_fig_7.py --input-dir results\fig_7_full --sigma 2.0 --out-path results\fig_7_full\figure7_full.png" `
    -LogFile $pipelineLog

# --- BƯỚC 4: Thí nghiệm So sánh các Scheme & Ablation (Figure 4, 5, 6) ---
Run-Step -Name "Experiment: Scheme Comparison (Fig 4, 5, 6)" `
    -Command "$pythonUnbuffered scripts\run_fig_4_5_6.py --rounds 1000 --m-values 9 16 25 36 49 --out-dir results\eval_schemes --model-path `"results\fig_7_full\ppo_baseline_model`"" `
    -LogFile $pipelineLog

"==== Two-stage baseline pipeline finished at $(Get-Date -Format "yyyy-MM-dd HH:mm:ss") ====" | Out-File -FilePath $pipelineLog -Append -Encoding utf8
Write-Host "Pipeline done."
Write-Host "- Pipeline log : $pipelineLog"
Write-Host "- Medium steps : $mediumStepLog"
Write-Host "- Full steps   : $fullStepLog"
