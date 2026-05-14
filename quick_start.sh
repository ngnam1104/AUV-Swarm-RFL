#!/usr/bin/env bash
# =============================================================================
# quick_start.sh  —  Ubuntu / bash — AUV-Swarm-RFL full experiment pipeline
#
# Cách dùng:
#   bash quick_start.sh [WORKSPACE_ROOT] [EPISODES] [ROUNDS]
#
# Mặc định:
#   WORKSPACE_ROOT = thư mục chứa script này
#   EPISODES       = 1000
#   ROUNDS         = 1000
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Tham số đầu vào
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${1:-$SCRIPT_DIR}"
EPISODES="${2:-1000}"
ROUNDS="${3:-1000}"

cd "$WORKSPACE_ROOT"

# ---------------------------------------------------------------------------
# 1. Kích hoạt môi trường Conda
# ---------------------------------------------------------------------------
CONDA_ENV="auv_rfl"

if   [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ];  then source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ];   then source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ];        then source "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ];  then source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
    if ! command -v conda &>/dev/null; then
        echo "[ERROR] conda not found. Install Miniconda/Anaconda first." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

echo "[INFO] Activating conda environment: $CONDA_ENV"
conda activate "$CONDA_ENV"

# GPU settings
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="python"

# ---------------------------------------------------------------------------
# 2. Chuẩn bị thư mục results/logs
# ---------------------------------------------------------------------------
RESULTS_DIR="$WORKSPACE_ROOT/results"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"

PIPELINE_LOG="$LOG_DIR/pipeline.log"

# ---------------------------------------------------------------------------
# 3. Hàm run_step — in stdout + ghi pipeline log, KHÔNG exit toàn bộ khi fail
# ---------------------------------------------------------------------------
run_step() {
    local name="$1"
    local log_file="$2"
    shift 2

    local stamp
    stamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "[$stamp] ===== START: $name =====" | tee -a "$log_file"
    echo "[$stamp] CMD: $*" | tee -a "$log_file"

    set +e
    PYTHONUNBUFFERED=1 "$@" 2>&1 | tee -a "$log_file"
    local exit_code="${PIPESTATUS[0]}"
    set -e

    stamp="$(date '+%Y-%m-%d %H:%M:%S')"
    if [ "$exit_code" -ne 0 ]; then
        echo "[$stamp] FAIL: $name (exit=$exit_code)" | tee -a "$log_file"
        echo "[WARN] Step '$name' failed with exit=$exit_code — continuing pipeline..." | tee -a "$log_file"
        return 0
    fi
    echo "[$stamp] DONE: $name" | tee -a "$log_file"
}

echo "==== Pipeline started at $(date '+%Y-%m-%d %H:%M:%S') ====" >> "$PIPELINE_LOG"

# ---------------------------------------------------------------------------
# 4. Bước 1 — Beta Sensitivity (Mô phỏng Beta)
# ---------------------------------------------------------------------------
run_step "Beta Sensitivity Evaluation" "$PIPELINE_LOG" \
    $PYTHON -u scripts/eval_beta_sensitivity.py \
        --rounds $ROUNDS \
        --enable-early-stopping \
        --m-values 9 16 25 \
        --beta-start 0.1 \
        --beta-end 0.9 \
        --beta-step 0.1 \
        --out-dir "$RESULTS_DIR/beta_sensitivity"

# ---------------------------------------------------------------------------
# 5. Bước 2 — Physical Parameter Sensitivity
# ---------------------------------------------------------------------------
run_step "Physical Parameter Sensitivity" "$PIPELINE_LOG" \
    $PYTHON -u scripts/eval_physical_params.py

# ---------------------------------------------------------------------------
# 6. Bước 3 — Train PPO, Random, Greedy cho M=9 (Chuẩn bị cho Fig 7)
# ---------------------------------------------------------------------------
run_step "Train Baselines (PPO, Random, Greedy) for M=9" "$PIPELINE_LOG" \
    $PYTHON -u scripts/train_baselines.py \
        --m 9 \
        --max-fl-rounds $ROUNDS \
        --episodes $EPISODES \
        --eval-interval 5 \
        --enable-early-stopping \
        --algorithms ppo random greedy \
        --print-every-steps 10 \
        --out-dir "$RESULTS_DIR/fig_7_M9" \
        --log-dir "$LOG_DIR/fig_7_M9"

# ---------------------------------------------------------------------------
# 7. Bước 4 — Plot Figure 7 (M=9)
# ---------------------------------------------------------------------------
run_step "Plot Figure 7 (Cumulative Metrics)" "$PIPELINE_LOG" \
    $PYTHON -u scripts/plot_fig_7.py \
        --input-dir "$RESULTS_DIR/fig_7_M9" \
        --out-dir "$RESULTS_DIR/fig_7_M9"

# ---------------------------------------------------------------------------
# 8. Bước 5 — Train PPO cho các M còn lại (Chuẩn bị cho Fig 4, 5, 6)
# ---------------------------------------------------------------------------
for m_val in 16 25 36 49; do
    run_step "Train PPO baseline for M=$m_val" "$PIPELINE_LOG" \
        $PYTHON -u scripts/train_baselines.py \
            --m "$m_val" \
            --max-fl-rounds $ROUNDS \
            --episodes $EPISODES \
            --eval-interval 5 \
            --enable-early-stopping \
            --algorithms ppo \
            --print-every-steps 10 \
            --out-dir "$RESULTS_DIR/fig_7_M${m_val}" \
            --log-dir "$LOG_DIR/fig_7_M${m_val}"
done

# ---------------------------------------------------------------------------
# 9. Bước 6 — Scheme Comparison & Ablation (Figure 4, 5, 6)
# ---------------------------------------------------------------------------
PPO_MODEL_PATH="$RESULTS_DIR/fig_7_M{M}/ppo_baseline_model"

run_step "Scheme Comparison (Figs 4, 5, 6) for all M" "$PIPELINE_LOG" \
    $PYTHON -u scripts/run_fig_4_5_6.py \
        --rounds $ROUNDS \
        --m-values 9 16 25 36 49 \
        --model-path "$PPO_MODEL_PATH" \
        --lag-threshold 1e4 \
        --enable-early-stopping \
        --out-dir "$RESULTS_DIR/eval_schemes_all"

# ---------------------------------------------------------------------------
# 10. Zip kết quả
# ---------------------------------------------------------------------------
ZIP_PATH="$RESULTS_DIR/experiment_results.zip"
echo "[INFO] Zipping results..." | tee -a "$PIPELINE_LOG"
set +e
zip -r "$ZIP_PATH" "$RESULTS_DIR/" --exclude "*.zip" 2>&1 | tee -a "$PIPELINE_LOG"
set -e

# ---------------------------------------------------------------------------
# 11. Kết thúc
# ---------------------------------------------------------------------------
echo "" | tee -a "$PIPELINE_LOG"
echo "==== Pipeline finished at $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "$PIPELINE_LOG"
echo ""
echo "Pipeline done."
echo "  Results  : $RESULTS_DIR"
echo "  Pipeline log: $PIPELINE_LOG"
echo "  Zip      : $ZIP_PATH"
