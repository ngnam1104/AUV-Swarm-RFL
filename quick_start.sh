#!/usr/bin/env bash
# =============================================================================
# quick_start.sh  —  Ubuntu / bash — AUV-Swarm-RFL full experiment pipeline
#
# Cách dùng:
#   bash quick_start.sh [WORKSPACE_ROOT] [EPISODES] [M]
#
# Mặc định:
#   WORKSPACE_ROOT = thư mục chứa script này
#   EPISODES       = 1000
#   M              = 9
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Tham số đầu vào
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${1:-$SCRIPT_DIR}"
EPISODES="${2:-1000}"
M="${3:-9}"

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
# 2. Cài đặt dependencies
# ---------------------------------------------------------------------------
echo "[INFO] Installing requirements..."
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 3. Chuẩn bị thư mục results/logs
# ---------------------------------------------------------------------------
RESULTS_DIR="$WORKSPACE_ROOT/results"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"

PIPELINE_LOG="$LOG_DIR/pipeline.log"

# ---------------------------------------------------------------------------
# 4. Hàm run_step — in stdout + ghi pipeline log, KHÔNG exit toàn bộ khi fail
#    Dùng run_step_safe cho các bước không critical (vd: zip)
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
        # Không exit — cho pipeline tiếp tục các bước còn lại
        return 0
    fi
    echo "[$stamp] DONE: $name" | tee -a "$log_file"
}

echo "==== Pipeline started at $(date '+%Y-%m-%d %H:%M:%S') ====" >> "$PIPELINE_LOG"



# ---------------------------------------------------------------------------
# 7. Bước 2 — Train ALL 7 RL algorithms (bootstrap cho Scheme Comparison)
#    Dùng --out-dir results/fig_7  (PPO model cần ở đây để Scheme 1 load)
# ---------------------------------------------------------------------------
run_step "Train 7 RL algorithms bootstrap ($EPISODES ep x 1000 rounds)" "$PIPELINE_LOG" \
    $PYTHON -u scripts/train_baselines.py \
        --m "$M" \
        --max-fl-rounds 1000 \
        --episodes "$EPISODES" \
        --eval-interval 5 \
        --enable-early-stopping \
        --algorithms ppo sac td3 ddpg a2c greedy random \
        --parallel \
        --print-every-steps 10 \
        --out-dir "$RESULTS_DIR/fig_7" \
        --log-dir "$LOG_DIR/fig_7_bootstrap"

# ---------------------------------------------------------------------------
# 8. Bước 3 — Plot Figure 7 (bootstrap)
# ---------------------------------------------------------------------------
run_step "Plot Figure 7 bootstrap" "$PIPELINE_LOG" \
    $PYTHON -u scripts/plot_fig_7.py \
        --input-dir "$RESULTS_DIR/fig_7" \
        --sigma 2.0 \
        --enable-early-stopping \
        --out-dir "$RESULTS_DIR/fig_7"

# ---------------------------------------------------------------------------
# 9. Bước 4 — Scheme Comparison & Ablation (Figure 4, 5, 6)
#    PPO model được load từ results/fig_7/ppo_baseline_model
# ---------------------------------------------------------------------------
PPO_MODEL_PATH="$RESULTS_DIR/fig_7/ppo_baseline_model"

# Kiểm tra model tồn tại (có thể là .zip hoặc không có extension)
if [ -f "${PPO_MODEL_PATH}.zip" ] || [ -f "$PPO_MODEL_PATH" ]; then
    run_step "Scheme Comparison (Figs 4, 5, 6)" "$PIPELINE_LOG" \
        $PYTHON -u scripts/run_fig_4_5_6.py \
            --rounds 1000 \
            --m-values 9 16 25 36 49 \
            --model-path "$PPO_MODEL_PATH" \
            --lag-threshold 1e4 \
            --enable-early-stopping \
            --out-dir "$RESULTS_DIR/eval_schemes"
else
    echo "[WARN] PPO model not found at $PPO_MODEL_PATH — skipping Scheme Comparison." | tee -a "$PIPELINE_LOG"
fi

# ---------------------------------------------------------------------------
# 12. Kết thúc
# ---------------------------------------------------------------------------
echo "" | tee -a "$PIPELINE_LOG"
echo "==== Pipeline finished at $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "$PIPELINE_LOG"
echo ""
echo "Pipeline done."
echo "  Results  : $RESULTS_DIR"
echo "  Pipeline log: $PIPELINE_LOG"
