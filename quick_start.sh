#!/usr/bin/env bash
# =============================================================================
# quick_start.sh  —  Ubuntu / bash — AUV-Swarm-RFL experiment pipeline
#
# Usage:
#   bash quick_start.sh [WORKSPACE_ROOT] [EPISODES] [M]
#
# Defaults:
#   WORKSPACE_ROOT = directory containing this script
#   EPISODES       = 1000
#   M              = 9
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Input parameters
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${1:-$SCRIPT_DIR}"
EPISODES="${2:-1000}"
M="${3:-9}"

cd "$WORKSPACE_ROOT"

# ---------------------------------------------------------------------------
# 1. Activate Conda environment
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
# 2. Install dependencies
# ---------------------------------------------------------------------------
echo "[INFO] Installing requirements..."
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 3. Prepare results/logs directories
# ---------------------------------------------------------------------------
RESULTS_DIR="$WORKSPACE_ROOT/results"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"

PIPELINE_LOG="$LOG_DIR/pipeline.log"

# ---------------------------------------------------------------------------
# 4. run_step helper — tee stdout to pipeline log, never abort on failure
# ---------------------------------------------------------------------------
run_step() {
    local name="$1"
    local log_file="$2"
    shift 2

    local stamp
    stamp="$(date '+%Y-%m-%d %H:%M:%S')"
    # Ghi vào file log (không in ra màn hình)
    echo "" >> "$log_file"
    echo "[$stamp] ===== START: $name =====" >> "$log_file"
    echo "[$stamp] CMD: $*" >> "$log_file"

    set +e
    # 1>> "$log_file": Ghi metrics (stdout) thẳng vào file, không hiện terminal
    # 2>&2: Giữ nguyên stderr (tqdm bar) trên terminal
    PYTHONUNBUFFERED=1 "$@" 1>> "$log_file" 2>&2
    local exit_code="${PIPESTATUS[0]}"
    set -e

    stamp="$(date '+%Y-%m-%d %H:%M:%S')"
    if [ "$exit_code" -ne 0 ]; then
        echo "[$stamp] FAIL: $name (exit=$exit_code)" >> "$log_file"
        echo "[FAIL] Step '$name' exited with code $exit_code. See: $log_file" >&2
    else
        echo "[$stamp] DONE: $name" >> "$log_file"
    fi
    return "$exit_code"
}

echo "==== Pipeline started at $(date '+%Y-%m-%d %H:%M:%S') ====" >> "$PIPELINE_LOG"

# ---------------------------------------------------------------------------
# Step 1 — Train PPO baseline
#
#   Output:
#     - PPO model  : results/fig_7/ppo_baseline_model.zip
#     - Step logs  : results/logs/fig_7_baselines/ppo_steps.log
#     - Metrics CSV: results/fig_7/ppo_metrics.csv
# ---------------------------------------------------------------------------
run_step "Train PPO baseline ($EPISODES ep x 1000 rounds)" "$PIPELINE_LOG" \
    $PYTHON -u scripts/train_baselines.py \
        --m "$M" \
        --max-fl-rounds 1000 \
        --episodes "$EPISODES" \
        --eval-interval 5 \
        --enable-early-stopping \
        --algorithms ppo \
        --print-every-steps 10 \
        --out-dir "$RESULTS_DIR/fig_7" \
        --log-dir "$LOG_DIR/fig_7_baselines" || {
    echo "[ERROR] PPO training failed. Pipeline cannot continue." >&2
    echo "        Check $PIPELINE_LOG for details." >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Step 2 — Plot Figure 7 (convergence / cost curves)
#
#   Output: results/fig_7/figure7.png
# ---------------------------------------------------------------------------
run_step "Plot Figure 7" "$PIPELINE_LOG" \
    $PYTHON -u scripts/plot_fig_7.py \
        --input-dir "$RESULTS_DIR/fig_7" \
        --sigma 2.0 \
        --enable-early-stopping \
        --out-dir "$RESULTS_DIR/fig_7"

# ---------------------------------------------------------------------------
# Step 3 — Scheme Comparison & Ablation (Figures 4, 5, 6)
#
#   PPO model loaded from: results/fig_7/ppo_baseline_model[.zip]
#   Output: results/eval_schemes/
# ---------------------------------------------------------------------------
PPO_MODEL_PATH="$RESULTS_DIR/fig_7/ppo_baseline_model"

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
    echo "[ERROR] PPO model not found at ${PPO_MODEL_PATH}[.zip]" | tee -a "$PIPELINE_LOG"
    echo "        Step 1 (PPO training) likely failed or did not save correctly." | tee -a "$PIPELINE_LOG"
    exit 1
fi


# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "" | tee -a "$PIPELINE_LOG"
echo "==== Pipeline finished at $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "$PIPELINE_LOG"
echo ""
echo "Pipeline done."
echo "  Results     : $RESULTS_DIR"
echo "  Pipeline log: $PIPELINE_LOG"
