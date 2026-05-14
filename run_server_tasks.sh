#!/usr/bin/env bash
set -euo pipefail

# 1. Activate Conda environment
CONDA_ENV="auv_rfl"

if   [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ];  then source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ];   then source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ];        then source "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ];  then source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

echo "[INFO] Activating conda environment: $CONDA_ENV"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
PYTHON="python"

EPISODES=1000
ROUNDS=1000
RESULTS_DIR="results"
LOG_DIR="$RESULTS_DIR/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ---------------------------------------------------------------------------
# Task 1: Train PPO, Random, and Greedy for M=9
# ---------------------------------------------------------------------------
echo "==== [TASK 1] Training PPO, random, and greedy for M=9 ===="
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
# Task 2: Train PPO baseline for M=16, 25, 36, 49
# ---------------------------------------------------------------------------
echo "==== [TASK 2] Training PPO for M=16, 25, 36, 49 ===="
for m_val in 16 25 36 49; do
    echo "[INFO] Training PPO for M=$m_val"
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
# Task 3: Plot Figure 7 for M=9 (PPO vs Baseline)
# ---------------------------------------------------------------------------
echo "==== [TASK 3] Plotting Figure 7 (Cumulative Metrics) ===="
$PYTHON -u scripts/plot_fig_7.py \
    --input-dir "$RESULTS_DIR/fig_7_M9" \
    --out-dir "$RESULTS_DIR/fig_7_M9"

# ---------------------------------------------------------------------------
# Task 4: Run Scheme Comparison (Figs 4, 5, 6) for M=9, 16, 25, 36, 49
# ---------------------------------------------------------------------------
echo "==== [TASK 4] Running Fig 4, 5, 6 evaluation for full schemes and full M ===="
$PYTHON -u scripts/run_fig_4_5_6.py \
    --rounds $ROUNDS \
    --m-values 9 16 25 36 49 \
    --model-path "$RESULTS_DIR/fig_7_M{M}/ppo_baseline_model" \
    --lag-threshold 1e4 \
    --enable-early-stopping \
    --out-dir "$RESULTS_DIR/eval_schemes_all"

echo "==== All Tasks Completed Successfully! ===="
