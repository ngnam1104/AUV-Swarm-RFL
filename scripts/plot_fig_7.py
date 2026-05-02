import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:
    gaussian_filter1d = None


SERIES_FILES = {
    "PPO (Proposed)": "ppo_metrics.csv",
    "SAC":            "sac_metrics.csv",
    "TD3":            "td3_metrics.csv",
    "DDPG":           "ddpg_metrics.csv",
    "A2C":            "a2c_metrics.csv",
    "Greedy":         "greedy_metrics.csv",
    "Random":         "random_metrics.csv",
}


def read_metric_series(csv_path: str, metric_col: str) -> np.ndarray:
    values = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if metric_col in row:
                values.append(float(row[metric_col]))
    return np.asarray(values, dtype=float)


def smooth_curve(y: np.ndarray, sigma: float) -> np.ndarray:
    if y.size == 0:
        return y

    if gaussian_filter1d is not None:
        return gaussian_filter1d(y, sigma=sigma)

    window = max(1, int(round(sigma * 3)))
    if window <= 1:
        return y

    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="same")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure 7 convergence from accumulated cost series.")
    parser.add_argument("--input-dir", type=str, default="results/fig_7", help="Directory containing baseline csv files")
    parser.add_argument("--sigma", type=float, default=2.0, help="Smoothing sigma")
    parser.add_argument("--out-dir", type=str, default="results/fig_7", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    metrics_to_plot = [
        ("accumulated_cost", "Accumulated Cost", "figure7_accumulated_cost.png"),
        ("avg_delay", "Average Delay (s)", "figure7_avg_delay.png"),
        ("avg_energy", "Average Energy (J)", "figure7_avg_energy.png"),
        ("avg_reward", "Average Reward", "figure7_avg_reward.png"),
        ("avg_comm", "Average Communication Times", "figure7_avg_comm.png"),
    ]

    for metric_col, ylabel, out_filename in metrics_to_plot:
        loaded = {}
        for label, filename in SERIES_FILES.items():
            path = os.path.join(args.input_dir, filename)
            if os.path.exists(path):
                series = read_metric_series(path, metric_col)
                if series.size > 0:
                    loaded[label] = series

        if not loaded:
            print(f"[WARN] No baseline data found for {metric_col}.")
            continue

        for label, series in loaded.items():
            if len(series) < 20:
                print(
                    f"[WARN] {label} has only {len(series)} episodes. "
                    f"Figure 7 ({metric_col}) will be under-sampled and may look flat."
                )

        plt.figure(figsize=(10, 6))
        for label, series in loaded.items():
            y_smoothed = smooth_curve(series, sigma=args.sigma)
            x = np.arange(len(y_smoothed))
            plt.plot(x, y_smoothed, linewidth=2, label=label)

        max_len = max(len(v) for v in loaded.values()) if loaded else 0
        plt.xlabel(f"Episodes (0 - {max_len})")
        plt.ylabel(ylabel)
        plt.title(f"Figure 7: RL Algorithm Convergence - {ylabel}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, out_filename)
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"[DONE] Saved figure: {out_path}")


if __name__ == "__main__":
    main()
