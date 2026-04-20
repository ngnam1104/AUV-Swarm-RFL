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
    "PPO (Proposed)": "ppo_accumulated_cost.csv",
    "DDPG": "ddpg_accumulated_cost.csv",
    "Greedy": "greedy_accumulated_cost.csv",
    "Random": "random_accumulated_cost.csv",
}


def read_cost_series(csv_path: str) -> np.ndarray:
    values = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row["accumulated_cost"]))
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
    parser.add_argument("--out-path", type=str, default="results/fig_7/figure7_convergence.png", help="Output figure path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    loaded = {}
    for label, filename in SERIES_FILES.items():
        path = os.path.join(args.input_dir, filename)
        if os.path.exists(path):
            loaded[label] = read_cost_series(path)

    if not loaded:
        raise FileNotFoundError(
            "No baseline csv file found. Run scripts/train_baselines.py first."
        )

    for label, series in loaded.items():
        if len(series) < 20:
            print(
                f"[WARN] {label} has only {len(series)} episodes. "
                "Figure 7 will be under-sampled and may look flat."
            )

    plt.figure(figsize=(10, 6))

    for label, series in loaded.items():
        y_smoothed = smooth_curve(series, sigma=args.sigma)
        x = np.arange(len(y_smoothed))
        plt.plot(x, y_smoothed, linewidth=2, label=label)

    max_len = max(len(v) for v in loaded.values()) if loaded else 0
    plt.xlabel(f"Episodes (0 - {max_len})")
    plt.ylabel("Accumulated Cost")
    plt.title("Figure 7: RL Algorithm Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=150)

    print("[DONE] Saved figure:")
    print(f"- {args.out_path}")


if __name__ == "__main__":
    main()
