import argparse
import csv
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.eval_schemes import SchemeEvaluator


SCHEME_LABELS = {
    "scheme1": "Scheme 1 (Proposed)",
    "scheme2": "Scheme 2 (Dynamic beta only)",
    "scheme3": "Scheme 3 (Fixed beta)",
    "scheme4": "Scheme 4 (LAG)",
    "scheme5": "Scheme 5 (Traditional Async FL)",
}

SCHEME_ORDER = ["scheme1", "scheme2", "scheme3", "scheme4", "scheme5"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_experiment(
    m_values: List[int],
    rounds: int,
    model_path: str,
    lag_threshold: float,
    beta_heuristic: str,
    enable_early_stopping: bool = False,
    log_dir: str | None = None,
) -> List[Dict]:
    all_rows: List[Dict] = []

    total_jobs = len(m_values) * len(SCHEME_ORDER)
    done_jobs = 0

    for m in m_values:
        cfg = SchemeEvaluator.build_config(m_value=m, eval_interval=1)
        evaluator = SchemeEvaluator(
            config=cfg,
            rounds=rounds,
            model_path=model_path,
            lag_threshold=lag_threshold,
            beta_heuristic=beta_heuristic,
            enable_early_stopping=enable_early_stopping,
            log_dir=os.path.join(log_dir, f"M{m}") if log_dir else None,
        )

        print(f"[INFO] Running all schemes for M={m}")
        results = evaluator.run_all_schemes()

        for row in results:
            done_jobs += 1
            row_out = dict(row)
            row_out["M"] = int(m)
            all_rows.append(row_out)
            print(
                f"[PROGRESS {done_jobs}/{total_jobs}] M={m}, {row_out['scheme']} "
                f"| comm={row_out['communication_times']:.1f} | acc={row_out['accuracy']:.4f} "
                f"| delay_total={row_out['time_consumption']:.2f}s "
                f"| energy={row_out['energy_consumption']:.4f}J "
                f"| cost={row_out['accumulated_cost']:.4f} | reward={row_out['total_reward']:.4f} "
                f"| rounds={row_out['rounds']}"
            )

    return all_rows


def run_fig5_experiment(
    m_values: List[int],
    rounds: int,
    model_path: str,
    beta_heuristic: str,
    enable_early_stopping: bool = False,
    log_dir: str | None = None,
) -> List[Dict]:
    """Runs ablation study for Figure 5 (Control Model Accuracy)."""
    all_rows: List[Dict] = []
    
    # Figure 5 compares:
    # 1. Proposed control model (Scheme 1, tau=5)
    # 2. Control model without tau (Scheme 1, tau=large)
    # 3. None control model (Scheme 5, All active)
    
    ablation_set = [
        ("The proposed control model", "scheme1", 5),
        ("Control model without \u03c4", "scheme1", 1000), # Large tau to disable force-active
        ("None control model", "scheme5", 5), # Scheme 5 is all active
    ]
    
    total_jobs = len(m_values) * len(ablation_set)
    done_jobs = 0
    
    for m in m_values:
        print(f"[INFO] Running Fig 5 ablation for M={m}")
        for label, base_scheme, tau in ablation_set:
            cfg = SchemeEvaluator.build_config(m_value=m, eval_interval=1)
            evaluator = SchemeEvaluator(
                config=cfg,
                rounds=rounds,
                model_path=model_path,
                beta_heuristic=beta_heuristic,
                enable_early_stopping=enable_early_stopping,
                force_active_rounds=tau,
                log_dir=os.path.join(log_dir, f"M{m}_fig5") if log_dir else None,
            )
            
            # Run specific scheme
            if base_scheme == "scheme1":
                res = evaluator.run_scheme1_proposed()
            else:
                res = evaluator.run_scheme5_traditional_async()
            
            done_jobs += 1
            res["M"] = int(m)
            res["fig5_label"] = label
            all_rows.append(res)

            print(
                f"[PROGRESS FIG5 {done_jobs}/{total_jobs}] M={m}, {label} "
                f"| acc={res['accuracy']:.4f} | delay_total={res['time_consumption']:.2f}s "
                f"| energy={res['energy_consumption']:.4f}J "
                f"| cost={res['accumulated_cost']:.4f} | reward={res['total_reward']:.4f} "
                f"| rounds={res['rounds']}"
            )
            
    return all_rows


def save_results_csv(rows: List[Dict], output_csv: str) -> None:
    if not rows:
        return

    ensure_dir(os.path.dirname(output_csv))
    # Tự động lấy tên cột động từ dữ liệu đầu vào để không bị lỗi thiếu cột
    fieldnames = list(rows[0].keys())

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_by_scheme(rows: List[Dict], m_values: List[int], metric_key: str, ylabel: str, title: str, output_path: str) -> None:
    plt.figure(figsize=(9, 5.5))

    for scheme in SCHEME_ORDER:
        scheme_rows = [r for r in rows if r["scheme"] == scheme]
        by_m = {int(r["M"]): float(r[metric_key]) for r in scheme_rows}
        x = [int(m) for m in m_values]
        y = [by_m.get(int(m), float("nan")) for m in x]

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=SCHEME_LABELS.get(scheme, scheme),
        )

    plt.xlabel("Number of AUVs (M)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_fig5(rows: List[Dict], m_values: List[int], output_path: str) -> None:
    plt.figure(figsize=(9, 5.5))
    
    labels = ["The proposed control model", "Control model without \u03c4", "None control model"]
    
    for label in labels:
        label_rows = [r for r in rows if r["fig5_label"] == label]
        by_m = {int(r["M"]): float(r["accuracy"]) for r in label_rows}
        x = [int(m) for m in m_values]
        y = [by_m.get(int(m), float("nan")) for m in x]
        
        plt.plot(
            x,
            y,
            marker="s",
            linewidth=2,
            label=label,
        )
        
    plt.xlabel("Number of AUVs (M)")
    plt.ylabel("Accuracy")
    plt.title("Figure 5: Comparison of accuracies for different control models")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Figures 4, 5, 6 experiments across M for 5 schemes.")
    parser.add_argument("--rounds", type=int, default=1000, help="FL rounds per scheme")
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[9, 16, 25, 36, 49],
        help="AUV counts to evaluate",
    )
    parser.add_argument("--model-path", type=str, default="./models/ppo_auv_final", help="Path to trained PPO model")
    parser.add_argument("--lag-threshold", type=float, default=1e4, help="LAG threshold")
    parser.add_argument(
        "--beta-heuristic",
        type=str,
        choices=["linear", "constant"],
        default="linear",
        help="Fallback beta schedule for Scheme 2 if PPO model is missing",
    )
    parser.add_argument("--enable-early-stopping", action="store_true", help="Enable early stopping in scheme runs")
    parser.add_argument("--out-dir", type=str, default="results/eval_schemes", help="Output directory")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    rows = run_experiment(
        m_values=[int(m) for m in args.m_values],
        rounds=int(args.rounds),
        model_path=args.model_path,
        lag_threshold=float(args.lag_threshold),
        beta_heuristic=args.beta_heuristic,
        enable_early_stopping=bool(args.enable_early_stopping),
        log_dir=os.path.join(args.out_dir, "scheme_logs"),
    )

    results_csv = os.path.join(args.out_dir, "scheme_results.csv")
    save_results_csv(rows, results_csv)

    print("\n[INFO] Starting Figure 5 Experiments...")
    fig5_rows = run_fig5_experiment(
        m_values=[int(m) for m in args.m_values],
        rounds=int(args.rounds),
        model_path=args.model_path,
        beta_heuristic=args.beta_heuristic,
        enable_early_stopping=bool(args.enable_early_stopping),
        log_dir=os.path.join(args.out_dir, "scheme_logs"),
    )

    fig5_csv = os.path.join(args.out_dir, "fig5_results.csv")
    save_results_csv(fig5_rows, fig5_csv)

    print("\n[INFO] Generating Plots...")

    # --- 7 đồ thị so sánh scheme theo từng chỉ số ---
    scheme_plots = [
        ("figure4a_avg_communication_times.png", "avg_communication_times", "Average Communication Times", "Fig 4(a): Average Comm Times vs M"),
        ("figure4b_accuracy.png",                "accuracy",                "Accuracy",                    "Fig 4(b): Accuracy vs M"),
        ("figure_avg_delay.png",                 "avg_delay",               "Average Delay (s)",           "Average Delay vs M"),
        ("figure_avg_energy.png",                "avg_energy",              "Average Energy (J)",          "Average Energy vs M"),
        ("figure6_cost.png",                     "accumulated_cost",        "Accumulated Cost",            "Fig 6: Accumulated Cost vs M"),
        ("figure_avg_reward.png",                "avg_reward",              "Average Reward",              "Average Reward vs M"),
        ("figure_converged_round.png",           "rounds",                  "Converged Round",             "Converged Round vs M"),
    ]
    saved_paths = [results_csv, fig5_csv]
    for fname, key, ylabel, title in scheme_plots:
        out_path = os.path.join(args.out_dir, fname)
        plot_by_scheme(
            rows=rows,
            m_values=[int(m) for m in args.m_values],
            metric_key=key,
            ylabel=ylabel,
            title=title,
            output_path=out_path,
        )
        saved_paths.append(out_path)

    fig5_path = os.path.join(args.out_dir, "figure5_control_models.png")
    plot_fig5(
        rows=fig5_rows,
        m_values=[int(m) for m in args.m_values],
        output_path=fig5_path,
    )
    saved_paths.append(fig5_path)

    print("[DONE] Saved outputs:")
    for p in saved_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
