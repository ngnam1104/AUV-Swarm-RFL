import argparse
import csv
import os
import sys
import time
from dataclasses import asdict
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.settings import ACOUSTIC_CFG, FL_CFG, HW_CFG, RL_CFG
from env.communication import CommunicationModel
from env.energy import EnergyModel
from env.latency import LatencyModel
from env.reward import RewardModel
from fl_core.simulator import FLSimulator


def _log(msg: str, fh=None) -> None:
    """Print to stdout and optionally write to a log file handle."""
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()


def build_config(m_value: int, max_rounds: int, eval_interval: int) -> SimpleNamespace:
    cfg = SimpleNamespace(
        **asdict(ACOUSTIC_CFG),
        **asdict(HW_CFG),
        **asdict(FL_CFG),
        **asdict(RL_CFG),
    )
    cfg.M = int(m_value)
    cfg.max_fl_rounds = int(max_rounds)
    cfg.eval_interval = int(eval_interval)
    return cfg


def evaluate_for_beta(
    cfg: SimpleNamespace,
    beta: float,
    rounds: int,
    enable_early_stopping: bool,
    log_fh=None,
) -> tuple[float, float, float]:
    fl_sim = FLSimulator(cfg)
    comm_model = CommunicationModel(cfg)
    latency_model = LatencyModel(cfg, comm_model)
    energy_model = EnergyModel(cfg)
    reward_model = RewardModel(cfg)

    # Co dinh cac tham so vat ly o muc trung binh theo yeu cau.
    p_mid = 0.5 * (float(cfg.p_max) + 0.01)
    f_mid = 0.5 * (float(cfg.f_max) + float(cfg.f_min))

    p_m = np.full(cfg.M, p_mid, dtype=float)
    f_m = np.full(cfg.M, f_mid, dtype=float)
    p_l = float(p_mid)
    f_l = float(f_mid)

    communication_times = 0.0
    total_time_consumption = 0.0
    total_energy_consumption = 0.0
    total_cost_consumption = 0.0
    total_reward_consumption = 0.0
    final_accuracy = 0.0
    
    total_local_train_sec = 0.0
    total_aggregate_sec = 0.0

    for rnd in range(1, rounds + 1):
        accuracy, active_indices, _, is_converged = fl_sim.sync_run_step(beta=beta, rnd=rnd)
        
        total_local_train_sec += fl_sim.last_timing_stats.get("local_train_and_grad_sec", 0.0)
        total_aggregate_sec += fl_sim.last_timing_stats.get("aggregate_sec", 0.0)

        lambda_m = np.zeros(cfg.M, dtype=float)
        lambda_m[active_indices] = 1.0

        E_total, energy_details, t_total, _ = energy_model.compute_total_energy_from_latency(
            latency_model=latency_model,
            lambda_m=lambda_m,
            f_m=f_m,
            p_m=p_m,
            f_L=f_l,
            p_L=p_l,
        )
        
        E_m_array = energy_details["E_Cp_m"] + energy_details["E_C_m"]
        E_L_val_total = energy_details["E_Cp_L"] + energy_details["E_C_L"]
        reward, cost, _ = reward_model.compute_reward(
            T_total=t_total,
            E_total=E_total,
            E_m=E_m_array,
            E_L=E_L_val_total
        )
        
        if rnd == 1:
            _log(f"    [Started] beta={beta:.1f} | First round training...", log_fh)

        communication_times += float(np.sum(lambda_m))
        total_time_consumption += float(t_total)
        total_energy_consumption += float(E_total)
        total_cost_consumption += float(cost)
        total_reward_consumption += float(reward)
        final_accuracy = float(accuracy)

        if rnd % 5 == 0 or rnd == rounds or (enable_early_stopping and is_converged):
            _log(
                f"    [Round {rnd}/{rounds}] beta={beta:.1f} "
                f"| acc={final_accuracy:.4f} | comm={communication_times:.0f} "
                f"| delay={t_total:.2f}s | energy={E_total:.4f}J "
                f"| cost={cost:.4f} | reward={reward:.4f}",
                log_fh,
            )

        if enable_early_stopping and is_converged:
            _log(
                f"    [Early Stopping] Converged at round {rnd} "
                f"| acc={final_accuracy:.4f} | comm={communication_times:.0f} "
                f"| delay={t_total:.2f}s | energy={E_total:.4f}J "
                f"| cost={cost:.4f} | reward={reward:.4f} | rounds={rnd}",
                log_fh,
            )
            break

    _log(f"    [Time Tracking] beta={beta:.1f} | Train mạng: {total_local_train_sec:.2f}s | Tổng hợp: {total_aggregate_sec:.2f}s", log_fh)

    avg_time = total_time_consumption / rnd if rnd > 0 else 0.0
    avg_energy = total_energy_consumption / rnd if rnd > 0 else 0.0
    avg_cost = total_cost_consumption / rnd if rnd > 0 else 0.0
    avg_reward = total_reward_consumption / rnd if rnd > 0 else 0.0
    return communication_times, final_accuracy, avg_time, avg_energy, avg_cost, avg_reward, rnd


def save_csv(rows: list[dict], output_csv: str) -> None:
    if not rows:
        return

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        "M",
        "beta",
        "communication_times",
        "accuracy_round_1000",
        "time_consumption",
        "energy_consumption",
        "cost_consumption",
        "reward_consumption",
        "converged_round",
        "elapsed_seconds",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    rows: list[dict],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: str,
) -> None:
    plt.figure(figsize=(8, 5))

    all_m = sorted({int(r["M"]) for r in rows})
    for m_value in all_m:
        filtered = [r for r in rows if int(r["M"]) == m_value]
        filtered.sort(key=lambda x: float(x["beta"]))

        x = [float(r["beta"]) for r in filtered]
        y = [float(r[metric_key]) for r in filtered]

        plt.plot(x, y, marker="o", linewidth=2, label=f"M={m_value}")

    plt.xlabel("beta")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_latency_reward_cost(rows: list[dict], output_path: str) -> None:
    plt.figure(figsize=(18, 5))
    all_m = sorted({int(r["M"]) for r in rows})
    
    # Latency
    plt.subplot(1, 3, 1)
    for m_value in all_m:
        filtered = [r for r in rows if int(r["M"]) == m_value]
        filtered.sort(key=lambda x: float(x["beta"]))
        plt.plot([float(r["beta"]) for r in filtered], [float(r["time_consumption"]) for r in filtered], marker="o", lw=2, label=f"M={m_value}")
    plt.xlabel("beta")
    plt.ylabel("Average Latency (s)")
    plt.title("Latency vs beta")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Reward
    plt.subplot(1, 3, 2)
    for m_value in all_m:
        filtered = [r for r in rows if int(r["M"]) == m_value]
        filtered.sort(key=lambda x: float(x["beta"]))
        plt.plot([float(r["beta"]) for r in filtered], [float(r["reward_consumption"]) for r in filtered], marker="s", lw=2, label=f"M={m_value}")
    plt.xlabel("beta")
    plt.ylabel("Average Reward")
    plt.title("Reward vs beta")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Cost
    plt.subplot(1, 3, 3)
    for m_value in all_m:
        filtered = [r for r in rows if int(r["M"]) == m_value]
        filtered.sort(key=lambda x: float(x["beta"]))
        plt.plot([float(r["beta"]) for r in filtered], [float(r["cost_consumption"]) for r in filtered], marker="^", lw=2, label=f"M={m_value}")
    plt.xlabel("beta")
    plt.ylabel("Average Cost")
    plt.title("Cost vs beta")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_input_warnings(rounds: int, betas: np.ndarray, m_values: list[int]) -> None:
    if rounds < 1000:
        print(
            f"[WARN] rounds={rounds} < 1000. This is a smoke run; curves may look flat or noisy."
        )
    if len(betas) < 5:
        print(
            f"[WARN] only {len(betas)} beta points. Use >= 5 points (e.g., 0.1..0.9 step 0.1) for Figure 1-3."
        )
    if len(m_values) < 3:
        print(
            f"[WARN] only {len(m_values)} M values. Paper setup expects M in {{9,16,25}} for this stage."
        )


def print_output_diagnostics(rows: list[dict]) -> None:
    if not rows:
        return

    all_m = sorted({int(r["M"]) for r in rows})
    for m_value in all_m:
        subset = [r for r in rows if int(r["M"]) == m_value]
        comm_vals = np.array([float(r["communication_times"]) for r in subset], dtype=float)
        acc_vals = np.array([float(r["accuracy_round_1000"]) for r in subset], dtype=float)
        time_vals = np.array([float(r["time_consumption"]) for r in subset], dtype=float)

        print(
            f"[DIAG M={m_value}] "
            f"comm_range=({comm_vals.min():.4f},{comm_vals.max():.4f}) | "
            f"acc_range=({acc_vals.min():.4f},{acc_vals.max():.4f}) | "
            f"time_range=({time_vals.min():.4f},{time_vals.max():.4f})"
        )

        if float(np.ptp(comm_vals)) < 1e-9:
            print(
                f"[WARN M={m_value}] Communication times are identical across beta in this run. "
                "Increase rounds or inspect lazy-node threshold sensitivity."
            )
        if float(np.ptp(time_vals)) < 1e-9:
            print(
                f"[WARN M={m_value}] Time consumption is identical across beta in this run. "
                "This can happen when active-node pattern does not change."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Beta sensitivity evaluation for FL (no RL).")
    parser.add_argument("--rounds", type=int, default=1000, help="FL rounds for each beta (default: 1000)")
    parser.add_argument("--m-values", type=int, nargs="+", default=[9, 16, 25], help="List of M values")
    parser.add_argument("--beta-start", type=float, default=0.1, help="Start beta")
    parser.add_argument("--beta-end", type=float, default=0.9, help="End beta")
    parser.add_argument("--beta-step", type=float, default=0.1, help="Beta step")
    parser.add_argument(
        "--enable-early-stopping",
        action="store_true",
        help="Enable early stopping during beta sweep (default: disabled for fixed-round Figure 1-3 reproduction)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluate accuracy every N rounds during this experiment (default: 1 for exact Figure 2)",
    )
    parser.add_argument("--out-dir", type=str, default="results/beta_sensitivity", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "run_detail.log")
    log_fh = open(log_path, "w", encoding="utf-8")

    betas = np.round(
        np.arange(args.beta_start, args.beta_end + 1e-9, args.beta_step),
        10,
    )

    rows: list[dict] = []
    total_jobs = len(args.m_values) * len(betas)
    job_idx = 0

    _log("[INFO] Start beta sensitivity evaluation", log_fh)
    _log(f"[INFO] rounds={args.rounds}, m_values={args.m_values}, betas={betas.tolist()}", log_fh)
    _log(f"[INFO] early_stopping={'ON' if args.enable_early_stopping else 'OFF (fixed rounds)'}", log_fh)
    print_input_warnings(rounds=int(args.rounds), betas=betas, m_values=[int(m) for m in args.m_values])

    for m_value in args.m_values:
        cfg = build_config(m_value=m_value, max_rounds=args.rounds, eval_interval=args.eval_interval)

        for beta in betas:
            job_idx += 1
            t0 = time.perf_counter()

            _log(f"\n[JOB START] M={m_value}, beta={beta:.1f}", log_fh)
            communication_times, accuracy, time_consumption, energy_consumption, cost_consumption, reward_consumption, converged_round = evaluate_for_beta(
                cfg=cfg,
                beta=float(beta),
                rounds=args.rounds,
                enable_early_stopping=bool(args.enable_early_stopping),
                log_fh=log_fh,
            )

            elapsed = time.perf_counter() - t0
            row = {
                "M": int(m_value),
                "beta": float(beta),
                "communication_times": float(communication_times),
                "accuracy_round_1000": float(accuracy),
                "time_consumption": float(time_consumption),
                "energy_consumption": float(energy_consumption),
                "cost_consumption": float(cost_consumption),
                "reward_consumption": float(reward_consumption),
                "converged_round": int(converged_round),
                "elapsed_seconds": float(elapsed),
            }
            rows.append(row)

            _log(
                f"[PROGRESS {job_idx}/{total_jobs}] M={m_value}, beta={beta:.1f} "
                f"| acc={accuracy:.4f} | comm={communication_times:.1f} "
                f"| delay={time_consumption:.2f}s | energy={energy_consumption:.4f}J "
                f"| cost={cost_consumption:.4f} | reward={reward_consumption:.4f} "
                f"| rounds={converged_round} | elapsed={elapsed:.2f}s",
                log_fh,
            )

    log_fh.close()

    csv_path = os.path.join(args.out_dir, "beta_sensitivity_results.csv")
    save_csv(rows, csv_path)
    print_output_diagnostics(rows)

    # --- 7 đồ thị riêng biệt theo từng chỉ số ---
    plots = [
        ("fig1_communication_times.png", "communication_times",  "Communication Times",    "Fig 1: Communication Times vs beta"),
        ("fig2_accuracy.png",            "accuracy_round_1000",   "Accuracy",               "Fig 2: Accuracy vs beta"),
        ("fig3_delay.png",               "time_consumption",      "Avg Delay (s)",          "Fig 3: Avg Delay vs beta"),
        ("fig4_energy.png",              "energy_consumption",    "Avg Energy (J)",         "Fig 4: Avg Energy vs beta"),
        ("fig5_cost.png",                "cost_consumption",      "Avg Cost",               "Fig 5: Avg Cost vs beta"),
        ("fig6_reward.png",              "reward_consumption",    "Avg Reward",             "Fig 6: Avg Reward vs beta"),
        ("fig7_converged_round.png",     "converged_round",       "Converged Round",        "Fig 7: Converged Round vs beta"),
    ]
    saved_paths = [csv_path, log_path]
    for fname, key, ylabel, title in plots:
        out_path = os.path.join(args.out_dir, fname)
        plot_metric(rows=rows, metric_key=key, ylabel=ylabel, title=title, output_path=out_path)
        saved_paths.append(out_path)

    # Combined figure (latency + reward + cost)
    fig_combined = os.path.join(args.out_dir, "fig_combined_latency_reward_cost.png")
    plot_latency_reward_cost(rows, fig_combined)
    saved_paths.append(fig_combined)

    print("[DONE] Saved outputs:")
    for p in saved_paths:
        print(f"  - {p}")

if __name__ == "__main__":
    main()
