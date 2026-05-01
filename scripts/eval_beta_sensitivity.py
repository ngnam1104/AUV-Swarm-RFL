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
    time_consumption_first_5 = 0.0
    energy_consumption_first_5 = 0.0
    cost_consumption_first_5 = 0.0
    reward_consumption_first_5 = 0.0
    final_accuracy = 0.0

    for rnd in range(1, rounds + 1):
        accuracy, active_indices, _, is_converged = fl_sim.sync_run_step(beta=beta, rnd=rnd)

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
            print(f"    [Started] beta={beta:.1f} | First round training...", flush=True)

        communication_times += float(np.sum(lambda_m))
        total_time_consumption += float(t_total)
        if rnd <= 5:
            time_consumption_first_5 += float(t_total)
            energy_consumption_first_5 += float(E_total)
            cost_consumption_first_5 += float(cost)
            reward_consumption_first_5 += float(reward)
        final_accuracy = float(accuracy)

        if rnd % 5 == 0 or rnd == rounds or (enable_early_stopping and is_converged):
            print(f"    [Round {rnd}/{rounds}] beta={beta:.1f} | acc={final_accuracy:.4f} | comm={communication_times:.0f} | delay={t_total:.2f}s", flush=True)

        if enable_early_stopping and is_converged:
            print(f"    [Early Stopping] Converged at round {rnd} with acc={final_accuracy:.4f}", flush=True)
            break

    avg_time_first_5 = time_consumption_first_5 / min(rounds, 5) if rounds > 0 else 0.0
    avg_energy_first_5 = energy_consumption_first_5 / min(rounds, 5) if rounds > 0 else 0.0
    avg_cost_first_5 = cost_consumption_first_5 / min(rounds, 5) if rounds > 0 else 0.0
    avg_reward_first_5 = reward_consumption_first_5 / min(rounds, 5) if rounds > 0 else 0.0
    return communication_times, final_accuracy, avg_time_first_5, avg_energy_first_5, avg_cost_first_5, avg_reward_first_5


def save_csv(rows: list[dict], output_csv: str) -> None:
    if not rows:
        return

    fieldnames = [
        "M",
        "beta",
        "communication_times",
        "accuracy_round_1000",
        "time_consumption",
        "energy_consumption",
        "cost_consumption",
        "reward_consumption",
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

    betas = np.round(
        np.arange(args.beta_start, args.beta_end + 1e-9, args.beta_step),
        10,
    )

    rows: list[dict] = []
    total_jobs = len(args.m_values) * len(betas)
    job_idx = 0

    print("[INFO] Start beta sensitivity evaluation")
    print(f"[INFO] rounds={args.rounds}, m_values={args.m_values}, betas={betas.tolist()}")
    print(f"[INFO] early_stopping={'ON' if args.enable_early_stopping else 'OFF (fixed rounds)'}")
    print_input_warnings(rounds=int(args.rounds), betas=betas, m_values=[int(m) for m in args.m_values])

    for m_value in args.m_values:
        cfg = build_config(m_value=m_value, max_rounds=args.rounds, eval_interval=args.eval_interval)

        for beta in betas:
            job_idx += 1
            t0 = time.perf_counter()

            print(f"\n[JOB START] M={m_value}, beta={beta:.1f}", flush=True)
            communication_times, accuracy, time_consumption, energy_consumption, cost_consumption, reward_consumption = evaluate_for_beta(
                cfg=cfg,
                beta=float(beta),
                rounds=args.rounds,
                enable_early_stopping=bool(args.enable_early_stopping),
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
                "elapsed_seconds": float(elapsed),
            }
            rows.append(row)

            print(
                f"[PROGRESS {job_idx}/{total_jobs}] M={m_value}, beta={beta:.1f} | "
                f"comm={communication_times:.1f}, acc={accuracy:.4f}, T_total={time_consumption:.2f}, elapsed={elapsed:.2f}s"
            )

    csv_path = os.path.join(args.out_dir, "beta_sensitivity_results.csv")
    save_csv(rows, csv_path)
    print_output_diagnostics(rows)

    # Figure 1: Communication times vs beta
    # fig1_path = os.path.join(args.out_dir, "figure1_communication_times.png")
    # plot_metric(
    #     rows=rows,
    #     metric_key="communication_times",
    #     ylabel="Communication times",
    #     title="Figure 1: Communication times vs beta",
    #     output_path=fig1_path,
    # )

    # Figure 2: Accuracy vs beta
    # fig2_path = os.path.join(args.out_dir, "figure2_accuracy.png")
    # plot_metric(
    #     rows=rows,
    #     metric_key="accuracy_round_1000",
    #     ylabel="Accuracy",
    #     title="Figure 2: Accuracy at round 1000 vs beta",
    #     output_path=fig2_path,
    # )

    # Figure 3: Latency, Reward, and Cost vs beta
    fig3_path = os.path.join(args.out_dir, "figure3_latency_reward_cost.png")
    plot_latency_reward_cost(rows, fig3_path)

    print("[DONE] Saved outputs:")
    print(f"- {csv_path}")
    print(f"- {fig3_path}")


if __name__ == "__main__":
    main()
