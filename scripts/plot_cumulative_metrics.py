import argparse
import os
import sys
import re

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

LOG_FILES = {
    "PPO (Proposed)": "ppo_steps.log",
    "SAC":            "sac_steps.log",
    "TD3":            "td3_steps.log",
    "DDPG":           "ddpg_steps.log",
    "A2C":            "a2c_steps.log",
}

BASELINE_LOG_FILES = {
    "Greedy": "greedy_episodes.log",
    "Random": "random_episodes.log",
}

def parse_rl_log(log_path: str, metric_type: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses RL step logs (like ppo_steps.log).
    metric_type can be "cost", "delay", "energy".
    Returns: (episodes, cumulative_values)
    """
    pattern = re.compile(r'ep_step=(\d+)/\d+ .*?acc_cost=([\d\.-]+) .*?T=([\d\.]+)s \| E=([\d\.]+)J')
    
    episodes = []
    cum_values = []
    
    ep_count = 0
    last_step = 0
    
    current_ep_t_sum = 0.0
    current_ep_e_sum = 0.0
    current_ep_acc_cost = 0.0
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step_idx = int(m.group(1))
                acc_cost_val = float(m.group(2))
                t_val = float(m.group(3))
                e_val = float(m.group(4))
                
                # Check episode boundary (e.g. 1000 -> 10)
                if step_idx < last_step:
                    ep_count += 1
                    
                    # Cứ thế vẽ sau mỗi 10 ep
                    if ep_count % 10 == 0:
                        episodes.append(ep_count)
                        if metric_type == "cost":
                            cum_values.append(current_ep_acc_cost)  # cost đã được cộng dồn nội bộ ep
                        elif metric_type == "delay":
                            cum_values.append(current_ep_t_sum)     # tổng T trong 1 ep
                        elif metric_type == "energy":
                            cum_values.append(current_ep_e_sum)     # tổng E trong 1 ep
                            
                    current_ep_t_sum = 0.0
                    current_ep_e_sum = 0.0
                    current_ep_acc_cost = 0.0
                
                # We multiply by 10 because it prints every 10 steps
                current_ep_t_sum += t_val * 10
                current_ep_e_sum += e_val * 10
                current_ep_acc_cost = acc_cost_val
                
                last_step = step_idx
                
    # Bỏ qua tập (episode) cuối cùng vì chưa chạy đủ bước
    return np.array(episodes), np.array(cum_values)

def parse_baseline_log(log_path: str, metric_type: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses baseline logs (like greedy_episodes.log), which print every 10 episodes.
    metric_type can be "cost", "delay", "energy".
    Returns: (episodes, cumulative_values)
    """
    pattern = re.compile(r'Episode (\d+)/\d+ \| accumulated_cost=([\d\.]+) \| avg_delay=([\d\.]+)s \| avg_energy=([\d\.]+)J')
    
    episodes = []
    cum_values = []
    
    STEPS_PER_EPISODE = 1000
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ep_idx = int(m.group(1))
                
                if ep_idx == 0:
                    continue
                    
                acc_cost = float(m.group(2))
                avg_t = float(m.group(3))
                avg_e = float(m.group(4))
                
                episodes.append(ep_idx)
                if metric_type == "cost":
                    cum_values.append(acc_cost)  # Cost plot trực tiếp
                elif metric_type == "delay":
                    cum_values.append(avg_t * STEPS_PER_EPISODE)  # Tổng delay 1 ep
                elif metric_type == "energy":
                    cum_values.append(avg_e * STEPS_PER_EPISODE)  # Tổng energy 1 ep
                
    return np.array(episodes), np.array(cum_values)

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Cumulative metrics across episodes directly from logs.")
    parser.add_argument("--log-dir", type=str, default="results/logs/fig_7_bootstrap", help="Directory containing log files")
    parser.add_argument("--out-dir", type=str, default="results/fig_7", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    metrics_to_plot = [
        ("cost", "Total Cost", "total_cost.png"),
        ("delay", "Total Latency (s)", "total_latency.png"),
        ("energy", "Total Energy (J)", "total_energy.png"),
    ]

    has_data = False
    ppo_combined_data = {}
    all_algo_combined_data = {}

    for metric_type, ylabel, out_filename in metrics_to_plot:
        # TẤT CẢ THUẬT TOÁN
        all_algo_combined_data[metric_type] = {'ylabel': ylabel, 'data': []}
        plt.figure(figsize=(10, 6))
        
        ppo_episodes, ppo_values = None, None
        
        # RL Algorithms
        for label, filename in LOG_FILES.items():
            path = os.path.join(args.log_dir, filename)
            if os.path.exists(path):
                episodes, cum_values = parse_rl_log(path, metric_type)
                if len(episodes) > 0:
                    plt.plot(episodes, cum_values, linewidth=2, label=label)
                    all_algo_combined_data[metric_type]['data'].append((label, episodes, cum_values))
                    has_data = True
                    if label == "PPO (Proposed)":
                        ppo_episodes, ppo_values = episodes, cum_values
                    
        # Baseline Algorithms
        for label, filename in BASELINE_LOG_FILES.items():
            path = os.path.join(args.log_dir, filename)
            if os.path.exists(path):
                episodes, cum_values = parse_baseline_log(path, metric_type)
                if len(episodes) > 0:
                    plt.plot(episodes, cum_values, linewidth=2, label=label)
                    all_algo_combined_data[metric_type]['data'].append((label, episodes, cum_values))
                    has_data = True

        if not has_data:
            print(f"[WARN] No log files found in {args.log_dir}. Please check your path.")
            plt.close()
            return
            
        plt.xlabel("Episodes")
        plt.ylabel(ylabel)
        plt.title(f"Performance Metrics - {ylabel}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, out_filename)
        plt.savefig(out_path, dpi=150)
        plt.close()
        
        # VẼ RIÊNG CHO PPO
        if ppo_episodes is not None and ppo_values is not None:
            ppo_combined_data[metric_type] = (ppo_episodes, ppo_values, ylabel)
            
            plt.figure(figsize=(10, 6))
            plt.plot(ppo_episodes, ppo_values, linewidth=2, label="PPO (Proposed)", color='tab:blue')
            plt.xlabel("Episodes")
            plt.ylabel(ylabel)
            plt.title(f"Performance Metrics (PPO Only) - {ylabel}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            out_path_ppo = os.path.join(args.out_dir, f"ppo_only_{out_filename}")
            plt.savefig(out_path_ppo, dpi=150)
            plt.close()

        print(f"[DONE] Saved figure: {out_path}")

    # VẼ COMBINED 3 TRONG 1 CHO PPO MÀ THÔI
    if len(ppo_combined_data) == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (m_type, yl, _out) in zip(axes, metrics_to_plot):
            eps, vals, y_label = ppo_combined_data[m_type]
            ax.plot(eps, vals, linewidth=2, label="PPO (Proposed)", color='tab:blue')
            ax.set_xlabel("Episodes")
            ax.set_ylabel(y_label)
            ax.set_title(f"PPO Only: {y_label}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        plt.tight_layout()
        out_path_combined = os.path.join(args.out_dir, "ppo_only_combined.png")
        plt.savefig(out_path_combined, dpi=150)
        plt.close()
        print(f"[DONE] Saved combined figure: {out_path_combined}")
        
    # VẼ COMBINED 3 TRONG 1 CHO TẤT CẢ CÁC THUẬT TOÁN
    if len(all_algo_combined_data) == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (m_type, yl, _out) in zip(axes, metrics_to_plot):
            y_label = all_algo_combined_data[m_type]['ylabel']
            data_list = all_algo_combined_data[m_type]['data']
            for label, eps, vals in data_list:
                ax.plot(eps, vals, linewidth=1.5, label=label)
            ax.set_xlabel("Episodes")
            ax.set_ylabel(y_label)
            ax.set_title(f"Performance Metrics - {y_label}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        plt.tight_layout()
        out_path_combined_all = os.path.join(args.out_dir, "total_combined.png")
        plt.savefig(out_path_combined_all, dpi=150)
        plt.close()
        print(f"[DONE] Saved combined figure (All algorithms): {out_path_combined_all}")

if __name__ == "__main__":
    main()
