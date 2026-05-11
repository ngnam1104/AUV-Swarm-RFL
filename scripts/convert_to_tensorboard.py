import argparse
import os
import sys
import re

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("Vui lòng cài đặt thư viện tensorboard bằng lệnh: pip install tensorboard")
    sys.exit(1)

LOG_FILES = {
    "PPO": "ppo_steps.log",
    "SAC": "sac_steps.log",
    "TD3": "td3_steps.log",
    "DDPG": "ddpg_steps.log",
    "A2C": "a2c_steps.log",
}

BASELINE_LOG_FILES = {
    "Greedy": "greedy_episodes.log",
    "Random": "random_episodes.log",
}

def convert_rl_log(log_path: str, algo_name: str, tb_dir: str):
    pattern = re.compile(r'ep_step=(\d+)/\d+ .*?acc_cost=([\d\.-]+) .*?T=([\d\.]+)s \| E=([\d\.]+)J')
    
    writer = SummaryWriter(log_dir=os.path.join(tb_dir, algo_name))
    
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
                
                if step_idx < last_step:
                    ep_count += 1
                    
                    if ep_count % 10 == 0 or ep_count == 1:
                        # Ghi dữ liệu vào TensorBoard
                        writer.add_scalar("Metrics/1_Total_Cost", current_ep_acc_cost, ep_count)
                        writer.add_scalar("Metrics/2_Total_Latency", current_ep_t_sum, ep_count)
                        writer.add_scalar("Metrics/3_Total_Energy", current_ep_e_sum, ep_count)
                        
                    current_ep_t_sum = 0.0
                    current_ep_e_sum = 0.0
                    current_ep_acc_cost = 0.0
                
                current_ep_t_sum += t_val * 10
                current_ep_e_sum += e_val * 10
                current_ep_acc_cost = acc_cost_val
                
                last_step = step_idx
                
    writer.close()

def convert_baseline_log(log_path: str, algo_name: str, tb_dir: str):
    pattern = re.compile(r'Episode (\d+)/\d+ \| accumulated_cost=([\d\.]+) \| avg_delay=([\d\.]+)s \| avg_energy=([\d\.]+)J')
    
    writer = SummaryWriter(log_dir=os.path.join(tb_dir, algo_name))
    STEPS_PER_EPISODE = 1000
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ep_idx = int(m.group(1))
                acc_cost = float(m.group(2))
                avg_t = float(m.group(3))
                avg_e = float(m.group(4))
                
                writer.add_scalar("Metrics/1_Total_Cost", acc_cost, ep_idx)
                writer.add_scalar("Metrics/2_Total_Latency", avg_t * STEPS_PER_EPISODE, ep_idx)
                writer.add_scalar("Metrics/3_Total_Energy", avg_e * STEPS_PER_EPISODE, ep_idx)
                
    writer.close()

def main():
    parser = argparse.ArgumentParser(description="Convert TXT logs to TensorBoard format")
    parser.add_argument("--log-dir", type=str, default="results/logs/fig_7_bootstrap", help="Text logs folder")
    parser.add_argument("--tb-dir", type=str, default="results/tensorboard_logs", help="Output TensorBoard folder")
    args = parser.parse_args()

    os.makedirs(args.tb_dir, exist_ok=True)

    print(f"Reading logs from {args.log_dir}")
    print(f"Exporting to TensorBoard directory: {args.tb_dir}")

    # Convert RL
    for algo_name, filename in LOG_FILES.items():
        path = os.path.join(args.log_dir, filename)
        if os.path.exists(path):
            convert_rl_log(path, algo_name, args.tb_dir)
            print(f"- Processed {algo_name}")

    # Convert Baselines
    for algo_name, filename in BASELINE_LOG_FILES.items():
        path = os.path.join(args.log_dir, filename)
        if os.path.exists(path):
            convert_baseline_log(path, algo_name, args.tb_dir)
            print(f"- Processed {algo_name}")

    print("\n[DONE] Hoàn tất convert sang TensorBoard.")
    print("Khởi động TensorBoard bằng cách chạy lệnh sau trên Terminal:")
    print(f"    tensorboard --logdir {args.tb_dir}")

if __name__ == "__main__":
    main()
