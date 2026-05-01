import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from dataclasses import asdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.settings import ACOUSTIC_CFG, HW_CFG, FL_CFG, RL_CFG
from env.communication import CommunicationModel
from env.energy import EnergyModel
from env.latency import LatencyModel
from env.reward import RewardModel

def build_config():
    cfg = SimpleNamespace(
        **asdict(ACOUSTIC_CFG),
        **asdict(HW_CFG),
        **asdict(FL_CFG),
        **asdict(RL_CFG),
    )
    return cfg

def main():
    cfg = build_config()
    comm_model = CommunicationModel(cfg)
    latency_model = LatencyModel(cfg, comm_model)
    energy_model = EnergyModel(cfg)
    reward_model = RewardModel(cfg)

    # Lấy giới hạn vật lý từ config
    p_min, p_max = 0.01, cfg.p_max
    f_min, f_max = cfg.f_min, cfg.f_max
    
    # Đặt giá trị trung vị cho các tham số bị giữ cố định
    p_mid = (p_min + p_max) / 2.0
    f_mid = (f_min + f_max) / 2.0

    M = cfg.M
    # Giả định tất cả M node đều tham gia (beta = 1.0) để đo lường vật lý thuần túy
    lambda_m = np.ones(M, dtype=float)

    def evaluate_state(p_m_val, f_m_val, p_L_val, f_L_val):
        """Tính toán E, T, Cost cho một tập hợp tham số vật lý."""
        p_m_arr = np.full(M, p_m_val)
        f_m_arr = np.full(M, f_m_val)
        
        # 1. Tính toán Latency và Energy
        E_total, energy_details, T_total, latency_details = energy_model.compute_total_energy_from_latency(
            latency_model=latency_model,
            lambda_m=lambda_m,
            f_m=f_m_arr,
            p_m=p_m_arr,
            f_L=f_L_val,
            p_L=p_L_val
        )
        
        E_m_array = energy_details["E_Cp_m"] + energy_details["E_C_m"]
        E_L_val_total = energy_details["E_Cp_L"] + energy_details["E_C_L"]
        
        # 2. Tính Cost (bỏ qua reward vì ta chỉ quan tâm hàm mục tiêu Cost = phi*E + chi*T)
        _, cost, _ = reward_model.compute_reward(
            T_total=T_total,
            E_total=E_total,
            E_m=E_m_array,
            E_L=E_L_val_total
        )
        return float(E_total), float(T_total), float(cost)

    # Định nghĩa kịch bản test cho 4 tham số với dải rộng hơn để thấy rõ cực trị
    params_to_test = {
        "f_m": {
            "range": np.linspace(0.1e9, 3.0e9, 100), 
            "default_args": lambda v: (p_mid, v, p_mid, f_mid), 
            "label": "Follower CPU Frequency (Hz)",
            "x_scale": 1e9, "x_unit": "GHz"
        },
        "p_m": {
            "range": np.linspace(0.005, 5.0, 100), 
            "default_args": lambda v: (v, f_mid, p_mid, f_mid), 
            "label": "Follower Transmit Power (W)",
            "x_scale": 1.0, "x_unit": "W"
        },
        "f_L": {
            "range": np.linspace(0.1e9, 3.0e9, 100), 
            "default_args": lambda v: (p_mid, f_mid, p_mid, v), 
            "label": "Leader CPU Frequency (Hz)",
            "x_scale": 1e9, "x_unit": "GHz"
        },
        "p_L": {
            "range": np.linspace(0.005, 5.0, 100), 
            "default_args": lambda v: (p_mid, f_mid, v, f_mid), 
            "label": "Leader Transmit Power (W)",
            "x_scale": 1.0, "x_unit": "W"
        },
    }

    out_dir = "results/physical_sensitivity"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Start sweeping physical parameters. Saving to {out_dir}")

    for param_name, info in params_to_test.items():
        x_vals_raw = info["range"]
        x_vals_plot = x_vals_raw / info["x_scale"]
        
        E_vals, T_vals = [], []
        
        for val in x_vals_raw:
            args = info["default_args"](val)
            E, T, _ = evaluate_state(*args)
            E_vals.append(E)
            T_vals.append(T)
            
        E_vals = np.array(E_vals)
        T_vals = np.array(T_vals)
        
        # Min-Max Normalization để vẽ giao điểm đẹp nhất [0, 1]
        E_norm = (E_vals - np.min(E_vals)) / (np.max(E_vals) - np.min(E_vals))
        T_norm = (T_vals - np.min(T_vals)) / (np.max(T_vals) - np.min(T_vals))
        Cost_vals = E_norm + T_norm
        
        plt.figure(figsize=(18, 5))
        
        # Plot 1: Total Energy
        plt.subplot(1, 3, 1)
        plt.plot(x_vals_plot, E_vals, color='#2ca02c', lw=2.5)
        plt.xlabel(f'{info["label"]} [{info["x_unit"]}]', fontsize=12)
        plt.ylabel("Total Energy (Joules)", fontsize=12)
        plt.title(f"Energy vs {param_name}", fontsize=14, pad=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Total Latency
        plt.subplot(1, 3, 2)
        plt.plot(x_vals_plot, T_vals, color='#1f77b4', lw=2.5)
        plt.xlabel(f'{info["label"]} [{info["x_unit"]}]', fontsize=12)
        plt.ylabel("Total Latency (Seconds)", fontsize=12)
        plt.title(f"Latency vs {param_name}", fontsize=14, pad=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 3: Normalized Cost (Trade-off Visualization)
        plt.subplot(1, 3, 3)
        plt.plot(x_vals_plot, E_norm, color='#2ca02c', lw=1.5, linestyle=':', label='Norm. Energy')
        plt.plot(x_vals_plot, T_norm, color='#1f77b4', lw=1.5, linestyle=':', label='Norm. Latency')
        plt.plot(x_vals_plot, Cost_vals, color='#d62728', lw=3, label='Cost (Sum)')
        plt.xlabel(f'{info["label"]} [{info["x_unit"]}]', fontsize=12)
        plt.ylabel("Normalized Scale [0, 1]", fontsize=12)
        plt.title(f"Trade-off U-Shape: Cost vs {param_name}", fontsize=14, pad=10)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"sensitivity_{param_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  -> [DONE] Generated {param_name}. Saved to: {out_path}")

if __name__ == "__main__":
    main()
