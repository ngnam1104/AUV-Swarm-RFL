import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # File path
    csv_path = r"d:\Documents\HUST\2022-2026\Research_Thesis\AUV-Swarm-RFL\results\beta_sensitivity\beta_sensitivity_results.csv"
    output_dir = os.path.dirname(csv_path)
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    # Load data
    df = pd.read_csv(csv_path)
    
    M_values = sorted(df['M'].unique())
    markers = {9: 'o', 16: 's', 25: '^', 36: 'D', 49: 'v'}
    
    # 1. Vẽ 3 đồ thị gộp chung (tùy chọn theo code cũ)
    metrics_3 = ["time_consumption", "energy_consumption", "reward_consumption"]
    labels_3 = ["Time Consumption", "Energy Consumption", "Reward Consumption"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, label in zip(axes, metrics_3, labels_3):
        for m in M_values:
            subset = df[df['M'] == m].sort_values('beta')
            x = subset['beta']
            y = subset[metric]
            ax.plot(x, y, marker=markers.get(m, 'o'), linewidth=2, label=f'M={m}')
            
        ax.set_xlabel('Beta')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Vs Beta'.title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    output_path = os.path.join(output_dir, "beta_metrics_subplot.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    # 2. Vẽ 7 đồ thị riêng lẻ theo format y chang ảnh đính kèm
    individual_metrics = [
        ("communication_times", "Communication Times", "Communication Times Vs Beta", "fig1_comm_times.png"),
        ("accuracy_round_1000", "Accuracy", "Accuracy Vs Beta", "fig2_accuracy.png"),
        ("time_consumption", "Average Latency (s)", "Average Latency Vs Beta", "fig3_avg_delay.png"),
        ("energy_consumption", "Average Energy (J)", "Average Energy Vs Beta", "fig4_avg_energy.png"),
        ("cost_consumption", "Average Cost", "Average Cost Vs Beta", "fig5_avg_cost.png"),
        ("reward_consumption", "Average Reward", "Average Reward Vs Beta", "fig6_avg_reward.png"),
        ("converged_round", "Converged Round", "Converged Round Vs Beta", "fig7_converged_round.png")
    ]
    
    for metric_col, ylabel, title, out_filename in individual_metrics:
        plt.figure(figsize=(8, 5))
        for m in M_values:
            subset = df[df['M'] == m].sort_values('beta')
            x = subset['beta']
            y = subset[metric_col]
            plt.plot(x, y, marker=markers.get(m, 'o'), linewidth=2, label=f'M={m}')
            
        plt.xlabel('Beta')
        # ylabel và title đã được định dạng chuẩn từ mảng individual_metrics
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, out_filename)
        plt.savefig(out_path, dpi=150)
        plt.close()
        
        print(f"Saved: {out_path}")
        
    print("All plots successfully saved!")

if __name__ == "__main__":
    main()
