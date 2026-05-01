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
    
    metrics = ["time_consumption", "energy_consumption", "reward_consumption"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    M_values = sorted(df['M'].unique())
    
    for ax, metric in zip(axes, metrics):
        for m in M_values:
            subset = df[df['M'] == m].sort_values('beta')
            # Extract beta and the metric
            x = subset['beta']
            # Pandas handles duplicate columns by adding .1, so 'energy_consumption' refers to the first one
            y = subset[metric]
            
            # Using different markers for different M values
            markers = {9: 'o', 16: 's', 25: '^'}
            marker = markers.get(m, 'o')
            
            ax.plot(x, y, marker=marker, linewidth=2, label=f'M={m}')
            
        ax.set_xlabel('Beta')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs Beta')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "beta_metrics_plot.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Plot successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
