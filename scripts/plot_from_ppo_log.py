#!/usr/bin/env python3
"""
Parse ppo_steps.log and plot latency, energy, cost metrics.
Handles multiple training runs within the same log file.
"""

import re
import os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


def smooth_data(data: List[float], window_size: int = 5) -> List[float]:
    """Apply moving average smoothing to data."""
    if len(data) < window_size:
        return data
    return uniform_filter1d(data, size=window_size, mode='nearest').tolist()


def parse_ppo_log(log_path: str) -> Dict[int, Dict[str, List[float]]]:
    """
    Parse ppo_steps.log and extract metrics, grouped by episode.
    
    Returns:
        Dict mapping run_id -> {episode_id -> {metric: [values]}}
        Metrics: latency (T), energy (E), cost, reward, acc_cost
    """
    metrics = {}
    current_run = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            # Detect new training run
            if '=== PPO baseline training started' in line:
                current_run += 1
                metrics[current_run] = {}
                current_episode = None
                continue
            
            # Parse FL-RL Step lines
            if '[FL-RL Step' in line and current_run > 0:
                try:
                    # Extract step number
                    step_match = re.search(r'\[FL-RL Step (\d+)\]', line)
                    if not step_match:
                        continue
                    step = int(step_match.group(1))
                    
                    # Extract episode step
                    ep_step_match = re.search(r'ep_step=(\d+)/(\d+)', line)
                    if ep_step_match:
                        ep_step = int(ep_step_match.group(1))
                        ep_total = int(ep_step_match.group(2))
                        # Detect new episode (ep_step resets or is small)
                        if current_episode is None or ep_step < 20:
                            current_episode = len(metrics[current_run])
                            metrics[current_run][current_episode] = {
                                'step': [],
                                'ep_step': [],
                                'latency': [],      # T in seconds
                                'energy': [],        # E in Joules
                                'cost': [],
                                'reward': [],
                                'acc_cost': [],
                                'accuracy': [],
                                'active_auvs': []
                            }
                    
                    if current_episode is None:
                        continue
                    
                    # Extract metrics using regex
                    reward_match = re.search(r'reward=([-\d.]+)', line)
                    cost_match = re.search(r'cost=([-\d.]+)', line)
                    acc_cost_match = re.search(r'acc_cost=([-\d.]+)', line)
                    active_match = re.search(r'active=(\d+)', line)
                    acc_match = re.search(r'acc=([-\d.]+)', line)
                    t_match = re.search(r'T=([-\d.]+)s', line)
                    e_match = re.search(r'E=([-\d.]+)J', line)
                    
                    if all([reward_match, cost_match, acc_cost_match, t_match, e_match]):
                        metrics[current_run][current_episode]['step'].append(step)
                        if ep_step_match:
                            metrics[current_run][current_episode]['ep_step'].append(int(ep_step_match.group(1)))
                        metrics[current_run][current_episode]['reward'].append(float(reward_match.group(1)))
                        metrics[current_run][current_episode]['cost'].append(float(cost_match.group(1)))
                        metrics[current_run][current_episode]['acc_cost'].append(float(acc_cost_match.group(1)))
                        metrics[current_run][current_episode]['latency'].append(float(t_match.group(1)))
                        metrics[current_run][current_episode]['energy'].append(float(e_match.group(1)))
                        if active_match:
                            metrics[current_run][current_episode]['active_auvs'].append(int(active_match.group(1)))
                        if acc_match:
                            metrics[current_run][current_episode]['accuracy'].append(float(acc_match.group(1)))
                except Exception as e:
                    print(f"[WARN] Failed to parse line: {line[:100]}... Error: {e}")
                    continue
    
    # Remove empty runs
    metrics = {k: v for k, v in metrics.items() if v}
    return metrics


def plot_metrics(metrics: Dict[int, Dict[int, Dict[str, List[float]]]], output_dir: str, smooth_window: int = 5) -> None:
    """Plot metrics from all episodes with smoothing."""
    os.makedirs(output_dir, exist_ok=True)
    
    for run_id, episodes in metrics.items():
        print(f"\n[RUN {run_id}] Generating {len(episodes)} episode plots...")
        
        # Plot 1: Latency vs Episode Step
        plt.figure(figsize=(14, 7))
        for ep_id, data in episodes.items():
            if data['ep_step']:
                y = smooth_data(data['latency'], window_size=smooth_window)
                plt.plot(data['ep_step'], y, linewidth=2, alpha=0.7, label=f'Episode {ep_id}')
        plt.xlabel('Episode Step')
        plt.ylabel('Latency (seconds)')
        plt.title(f'Run {run_id}: Average Latency per Episode Step (Smoothed)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'run{run_id}_latency_per_episode.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved: run{run_id}_latency_per_episode.png")
        
        # Plot 2: Energy vs Episode Step
        plt.figure(figsize=(14, 7))
        for ep_id, data in episodes.items():
            if data['ep_step']:
                y = smooth_data(data['energy'], window_size=smooth_window)
                plt.plot(data['ep_step'], y, linewidth=2, alpha=0.7, label=f'Episode {ep_id}')
        plt.xlabel('Episode Step')
        plt.ylabel('Energy (Joules)')
        plt.title(f'Run {run_id}: Average Energy per Episode Step (Smoothed)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'run{run_id}_energy_per_episode.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved: run{run_id}_energy_per_episode.png")
        
        # Plot 3: Cost vs Episode Step
        plt.figure(figsize=(14, 7))
        for ep_id, data in episodes.items():
            if data['ep_step']:
                y = smooth_data(data['cost'], window_size=smooth_window)
                plt.plot(data['ep_step'], y, linewidth=2, alpha=0.7, label=f'Episode {ep_id}')
        plt.xlabel('Episode Step')
        plt.ylabel('Cost')
        plt.title(f'Run {run_id}: Cost per Episode Step (Smoothed)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'run{run_id}_cost_per_episode.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved: run{run_id}_cost_per_episode.png")
        
        # Plot 4: Accumulated Cost vs Episode Step
        plt.figure(figsize=(14, 7))
        for ep_id, data in episodes.items():
            if data['ep_step']:
                plt.plot(data['ep_step'], data['acc_cost'], linewidth=2, alpha=0.7, label=f'Episode {ep_id}')
        plt.xlabel('Episode Step')
        plt.ylabel('Accumulated Cost')
        plt.title(f'Run {run_id}: Accumulated Cost Over Episode Steps')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'run{run_id}_acc_cost_per_episode.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved: run{run_id}_acc_cost_per_episode.png")
        
        # Plot 5: Reward vs Episode Step
        plt.figure(figsize=(14, 7))
        for ep_id, data in episodes.items():
            if data['ep_step']:
                y = smooth_data(data['reward'], window_size=smooth_window)
                plt.plot(data['ep_step'], y, linewidth=2, alpha=0.7, label=f'Episode {ep_id}', marker='.')
        plt.xlabel('Episode Step')
        plt.ylabel('Reward')
        plt.title(f'Run {run_id}: Reward per Episode Step (Smoothed)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'run{run_id}_reward_per_episode.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved: run{run_id}_reward_per_episode.png")
        
        # Plot 6: Accuracy vs Episode Step (if available)
        has_accuracy = any(episodes[ep]['accuracy'] for ep in episodes)
        if has_accuracy:
            plt.figure(figsize=(14, 7))
            for ep_id, data in episodes.items():
                if data['ep_step'] and data['accuracy']:
                    y = smooth_data(data['accuracy'], window_size=smooth_window)
                    plt.plot(data['ep_step'], y, linewidth=2, alpha=0.7, label=f'Episode {ep_id}')
            plt.xlabel('Episode Step')
            plt.ylabel('Model Accuracy')
            plt.title(f'Run {run_id}: Model Accuracy per Episode Step (Smoothed)')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'run{run_id}_accuracy_per_episode.png'), dpi=150)
            plt.close()
            print(f"  ✓ Saved: run{run_id}_accuracy_per_episode.png")
        
        # Plot 7: All metrics in one figure (first 5 episodes only for clarity)
        episodes_to_plot = list(episodes.items())[:5]
        n_episodes = len(episodes_to_plot)
        
        fig, axes = plt.subplots(n_episodes, 3, figsize=(18, 5*n_episodes))
        if n_episodes == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (ep_id, data) in enumerate(episodes_to_plot):
            if not data['ep_step']:
                continue
            
            # Latency
            y_lat = smooth_data(data['latency'], window_size=smooth_window)
            axes[idx, 0].plot(data['ep_step'], y_lat, linewidth=2, color='#1f77b4')
            axes[idx, 0].set_ylabel('Latency (s)', fontsize=10)
            axes[idx, 0].set_title(f'Episode {ep_id} - Latency', fontsize=11)
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Energy
            y_eng = smooth_data(data['energy'], window_size=smooth_window)
            axes[idx, 1].plot(data['ep_step'], y_eng, linewidth=2, color='#ff7f0e')
            axes[idx, 1].set_ylabel('Energy (J)', fontsize=10)
            axes[idx, 1].set_title(f'Episode {ep_id} - Energy', fontsize=11)
            axes[idx, 1].grid(True, alpha=0.3)
            
            # Cost
            y_cost = smooth_data(data['cost'], window_size=smooth_window)
            axes[idx, 2].plot(data['ep_step'], y_cost, linewidth=2, color='#2ca02c')
            axes[idx, 2].set_ylabel('Cost', fontsize=10)
            axes[idx, 2].set_title(f'Episode {ep_id} - Cost', fontsize=11)
            axes[idx, 2].grid(True, alpha=0.3)
            
            axes[idx, 0].set_xlabel('Episode Step', fontsize=10)
            axes[idx, 1].set_xlabel('Episode Step', fontsize=10)
            axes[idx, 2].set_xlabel('Episode Step', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'run{run_id}_all_metrics_episodes.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved: run{run_id}_all_metrics_episodes.png")

        # Plot 8: Cumulative metrics for lowest-cost episode
        best_ep_id = None
        best_ep_mean_cost = None
        for ep_id, data in episodes.items():
            if data['cost']:
                mean_cost = float(np.mean(data['cost']))
                if best_ep_mean_cost is None or mean_cost < best_ep_mean_cost:
                    best_ep_mean_cost = mean_cost
                    best_ep_id = ep_id

        if best_ep_id is not None:
            best_data = episodes[best_ep_id]
            if best_data['ep_step']:
                cum_cost = np.cumsum(best_data['cost'])
                cum_energy = np.cumsum(best_data['energy'])
                cum_latency = np.cumsum(best_data['latency'])

                plt.figure(figsize=(14, 7))
                plt.plot(best_data['ep_step'], cum_cost, linewidth=2, label='Cumulative Cost')
                plt.plot(best_data['ep_step'], cum_energy, linewidth=2, label='Cumulative Energy')
                plt.plot(best_data['ep_step'], cum_latency, linewidth=2, label='Cumulative Latency')
                plt.xlabel('Episode Step')
                plt.ylabel('Cumulative Value')
                plt.title(f'Run {run_id}: Cumulative Metrics (Lowest-Cost Episode {best_ep_id})')
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'run{run_id}_cumulative_best_episode.png'), dpi=150)
                plt.close()
                print(f"  ✓ Saved: run{run_id}_cumulative_best_episode.png")


def print_statistics(metrics: Dict[int, Dict[int, Dict[str, List[float]]]]) -> None:
    """Print summary statistics for each run and episode."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for run_id, episodes in metrics.items():
        print(f"\n[RUN {run_id}] Total episodes: {len(episodes)}")
        
        for ep_id, data in episodes.items():
            print(f"\n  [EPISODE {ep_id}]")
            print(f"    Total steps: {len(data['step'])}")
            if data['ep_step']:
                print(f"    Episode step range: {min(data['ep_step'])} - {max(data['ep_step'])}")
            
            if data['latency']:
                print(f"    Latency - Mean: {np.mean(data['latency']):.4f}s, "
                      f"Min: {np.min(data['latency']):.4f}s, Max: {np.max(data['latency']):.4f}s")
            
            if data['energy']:
                print(f"    Energy  - Mean: {np.mean(data['energy']):.4f}J, "
                      f"Min: {np.min(data['energy']):.4f}J, Max: {np.max(data['energy']):.4f}J")
            
            if data['cost']:
                print(f"    Cost    - Mean: {np.mean(data['cost']):.4f}, "
                      f"Min: {np.min(data['cost']):.4f}, Max: {np.max(data['cost']):.4f}")
            
            if data['reward']:
                print(f"    Reward  - Mean: {np.mean(data['reward']):.4f}, "
                      f"Min: {np.min(data['reward']):.4f}, Max: {np.max(data['reward']):.4f}")
            
            if data['acc_cost']:
                print(f"    Acc Cost- Final: {data['acc_cost'][-1]:.4f}")
            
            if data['accuracy']:
                print(f"    Accuracy- Mean: {np.mean(data['accuracy']):.4f}, "
                      f"Final: {data['accuracy'][-1]:.4f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot metrics from ppo_steps.log")
    parser.add_argument("--log-file", type=str, 
                       default="results/logs/fig_7_bootstrap/ppo_steps.log",
                       help="Path to ppo_steps.log")
    parser.add_argument("--output-dir", type=str,
                       default="results/plots_from_log",
                       help="Output directory for plots")
    parser.add_argument("--smooth-window", type=int, default=5,
                       help="Window size for smoothing (default: 5)")
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"❌ Log file not found: {args.log_file}")
        return
    
    print(f"📖 Parsing log file: {args.log_file}")
    metrics = parse_ppo_log(args.log_file)
    
    if not metrics:
        print("❌ No metrics found in log file")
        return
    
    total_episodes = sum(len(episodes) for episodes in metrics.values())
    print(f"✓ Found {len(metrics)} training run(s) with {total_episodes} total episode(s)")
    
    print_statistics(metrics)
    
    print(f"\n📊 Generating plots with smoothing window={args.smooth_window}...")
    plot_metrics(metrics, args.output_dir, smooth_window=args.smooth_window)
    
    print(f"\n✓ All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
