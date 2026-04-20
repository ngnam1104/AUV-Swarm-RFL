from dataclasses import asdict
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from config.settings import ACOUSTIC_CFG, FL_CFG, HW_CFG, RL_CFG
from env.auv_env import AUVSwarmEnv
from fl_core.simulator import FLSimulator


def build_full_config() -> SimpleNamespace:
	return SimpleNamespace(
		**asdict(ACOUSTIC_CFG),
		**asdict(HW_CFG),
		**asdict(FL_CFG),
		**asdict(RL_CFG),
	)


def main():
	config = build_full_config()
	fl_sim = FLSimulator(config)
	env = AUVSwarmEnv(fl_sim=fl_sim, config=config)

	model = PPO.load("./models/ppo_auv_final")

	obs, _ = env.reset()
	done = False

	accumulated_costs = []
	rewards = []
	active_nodes = []
	t_totals = []
	accuracies = []

	while not done:
		action, _states = model.predict(obs, deterministic=True)
		obs, reward, terminated, truncated, info = env.step(action)
		done = bool(terminated or truncated)

		accumulated_costs.append(float(info.get("accumulated_cost", np.nan)))
		rewards.append(float(reward))
		active_nodes.append(float(info.get("active_nodes", np.nan)))
		t_totals.append(float(info.get("T_total", np.nan)))
		accuracies.append(float(info.get("accuracy", np.nan)))

	cumulative_active_nodes = np.cumsum(active_nodes)
	x = np.arange(len(accumulated_costs))
	fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

	axes[0].plot(x, accumulated_costs, color="tab:blue", linewidth=2)
	axes[0].set_ylabel("Accumulated Cost")
	axes[0].set_title("Episode Step vs Accumulated Cost (Eq. 33)")
	axes[0].grid(True, alpha=0.3)

	axes[1].plot(x, cumulative_active_nodes, color="tab:orange", linewidth=2)
	axes[1].set_ylabel("Cumulative Comm. Times")
	axes[1].set_title("Episode Step vs Cumulative Communication Times")
	axes[1].grid(True, alpha=0.3)

	axes[2].plot(x, accuracies, color="tab:green", linewidth=2)
	axes[2].set_xlabel("Episode Step")
	axes[2].set_ylabel("Accuracy")
	axes[2].set_title("Episode Step vs Accuracy")
	axes[2].grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig("eval_results.png", dpi=150)


if __name__ == "__main__":
	main()
