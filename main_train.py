import os
from dataclasses import asdict
from types import SimpleNamespace

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config.settings import ACOUSTIC_CFG, FL_CFG, HW_CFG, RL_CFG
from env.auv_env import AUVSwarmEnv
from fl_core.simulator import FLSimulator
from rl_agent.callbacks import AUVTensorboardCallback, SaveOnBestTrainingRewardCallback
from rl_agent.ppo_trainer import AUVPPOTrainer


def build_full_config() -> SimpleNamespace:
	return SimpleNamespace(
		**asdict(ACOUSTIC_CFG),
		**asdict(HW_CFG),
		**asdict(FL_CFG),
		**asdict(RL_CFG),
	)


def main():
	config = build_full_config()

	os.makedirs("./logs", exist_ok=True)
	os.makedirs("./models", exist_ok=True)

	fl_sim = FLSimulator(config)

	def make_env():
		env = AUVSwarmEnv(fl_sim=fl_sim, config=config)
		env = Monitor(env)
		return env

	vec_env = DummyVecEnv([make_env])

	trainer = AUVPPOTrainer(env=vec_env, config=config)
	model = trainer.build_model()

	callbacks = [
		AUVTensorboardCallback(
			print_every_steps=1,
			log_file_path="./results/logs/fl_rl_train_log.txt",
			append_log=True,
		),
		SaveOnBestTrainingRewardCallback(save_path="./models/ppo_auv_best"),
	]

	total_timesteps = int(config.max_fl_rounds * 1000)
	model.learn(total_timesteps=total_timesteps, callback=callbacks)
	model.save("./models/ppo_auv_final")


if __name__ == "__main__":
	main()
