from stable_baselines3 import PPO

class AUVPPOTrainer:
    """Bọc logic khởi tạo PPO cho môi trường AUV-Swarm-RFL."""

    def __init__(self, env, config):
        self.env = env
        self.cfg = config
        self.ppo_lr = float(getattr(config, "ppo_lr"))
        self.ppo_n_steps = int(getattr(config, "ppo_n_steps"))
        self.ppo_batch_size = int(getattr(config, "ppo_batch_size"))
        self.ppo_n_epochs = int(getattr(config, "ppo_n_epochs"))
        self.ppo_gamma = float(getattr(config, "ppo_gamma", 0.99))

    def build_model(self):
        return PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.ppo_lr,
            n_steps=self.ppo_n_steps,
            batch_size=self.ppo_batch_size,
            n_epochs=self.ppo_n_epochs,
            gamma=self.ppo_gamma,
            tensorboard_log="./results/tensorboard/",
            verbose=1,
        )