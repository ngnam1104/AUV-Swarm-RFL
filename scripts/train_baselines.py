import argparse
import csv
import os
import sys
from dataclasses import asdict
from datetime import datetime
from types import SimpleNamespace

import numpy as np
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.settings import ACOUSTIC_CFG, FL_CFG, HW_CFG, RL_CFG
from env.auv_env import AUVSwarmEnv
from fl_core.simulator import FLSimulator


def build_config(m_value: int, max_fl_rounds: int, eval_interval: int = 10) -> SimpleNamespace:
    cfg = SimpleNamespace(
        **asdict(ACOUSTIC_CFG),
        **asdict(HW_CFG),
        **asdict(FL_CFG),
        **asdict(RL_CFG),
    )
    cfg.M = int(m_value)
    cfg.max_fl_rounds = int(max_fl_rounds)
    cfg.eval_interval = int(eval_interval)
    return cfg


def save_cost_series(costs: list[float], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "accumulated_cost"])
        for ep_idx, value in enumerate(costs):
            writer.writerow([ep_idx, float(value)])


class EpisodeCostCallback(BaseCallback):
    def __init__(self, label: str, verbose: int = 0, print_every_episode: int = 1):
        super().__init__(verbose)
        self.label = label
        self.print_every_episode = max(1, int(print_every_episode))
        self.episode_costs: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for idx, done in enumerate(dones):
            if bool(done) and idx < len(infos) and isinstance(infos[idx], dict):
                acc_cost = float(infos[idx].get("accumulated_cost", np.nan))
                self.episode_costs.append(acc_cost)
                if len(self.episode_costs) % self.print_every_episode == 0:
                    print(
                        f"[{self.label}] Episode {len(self.episode_costs)} | accumulated_cost={acc_cost:.4f}",
                        flush=True,
                    )
        return True


class StepInfoCallback(BaseCallback):
    def __init__(
        self,
        label: str,
        print_every_steps: int = 1,
        log_file_path: str | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.label = label
        self.print_every_steps = max(1, int(print_every_steps))
        self.log_file_path = log_file_path
        self._log_file = None

    def _on_training_start(self) -> None:
        if self.log_file_path:
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            self._log_file = open(self.log_file_path, "a", encoding="utf-8")
            self._log_file.write(
                f"=== {self.label} baseline training started at {datetime.utcnow().isoformat()} ===\n"
            )
            self._log_file.flush()

    def _on_training_end(self) -> None:
        if self._log_file is not None:
            self._log_file.write(
                f"=== {self.label} baseline training ended at {datetime.utcnow().isoformat()} ===\n"
            )
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_every_steps != 0:
            return True

        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        if not infos:
            return True

        info = infos[0] if isinstance(infos[0], dict) else {}
        reward = float(rewards[0]) if len(rewards) > 0 else float("nan")
        timing = info.get("timing", {}) if isinstance(info, dict) else {}

        step_line = (
            f"[FL-RL Step {self.num_timesteps}] "
            f"ep_step={info.get('step_idx', 'n/a')}/{info.get('max_steps', 'n/a')} | "
            f"reward={reward:.4f} | cost={float(info.get('cost', np.nan)):.4f} | "
            f"acc_cost={float(info.get('accumulated_cost', np.nan)):.4f} | "
            f"active={info.get('active_nodes', 'n/a')} | acc={float(info.get('accuracy', np.nan)):.4f} | "
            f"conv={info.get('is_converged', False)} | "
            f"T={float(info.get('T_total', np.nan)):.4f}s | E={float(info.get('E_total', np.nan)):.4f}J"
        )
        print(step_line, flush=True)
        if self._log_file is not None:
            self._log_file.write(step_line + "\n")

        if timing:
            timing_line = (
                f"[FL TIMING] total={timing.get('step_total_sec', 0):.2f}s | "
                f"local_train={timing.get('local_train_and_grad_sec', 0):.2f}s | "
                f"eval={timing.get('evaluate_sec', 0):.2f}s "
                f"(every {timing.get('eval_interval', 'n/a')} steps, ran={timing.get('should_evaluate', False)}) | "
                f"agg={timing.get('aggregate_sec', 0):.2f}s | "
                f"slowest={timing.get('slowest_stage', 'n/a')}"
            )
            print(timing_line, flush=True)
            if self._log_file is not None:
                self._log_file.write(timing_line + "\n")
                self._log_file.flush()
        return True


def make_env(config: SimpleNamespace):
    fl_sim = FLSimulator(config)
    env = AUVSwarmEnv(fl_sim=fl_sim, config=config)
    return Monitor(env)


def train_ppo(
    config: SimpleNamespace,
    episodes: int,
    model_out: str | None = None,
    print_every_steps: int = 1,
    step_log_file: str | None = None,
) -> list[float]:
    vec_env = DummyVecEnv([lambda: make_env(config)])
    callback = [
        EpisodeCostCallback(label="PPO", print_every_episode=1),
        StepInfoCallback(
            label="PPO",
            print_every_steps=print_every_steps,
            log_file_path=step_log_file,
        ),
    ]

    n_steps = max(2, min(int(config.ppo_n_steps), int(config.max_fl_rounds)))
    batch_size = max(2, min(int(config.ppo_batch_size), n_steps))

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=float(config.ppo_lr),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=int(config.ppo_n_epochs),
        verbose=1,
    )

    total_timesteps = int(episodes * config.max_fl_rounds)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    if model_out:
        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        model.save(model_out)

    vec_env.close()
    return callback[0].episode_costs


def train_ddpg(
    config: SimpleNamespace,
    episodes: int,
    model_out: str | None = None,
    print_every_steps: int = 1,
    step_log_file: str | None = None,
) -> list[float]:
    vec_env = DummyVecEnv([lambda: make_env(config)])
    callback = [
        EpisodeCostCallback(label="DDPG", print_every_episode=1),
        StepInfoCallback(
            label="DDPG",
            print_every_steps=print_every_steps,
            log_file_path=step_log_file,
        ),
    ]

    model = DDPG(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-3,
        buffer_size=100000,
        learning_starts=int(config.max_fl_rounds),
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        verbose=1,
    )

    total_timesteps = int(episodes * config.max_fl_rounds)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    if model_out:
        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        model.save(model_out)

    vec_env.close()
    return callback[0].episode_costs


def encode_action(env: AUVSwarmEnv, p_value: float, f_value: float, beta: float) -> np.ndarray:
    p_norm = 2.0 * (p_value - env.p_min) / (env.p_max - env.p_min) - 1.0
    f_norm = 2.0 * (f_value - env.f_min) / (env.f_max - env.f_min) - 1.0
    beta_norm = 2.0 * beta - 1.0

    action = np.zeros(env.dim, dtype=np.float32)
    action[: env.M] = p_norm
    action[env.M : 2 * env.M] = f_norm
    action[2 * env.M] = p_norm
    action[2 * env.M + 1] = f_norm
    action[2 * env.M + 2] = beta_norm
    return np.clip(action, -1.0, 1.0)


def greedy_action(env: AUVSwarmEnv) -> np.ndarray:
    p_candidates = [env.p_min, 0.5 * (env.p_min + env.p_max)]
    f_candidates = [0.5 * (env.f_min + env.f_max), env.f_max]
    beta_candidates = np.arange(0.1, 1.01, 0.1)

    best_cost = float("inf")
    best_tuple = (env.p_min, env.f_max, 0.5)

    for p_val in p_candidates:
        for f_val in f_candidates:
            for beta in beta_candidates:
                k = max(1, int(round(float(beta) * env.M)))
                lambda_m = np.zeros(env.M, dtype=float)
                lambda_m[:k] = 1.0

                p_m = np.full(env.M, p_val, dtype=float)
                f_m = np.full(env.M, f_val, dtype=float)

                e_total, _, t_total, _ = env.energy_model.compute_total_energy_from_latency(
                    latency_model=env.latency_model,
                    lambda_m=lambda_m,
                    f_m=f_m,
                    p_m=p_m,
                    f_L=float(f_val),
                    p_L=float(p_val),
                )
                immediate_cost = float(t_total + e_total)

                if immediate_cost < best_cost:
                    best_cost = immediate_cost
                    best_tuple = (float(p_val), float(f_val), float(beta))

    return encode_action(env, p_value=best_tuple[0], f_value=best_tuple[1], beta=best_tuple[2])


def run_policy_free_baseline(config: SimpleNamespace, episodes: int, mode: str) -> list[float]:
    env = make_env(config)
    base_env = env.env
    costs: list[float] = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        last_info = {}

        while not done:
            if mode == "random":
                action = env.action_space.sample()
            elif mode == "greedy":
                action = greedy_action(base_env)
            else:
                raise ValueError(f"Unknown policy-free mode: {mode}")

            obs, reward, terminated, truncated, info = env.step(action)
            _ = (obs, reward)
            done = bool(terminated or truncated)
            last_info = info

        costs.append(float(last_info.get("accumulated_cost", np.nan)))
        if (ep + 1) % 10 == 0:
            print(f"[{mode.upper()}] Episode {ep + 1} | accumulated_cost={costs[-1]:.4f}")

    env.close()
    return costs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate RL baselines for Figure 7.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--m", type=int, default=9, help="Number of AUVs")
    parser.add_argument("--max-fl-rounds", type=int, default=1000, help="FL rounds per episode")
    parser.add_argument("--eval-interval", type=int, default=10, help="Accuracy evaluation interval")
    parser.add_argument("--print-every-steps", type=int, default=1, help="Print FL-RL step log every N steps")
    parser.add_argument(
        "--step-log-file",
        type=str,
        default="results/logs/baselines_step_log.txt",
        help="File path to append detailed FL-RL step/timing logs",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["ppo", "ddpg", "greedy", "random"],
        choices=["ppo", "ddpg", "greedy", "random"],
        help="Algorithms to run",
    )
    parser.add_argument("--out-dir", type=str, default="results/fig_7", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    config = build_config(
        m_value=args.m,
        max_fl_rounds=args.max_fl_rounds,
        eval_interval=args.eval_interval,
    )

    if "ppo" in args.algorithms:
        ppo_costs = train_ppo(
            config=config,
            episodes=args.episodes,
            model_out=os.path.join(args.out_dir, "ppo_baseline_model"),
            print_every_steps=int(args.print_every_steps),
            step_log_file=args.step_log_file,
        )
        save_cost_series(ppo_costs, os.path.join(args.out_dir, "ppo_accumulated_cost.csv"))

    if "ddpg" in args.algorithms:
        ddpg_costs = train_ddpg(
            config=config,
            episodes=args.episodes,
            model_out=os.path.join(args.out_dir, "ddpg_baseline_model"),
            print_every_steps=int(args.print_every_steps),
            step_log_file=args.step_log_file,
        )
        save_cost_series(ddpg_costs, os.path.join(args.out_dir, "ddpg_accumulated_cost.csv"))

    if "greedy" in args.algorithms:
        greedy_costs = run_policy_free_baseline(config=config, episodes=args.episodes, mode="greedy")
        save_cost_series(greedy_costs, os.path.join(args.out_dir, "greedy_accumulated_cost.csv"))

    if "random" in args.algorithms:
        random_costs = run_policy_free_baseline(config=config, episodes=args.episodes, mode="random")
        save_cost_series(random_costs, os.path.join(args.out_dir, "random_accumulated_cost.csv"))

    print("[DONE] Baseline training complete.")
    print(f"Outputs saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
