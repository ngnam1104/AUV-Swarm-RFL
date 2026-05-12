import argparse
import csv
import multiprocessing
import concurrent.futures
import os
import sys
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast

warnings.filterwarnings("ignore")

import numpy as np
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tqdm import tqdm
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


def save_metrics_series(metrics_list: list[dict], out_csv: str) -> None:
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if not metrics_list: return
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode"] + list(metrics_list[0].keys()))
        writer.writeheader()
        for ep_idx, metrics in enumerate(metrics_list):
            row = {"episode": ep_idx}
            row.update(metrics)
            writer.writerow(row)


class EpisodeMetricsCallback(BaseCallback):
    def __init__(self, label: str, target_episodes: int, verbose: int = 0, print_every_episode: int = 1, model_out_prefix: str | None = None, pos: int = 0):
        super().__init__(verbose)
        self.label = label
        self.target_episodes = target_episodes
        self.print_every_episode = max(1, int(print_every_episode))
        self.model_out_prefix = model_out_prefix
        self.episode_metrics: list[dict] = []
        self.pbar = None
        self.pos = pos

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.target_episodes, desc=f"[{self.label:^6}] RL", position=self.pos, leave=True, file=sys.stderr)

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for idx, done in enumerate(dones):
            if bool(done) and idx < len(infos) and isinstance(infos[idx], dict):
                info = infos[idx]
                step_idx = max(1, info.get("step_idx", 1))
                metrics = {
                    "accumulated_cost": float(info.get("accumulated_cost", np.nan)),
                    "avg_delay": float(info.get("accumulated_delay", 0.0)) / step_idx,
                    "avg_energy": float(info.get("accumulated_energy", 0.0)) / step_idx,
                    "avg_reward": float(info.get("accumulated_reward", 0.0)) / step_idx,
                    "avg_comm": float(info.get("accumulated_comm", 0.0)) / step_idx,
                }
                self.episode_metrics.append(metrics)
                current_ep = len(self.episode_metrics)

                if self.pbar is not None:
                    self.pbar.set_postfix(
                        cost=f"{metrics['accumulated_cost']:.2f}",
                        rew=f"{metrics['avg_reward']:.2f}"
                    )
                    self.pbar.update(1)

                if self.model_out_prefix:
                    # Atomic save: ghi ra file .tmp trước, rồi rename
                    # => nếu disk full hoặc bị kill giữa chừng, file cũ vẫn còn nguyên
                    tmp_path = f"{self.model_out_prefix}.tmp"
                    try:
                        self.model.save(tmp_path)
                        final_zip = f"{tmp_path}.zip"
                        dest_zip  = f"{self.model_out_prefix}.zip"
                        if os.path.exists(final_zip):
                            os.replace(final_zip, dest_zip)
                        elif os.path.exists(tmp_path):
                            os.replace(tmp_path, self.model_out_prefix)
                    except Exception as _save_err:
                        print(f"[WARN] Checkpoint save failed (disk full?): {_save_err}", flush=True)

                    # Lưu checkpoint có tên riêng mỗi 50 episode (atomic)
                    if current_ep % 50 == 0:
                        ck_prefix = f"{self.model_out_prefix}_ep{current_ep}"
                        ck_tmp    = f"{ck_prefix}.tmp"
                        try:
                            self.model.save(ck_tmp)
                            ck_final_zip = f"{ck_tmp}.zip"
                            ck_dest_zip  = f"{ck_prefix}.zip"
                            if os.path.exists(ck_final_zip):
                                os.replace(ck_final_zip, ck_dest_zip)
                            elif os.path.exists(ck_tmp):
                                os.replace(ck_tmp, ck_prefix)
                        except Exception as _ck_err:
                            print(f"[WARN] Periodic checkpoint ep{current_ep} save failed: {_ck_err}", flush=True)

        if len(self.episode_metrics) >= self.target_episodes:
            return False

        return True


class StepInfoCallback(BaseCallback):
    def __init__(
        self,
        label: str,
        print_every_steps: int = 1,
        log_file_path: str | None = None,
        verbose: int = 0,
        pos: int = 0,
    ):
        super().__init__(verbose)
        self.label = label
        self.print_every_steps = max(1, int(print_every_steps))
        self.log_file_path = log_file_path
        self._log_file = None
        self.pos = pos
        self.fl_pbar = None

    def _on_training_start(self) -> None:
        if self.log_file_path:
            out_dir = os.path.dirname(self.log_file_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            self._log_file = open(self.log_file_path, "a", encoding="utf-8")
            self._log_file.write(
                f"=== {self.label} baseline training started at {datetime.now(timezone.utc).isoformat()} ===\n"
            )
            self._log_file.flush()

    def _on_training_end(self) -> None:
        if self.fl_pbar is not None:
            self.fl_pbar.close()
        if self._log_file is not None:
            self._log_file.write(
                f"=== {self.label} baseline training ended at {datetime.now(timezone.utc).isoformat()} ===\n"
            )
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        
        if not infos:
            return True

        info = infos[0] if isinstance(infos[0], dict) else {}
        reward = float(rewards[0]) if len(rewards) > 0 else float("nan")
        timing = info.get("timing", {}) if isinstance(info, dict) else {}
        
        step_idx = info.get('step_idx', 0)
        max_steps = info.get('max_steps', 1000)

        # Quản lý FL progress bar (tầng dưới)
        if self.fl_pbar is None:
            self.fl_pbar = tqdm(
                total=max_steps, 
                desc=f" └─ {self.label:>6} FL", 
                position=self.pos + 3, # Đặt ở phía dưới các RL bars
                leave=False,
                file=sys.stderr
            )
            
        if step_idx <= 1:
            self.fl_pbar.reset(total=max_steps)
            self.fl_pbar.refresh()
        
        self.fl_pbar.n = step_idx
        self.fl_pbar.set_postfix(acc=f"{info.get('accuracy', 0):.4f}")
        self.fl_pbar.refresh()

        if self.num_timesteps % self.print_every_steps != 0:
            return True

        # --- Ghi đầy đủ FL-RL step log vào file (không in ra stdout/tqdm) ---
        step_line = (
            f"[{self.label}][FL-RL Step {self.num_timesteps}] "
            f"ep_step={step_idx}/{max_steps} | "
            f"reward={reward:.4f} | cost={float(info.get('cost', np.nan)):.4f} | "
            f"acc_cost={float(info.get('accumulated_cost', np.nan)):.4f} | "
            f"active={info.get('active_nodes', 'n/a')} | "
            f"acc={float(info.get('accuracy', np.nan)):.4f} | "
            f"conv={info.get('is_converged', False)} | "
            f"T_total={float(info.get('T_total', np.nan)):.4f}s | "
            f"E_total={float(info.get('E_total', np.nan)):.4f}J | "
            f"active_comm_nodes={info.get('active_nodes', 'n/a')}/{info.get('total_nodes', 'n/a')} | "
            f"accumulated_comm={int(info.get('accumulated_comm', 0))} | "
            f"latency={float(info.get('accumulated_delay', np.nan)):.4f}s | "
            f"energy={float(info.get('accumulated_energy', np.nan)):.4f}J"
        )
        if self._log_file is not None:
            self._log_file.write(step_line + "\n")
        
        # tqdm.write đẩy ra stdout mà không làm hỏng progress bar
        # Giúp pipeline.log (qua stdout) bắt được thông số này
        tqdm.write(step_line, file=sys.stdout)

        if timing:
            timing_line = (
                f"[{self.label}][FL TIMING] total={timing.get('step_total_sec', 0):.2f}s | "
                f"local_train={timing.get('local_train_and_grad_sec', 0):.2f}s | "
                f"eval={timing.get('evaluate_sec', 0):.2f}s "
                f"(every {timing.get('eval_interval', 'n/a')} steps, ran={timing.get('should_evaluate', False)}) | "
                f"agg={timing.get('aggregate_sec', 0):.2f}s | "
                f"slowest={timing.get('slowest_stage', 'n/a')}"
            )
            if self._log_file is not None:
                self._log_file.write(timing_line + "\n")
                self._log_file.flush()
            tqdm.write(timing_line, file=sys.stdout)
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
    pos: int = 0
) -> list[float]:
    vec_env = DummyVecEnv([lambda: make_env(config)])
    callback = [
        EpisodeMetricsCallback(label="PPO", target_episodes=episodes, print_every_episode=1, model_out_prefix=model_out, pos=pos),
        StepInfoCallback(
            label="PPO",
            print_every_steps=print_every_steps,
            log_file_path=step_log_file,
            pos=pos
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
        verbose=0,
    )

    total_timesteps = int(episodes * config.max_fl_rounds)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    if model_out:
        out_dir = os.path.dirname(model_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        tmp_final = f"{model_out}.final_tmp"
        model.save(tmp_final)
        tmp_zip = f"{tmp_final}.zip"
        dest_zip = f"{model_out}.zip"
        if os.path.exists(tmp_zip):
            os.replace(tmp_zip, dest_zip)
        elif os.path.exists(tmp_final):
            os.replace(tmp_final, model_out)

    vec_env.close()
    return callback[0].episode_metrics


def train_ddpg(
    config: SimpleNamespace,
    episodes: int,
    model_out: str | None = None,
    print_every_steps: int = 1,
    step_log_file: str | None = None,
) -> list[float]:
    vec_env = DummyVecEnv([lambda: make_env(config)])
    callback = [
        EpisodeMetricsCallback(label="DDPG", target_episodes=episodes, print_every_episode=1),
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
        verbose=0,
    )

    total_timesteps = int(episodes * config.max_fl_rounds)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    if model_out:
        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        model.save(model_out)

    vec_env.close()
    return callback[0].episode_metrics


def train_sac(
    config: SimpleNamespace,
    episodes: int,
    model_out: str | None = None,
    print_every_steps: int = 1,
    step_log_file: str | None = None,
) -> list[float]:
    """SAC (Soft Actor-Critic) — maximum-entropy off-policy, best exploration."""
    vec_env = DummyVecEnv([lambda: make_env(config)])
    callback = [
        EpisodeMetricsCallback(label="SAC", target_episodes=episodes, print_every_episode=1),
        StepInfoCallback(label="SAC", print_every_steps=print_every_steps, log_file_path=step_log_file),
    ]
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=int(config.max_fl_rounds),
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",          # automatic entropy tuning
        target_entropy="auto",
        verbose=0,
    )
    model.learn(total_timesteps=int(episodes * config.max_fl_rounds), callback=callback)
    if model_out:
        os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
        model.save(model_out)
    vec_env.close()
    return callback[0].episode_metrics


def train_td3(
    config: SimpleNamespace,
    episodes: int,
    model_out: str | None = None,
    print_every_steps: int = 1,
    step_log_file: str | None = None,
) -> list[float]:
    """TD3 (Twin Delayed DDPG) — fixes DDPG overestimation bias."""
    vec_env = DummyVecEnv([lambda: make_env(config)])
    callback = [
        EpisodeMetricsCallback(label="TD3", target_episodes=episodes, print_every_episode=1),
        StepInfoCallback(label="TD3", print_every_steps=print_every_steps, log_file_path=step_log_file),
    ]
    model = TD3(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=int(config.max_fl_rounds),
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "episode"),
        gradient_steps=-1,        # update as many steps as episodes collected
        policy_delay=2,           # TD3 key: delayed policy update
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        verbose=0,
    )
    model.learn(total_timesteps=int(episodes * config.max_fl_rounds), callback=callback)
    if model_out:
        os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
        model.save(model_out)
    vec_env.close()
    return callback[0].episode_metrics


def train_a2c(
    config: SimpleNamespace,
    episodes: int,
    model_out: str | None = None,
    print_every_steps: int = 1,
    step_log_file: str | None = None,
) -> list[float]:
    """A2C (Advantage Actor-Critic) — lightweight on-policy synchronous AC."""
    vec_env = DummyVecEnv([lambda: make_env(config)])
    callback = [
        EpisodeMetricsCallback(label="A2C", target_episodes=episodes, print_every_episode=1),
        StepInfoCallback(label="A2C", print_every_steps=print_every_steps, log_file_path=step_log_file),
    ]
    n_steps = max(2, min(int(getattr(config, "ppo_n_steps", 512)), int(config.max_fl_rounds)))
    model = A2C(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=7e-4,
        n_steps=n_steps,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
    )
    model.learn(total_timesteps=int(episodes * config.max_fl_rounds), callback=callback)
    if model_out:
        os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
        model.save(model_out)
    vec_env.close()
    return callback[0].episode_metrics


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


def run_policy_free_baseline(
    config: SimpleNamespace,
    episodes: int,
    mode: str,
    log_file_path: str | None = None,
    pos: int = 0
) -> list[float]:
    env = make_env(config)
    base_env = cast(AUVSwarmEnv, env.env)
    metrics_list: list[dict] = []
    log_fh = None
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        log_fh = open(log_file_path, "w", encoding="utf-8")
        log_fh.write(
            f"=== {mode.upper()} baseline started at {datetime.now(timezone.utc).isoformat()} ===\n"
        )
        log_fh.flush()

    pbar = tqdm(total=episodes, desc=f"[{mode.upper():^6}] RL", position=pos, leave=True, file=sys.stderr)
    fl_pbar = None
    global_step = 0  # timestep counter across all episodes

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        last_info = {}
        ep_step = 0

        if fl_pbar is None:
            fl_pbar = tqdm(
                total=config.max_fl_rounds,
                desc=f" └─ {mode.upper():>6} FL",
                position=pos + 3,
                leave=False,
                file=sys.stderr
            )

        fl_pbar.reset(total=config.max_fl_rounds)
        fl_pbar.refresh()

        if log_fh is not None:
            log_fh.write(
                f"[{mode.upper()}] === Episode {ep + 1}/{episodes} START ===\n"
            )
            log_fh.flush()

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
            global_step += 1
            ep_step = int(info.get("step_idx", ep_step + 1))
            max_steps = int(info.get("max_steps", config.max_fl_rounds))

            # --- Cập nhật FL progress bar (terminal UI, không ghi log) ---
            fl_pbar.n = ep_step
            fl_pbar.set_postfix(acc=f"{info.get('accuracy', 0):.4f}")
            fl_pbar.refresh()

            # --- Ghi đầy đủ FL-RL step log vào file ---
            if log_fh is not None:
                step_line = (
                    f"[{mode.upper()}][FL-RL Step {global_step}] "
                    f"ep={ep + 1}/{episodes} ep_step={ep_step}/{max_steps} | "
                    f"reward={float(reward):.4f} | "
                    f"cost={float(info.get('cost', np.nan)):.4f} | "
                    f"acc_cost={float(info.get('accumulated_cost', np.nan)):.4f} | "
                    f"active={info.get('active_nodes', 'n/a')} | "
                    f"acc={float(info.get('accuracy', np.nan)):.4f} | "
                    f"conv={info.get('is_converged', False)} | "
                    f"T_total={float(info.get('T_total', np.nan)):.4f}s | "
                    f"E_total={float(info.get('E_total', np.nan)):.4f}J | "
                    f"active_comm_nodes={info.get('active_nodes', 'n/a')}/{info.get('total_nodes', 'n/a')} | "
                    f"accumulated_comm={int(info.get('accumulated_comm', 0))} | "
                    f"latency={float(info.get('accumulated_delay', np.nan)):.4f}s | "
                    f"energy={float(info.get('accumulated_energy', np.nan)):.4f}J"
                )
                log_fh.write(step_line + "\n")
                # tqdm.write đẩy ra stdout để pipeline.log bắt được
                tqdm.write(step_line, file=sys.stdout)

                # Ghi timing nếu có
                timing = info.get("timing", {})
                if timing:
                    timing_line = (
                        f"[{mode.upper()}][FL TIMING] "
                        f"total={timing.get('step_total_sec', 0):.2f}s | "
                        f"local_train={timing.get('local_train_and_grad_sec', 0):.2f}s | "
                        f"eval={timing.get('evaluate_sec', 0):.2f}s "
                        f"(every {timing.get('eval_interval', 'n/a')} steps, ran={timing.get('should_evaluate', False)}) | "
                        f"agg={timing.get('aggregate_sec', 0):.2f}s | "
                        f"slowest={timing.get('slowest_stage', 'n/a')}"
                    )
                    log_fh.write(timing_line + "\n")
                    tqdm.write(timing_line, file=sys.stdout)
                log_fh.flush()

        # --- Tổng kết cuối episode ---
        step_idx = max(1, last_info.get("step_idx", 1))
        metrics = {
            "accumulated_cost": float(last_info.get("accumulated_cost", np.nan)),
            "avg_delay": float(last_info.get("accumulated_delay", 0.0)) / step_idx,
            "avg_energy": float(last_info.get("accumulated_energy", 0.0)) / step_idx,
            "avg_reward": float(last_info.get("accumulated_reward", 0.0)) / step_idx,
            "avg_comm": float(last_info.get("accumulated_comm", 0.0)) / step_idx,
        }
        metrics_list.append(metrics)

        pbar.set_postfix(
            cost=f"{metrics['accumulated_cost']:.2f}",
            rew=f"{metrics['avg_reward']:.2f}",
        )
        pbar.update(1)

        if log_fh is not None:
            ep_summary = (
                f"[{mode.upper()}] === Episode {ep + 1}/{episodes} DONE | "
                f"accumulated_cost={metrics['accumulated_cost']:.4f} | "
                f"avg_delay={metrics['avg_delay']:.4f}s | "
                f"avg_energy={metrics['avg_energy']:.4f}J | "
                f"avg_comm={metrics['avg_comm']:.4f}s | "
                f"avg_reward={metrics['avg_reward']:.4f} ===\n"
            )
            log_fh.write(ep_summary)
            log_fh.flush()

    pbar.close()
    if fl_pbar is not None:
        fl_pbar.close()

    if log_fh is not None:
        log_fh.write(
            f"=== {mode.upper()} baseline finished at {datetime.now(timezone.utc).isoformat()} ===\n"
        )
        log_fh.close()

    env.close()
    return metrics_list


def _run_single_algo(algo: str, config: SimpleNamespace, args, log_dir: str, pos: int = 0):
    # print(f"[START] Training {algo.upper()} ...", flush=True)
    if algo == "ppo":
        metrics_list = train_ppo(
            config=config, episodes=args.episodes,
            model_out=os.path.join(args.out_dir, "ppo_baseline_model"),
            print_every_steps=int(args.print_every_steps),
            step_log_file=os.path.join(log_dir, "ppo_steps.log"),
            pos=pos
        )
        save_metrics_series(metrics_list, os.path.join(args.out_dir, "ppo_metrics.csv"))
    elif algo == "sac":
        metrics_list = train_sac(
            config=config, episodes=args.episodes,
            model_out=os.path.join(args.out_dir, "sac_baseline_model"),
            print_every_steps=int(args.print_every_steps),
            step_log_file=os.path.join(log_dir, "sac_steps.log"),
        )
        save_metrics_series(metrics_list, os.path.join(args.out_dir, "sac_metrics.csv"))
    elif algo == "td3":
        metrics_list = train_td3(
            config=config, episodes=args.episodes,
            model_out=os.path.join(args.out_dir, "td3_baseline_model"),
            print_every_steps=int(args.print_every_steps),
            step_log_file=os.path.join(log_dir, "td3_steps.log"),
        )
        save_metrics_series(metrics_list, os.path.join(args.out_dir, "td3_metrics.csv"))
    elif algo == "ddpg":
        metrics_list = train_ddpg(
            config=config, episodes=args.episodes,
            model_out=os.path.join(args.out_dir, "ddpg_baseline_model"),
            print_every_steps=int(args.print_every_steps),
            step_log_file=os.path.join(log_dir, "ddpg_steps.log"),
        )
        save_metrics_series(metrics_list, os.path.join(args.out_dir, "ddpg_metrics.csv"))
    elif algo == "a2c":
        metrics_list = train_a2c(
            config=config, episodes=args.episodes,
            model_out=os.path.join(args.out_dir, "a2c_baseline_model"),
            print_every_steps=int(args.print_every_steps),
            step_log_file=os.path.join(log_dir, "a2c_steps.log"),
        )
        save_metrics_series(metrics_list, os.path.join(args.out_dir, "a2c_metrics.csv"))
    elif algo == "greedy":
        metrics_list = run_policy_free_baseline(
            config=config, episodes=args.episodes, mode="greedy",
            log_file_path=os.path.join(log_dir, "greedy_episodes.log"),
            pos=pos
        )
        save_metrics_series(metrics_list, os.path.join(args.out_dir, "greedy_metrics.csv"))
    elif algo == "random":
        metrics_list = run_policy_free_baseline(
            config=config, episodes=args.episodes, mode="random",
            log_file_path=os.path.join(log_dir, "random_episodes.log"),
            pos=pos
        )
        save_metrics_series(metrics_list, os.path.join(args.out_dir, "random_metrics.csv"))
    
    # print(f"[DONE] Training {algo.upper()}.", flush=True)
    return algo


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
        default=["ppo", "random", "greedy"],
        choices=["ppo", "sac", "td3", "ddpg", "a2c", "greedy", "random"],
        help="Algorithms to run (default: ppo, random, greedy)",
    )
    parser.add_argument("--out-dir", type=str, default="results/fig_7", help="Output directory")
    parser.add_argument("--enable-early-stopping", action="store_true", help="Enable early stopping in FL simulator")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory to save per-algorithm step logs (default: <out-dir>/logs)")
    parser.add_argument("--parallel", action="store_true", help="Run algorithms in parallel")
    parser.add_argument("--max-parallel-workers", type=int, default=3, help="Max parallel workers if --parallel is used (to prevent OOM on 16GB RAM)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    config = build_config(
        m_value=args.m,
        max_fl_rounds=args.max_fl_rounds,
        eval_interval=args.eval_interval,
    )
    config.enable_early_stopping = bool(getattr(args, "enable_early_stopping", False))

    log_dir = args.log_dir or os.path.join(args.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if args.parallel and len(args.algorithms) > 1:
        max_workers = min(len(args.algorithms), args.max_parallel_workers)
        print(f"[INFO] Running {len(args.algorithms)} algorithms in parallel using ProcessPoolExecutor with {max_workers} workers...")
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(_run_single_algo, algo, config, args, log_dir, idx): algo
                for idx, algo in enumerate(args.algorithms)
            }
            has_error = False
            for fut in concurrent.futures.as_completed(futures):
                algo = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"[ERROR] Algorithm {algo.upper()} failed: {e}", flush=True)
                    has_error = True
            
            if has_error:
                sys.exit(1)
    else:
        for idx, algo in enumerate(args.algorithms):
            _run_single_algo(algo, config, args, log_dir, pos=idx)

    print("[DONE] Baseline training complete.")
    print(f"Outputs saved under: {args.out_dir}")
    print(f"Logs saved under   : {log_dir}")


if __name__ == "__main__":
    main()
