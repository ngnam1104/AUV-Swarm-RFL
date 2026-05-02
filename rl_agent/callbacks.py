import os
from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class AUVTensorboardCallback(BaseCallback):
    """Ghi các metric custom từ info dict lên TensorBoard."""

    def __init__(
        self,
        verbose: int = 0,
        print_every_steps: int = 1,
        log_file_path: str | None = None,
        append_log: bool = True,
    ):
        super().__init__(verbose)
        self.print_every_steps = max(1, int(print_every_steps))
        self.log_file_path = log_file_path
        self.append_log = append_log
        self._log_file_handle = None

    def _on_training_start(self) -> None:
        if not self.log_file_path:
            return
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        mode = "a" if self.append_log else "w"
        self._log_file_handle = open(self.log_file_path, mode, encoding="utf-8")
        self._write_log_line(
            f"=== Training started at {datetime.now().isoformat(timespec='seconds')} ==="
        )

    def _on_training_end(self) -> None:
        if self._log_file_handle is not None:
            self._write_log_line(
                f"=== Training ended at {datetime.now().isoformat(timespec='seconds')} ==="
            )
            self._log_file_handle.close()
            self._log_file_handle = None

    def _write_log_line(self, line: str) -> None:
        if self._log_file_handle is None:
            return
        self._log_file_handle.write(f"{line}\n")
        self._log_file_handle.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)

        if not infos:
            return True

        for idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            cost = float(info.get("cost", np.nan))
            accumulated_cost = float(info.get("accumulated_cost", np.nan))
            t_total = float(info.get("T_total", np.nan))
            e_total = float(info.get("E_total", np.nan))
            active_nodes = float(info.get("active_nodes", np.nan))
            total_nodes = int(info.get("total_nodes", -1))
            accuracy = float(info.get("accuracy", np.nan))
            step_idx = int(info.get("step_idx", -1))
            max_steps = int(info.get("max_steps", -1))

            if rewards is not None and len(rewards) > idx:
                reward = float(rewards[idx])
            else:
                reward = np.nan

            self.logger.record("custom/cost", cost)
            self.logger.record("custom/accumulated_cost", accumulated_cost)
            self.logger.record("custom/T_total", t_total)
            self.logger.record("custom/E_total", e_total)
            self.logger.record("custom/active_nodes", active_nodes)
            self.logger.record("custom/reward", reward)
            self.logger.record("custom/accuracy", accuracy)

            # In thời gian chi tiết FL/RL ra terminal để theo dõi realtime.
            timing = info.get("timing", {})
            if self.num_timesteps % self.print_every_steps == 0:
                step_line = (
                    f"[FL-RL Step {self.num_timesteps}] "
                    f"ep_step={step_idx}/{max_steps} | reward={reward:.4f} | "
                    f"cost={cost:.4f} | acc_cost={accumulated_cost:.4f} | "
                    f"active={active_nodes:.0f}/{total_nodes} | acc={accuracy:.4f} | "
                    f"conv={info.get('is_converged', False)} | "
                    f"T={t_total:.4f}s | E={e_total:.4f}J"
                )
                print(step_line, flush=True)
                self._write_log_line(step_line)

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
                    self._write_log_line(timing_line)

            if dones is not None and len(dones) > idx and bool(dones[idx]):
                episode_line = (
                    f"[EPISODE END] steps={step_idx} | accumulated_cost={accumulated_cost:.4f} | "
                    f"final_accuracy={accuracy:.4f}"
                )
                print(episode_line, flush=True)
                self._write_log_line(episode_line)

        return True


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """Lưu model tốt nhất khi mean episode reward được cải thiện."""

    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self._episode_rewards = []

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if isinstance(info, dict) and "episode" in info and "r" in info["episode"]:
                self._episode_rewards.append(float(info["episode"]["r"]))

        if len(self._episode_rewards) > 0:
            mean_reward = float(np.mean(self._episode_rewards))
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)

        return True
