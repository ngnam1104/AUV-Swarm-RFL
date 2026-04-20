import argparse
import csv
import os
import sys
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
from stable_baselines3 import PPO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.settings import ACOUSTIC_CFG, FL_CFG, HW_CFG, RL_CFG
from env.communication import CommunicationModel
from env.energy import EnergyModel
from env.latency import LatencyModel
from fl_core.control import LazyNodeController
from fl_core.simulator import FLSimulator


class LAGController(LazyNodeController):
    """LAG-style node selection using absolute gradient-difference threshold."""

    def __init__(self, *args, lag_threshold: float = 1e4, **kwargs):
        super().__init__(*args, **kwargs)
        self.lag_threshold = float(lag_threshold)

    def select_active_nodes(self, beta: float, local_grad_sq_norms: dict, global_diff_sq: float, rng: np.random.Generator) -> list[int]:
        _ = beta
        grad_ref = float(np.sqrt(max(global_diff_sq, 0.0)))
        active_indices = []

        for m in range(self.M):
            grad_norm = float(np.sqrt(max(float(local_grad_sq_norms[m]), 0.0)))
            if abs(grad_norm - grad_ref) > self.lag_threshold:
                active_indices.append(m)

        if len(active_indices) == 0:
            active_indices.append(int(rng.integers(low=0, high=self.M)))

        for m in range(self.M):
            if self.lazy_consecutive[m] >= self.force_active_rounds:
                active_indices.append(m)

        return list(set(active_indices))


class AllActiveController(LazyNodeController):
    """Traditional async FL baseline: all nodes active every round."""

    def select_active_nodes(self, beta: float, local_grad_sq_norms: dict, global_diff_sq: float, rng: np.random.Generator) -> list[int]:
        _ = (beta, local_grad_sq_norms, global_diff_sq, rng)
        return list(range(self.M))


class SchemeEvaluator:
    def __init__(
        self,
        config: SimpleNamespace,
        rounds: int = 1000,
        model_path: str = "./models/ppo_auv_final",
        lag_threshold: float = 1e4,
        beta_heuristic: str = "linear",
        force_active_rounds: int | None = None,
        seed: int = 42,
    ):
        self.cfg = config
        self.rounds = int(rounds)
        self.model_path = model_path
        self.lag_threshold = float(lag_threshold)
        self.beta_heuristic = beta_heuristic
        self.force_active_rounds = force_active_rounds

        self.rng = np.random.default_rng(seed=seed)
        self.p_min = 0.01
        self.p_max = float(getattr(config, "p_max", 0.2))
        self.f_min = float(getattr(config, "f_min", 0.2e9))
        self.f_max = float(getattr(config, "f_max", 0.4e9))

        self.p_mid = 0.5 * (self.p_min + self.p_max)
        self.f_mid = 0.5 * (self.f_min + self.f_max)

    @staticmethod
    def build_config(m_value: int, eval_interval: int = 1) -> SimpleNamespace:
        cfg = SimpleNamespace(
            **asdict(ACOUSTIC_CFG),
            **asdict(HW_CFG),
            **asdict(FL_CFG),
            **asdict(RL_CFG),
        )
        cfg.M = int(m_value)
        cfg.max_fl_rounds = 1000
        cfg.eval_interval = int(eval_interval)
        return cfg

    def _load_model(self):
        if self.model_path and os.path.exists(self.model_path):
            return PPO.load(self.model_path)
        alt_zip = f"{self.model_path}.zip"
        if self.model_path and os.path.exists(alt_zip):
            return PPO.load(alt_zip)
        return None

    def _fixed_physics(self, mode: str) -> tuple[np.ndarray, np.ndarray, float, float]:
        m = int(self.cfg.M)
        if mode == "minmax":
            p_m = np.full(m, self.p_min, dtype=float)
            f_m = np.full(m, self.f_max, dtype=float)
            p_l = self.p_min
            f_l = self.f_max
            return p_m, f_m, p_l, f_l

        p_m = np.full(m, self.p_mid, dtype=float)
        f_m = np.full(m, self.f_mid, dtype=float)
        p_l = self.p_mid
        f_l = self.f_mid
        return p_m, f_m, p_l, f_l

    def _unscale_action(self, action_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        action_norm = np.asarray(action_norm, dtype=np.float64).reshape(-1)
        m = int(self.cfg.M)

        p_m = (action_norm[0:m] + 1.0) / 2.0 * (self.p_max - self.p_min) + self.p_min
        f_m = (action_norm[m : 2 * m] + 1.0) / 2.0 * (self.f_max - self.f_min) + self.f_min
        p_l = (action_norm[2 * m] + 1.0) / 2.0 * (self.p_max - self.p_min) + self.p_min
        f_l = (action_norm[2 * m + 1] + 1.0) / 2.0 * (self.f_max - self.f_min) + self.f_min
        beta = float((action_norm[2 * m + 2] + 1.0) / 2.0)

        beta = float(np.clip(beta, 0.0, 1.0))
        return p_m, f_m, float(p_l), float(f_l), beta

    def _heuristic_beta(self, rnd: int) -> float:
        if self.beta_heuristic == "constant":
            return 0.5

        if self.rounds <= 1:
            return 0.5

        ratio = float((rnd - 1) / (self.rounds - 1))
        return float(0.1 + 0.8 * ratio)

    def _run_rounds(self, mode: str) -> dict:
        fl_sim = FLSimulator(self.cfg)
        comm_model = CommunicationModel(self.cfg)
        latency_model = LatencyModel(self.cfg, comm_model)
        energy_model = EnergyModel(self.cfg)

        model = None
        if mode in {"scheme1", "scheme2"}:
            model = self._load_model()
            if model is None and mode == "scheme1":
                raise FileNotFoundError(
                    "Scheme 1 requires trained PPO model. Expected path: "
                    f"{self.model_path} or {self.model_path}.zip"
                )

        force_active = self.force_active_rounds if self.force_active_rounds is not None else fl_sim.force_active_rounds

        if mode == "scheme4":
            fl_sim.controller = LAGController(
                M=fl_sim.M,
                N=fl_sim.N_total,
                N_m_dict=fl_sim.N_m_dict,
                lr=fl_sim.lr,
                force_active_rounds=force_active,
                lag_threshold=self.lag_threshold,
            )
        elif mode == "scheme5":
            fl_sim.controller = AllActiveController(
                M=fl_sim.M,
                N=fl_sim.N_total,
                N_m_dict=fl_sim.N_m_dict,
                lr=fl_sim.lr,
                force_active_rounds=force_active,
            )
        else:
            # For scheme 1, 2, 3 we use default LazyNodeController
            # But we must ensure force_active_rounds is set if provided
            if self.force_active_rounds is not None:
                fl_sim.controller.force_active_rounds = self.force_active_rounds

        dim = 2 * int(self.cfg.M) + 3
        obs = self.rng.uniform(low=-1.0, high=1.0, size=(dim,)).astype(np.float32)

        communication_times = 0.0
        total_time_consumption = 0.0
        total_energy_consumption = 0.0
        accumulated_cost = 0.0
        final_accuracy = 0.0

        for rnd in range(1, self.rounds + 1):
            if mode == "scheme1":
                action, _ = model.predict(obs, deterministic=True)
                p_m, f_m, p_l, f_l, beta = self._unscale_action(action)
                obs = np.asarray(action, dtype=np.float32).reshape(-1)
            elif mode == "scheme2":
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                    _, _, _, _, beta = self._unscale_action(action)
                    obs = np.asarray(action, dtype=np.float32).reshape(-1)
                else:
                    beta = self._heuristic_beta(rnd)
                p_m, f_m, p_l, f_l = self._fixed_physics(mode="minmax")
            elif mode == "scheme3":
                beta = 0.5
                p_m, f_m, p_l, f_l = self._fixed_physics(mode="mid")
            elif mode == "scheme4":
                beta = 0.5
                p_m, f_m, p_l, f_l = self._fixed_physics(mode="mid")
            elif mode == "scheme5":
                beta = 1.0
                p_m, f_m, p_l, f_l = self._fixed_physics(mode="mid")
            else:
                raise ValueError(f"Unknown mode: {mode}")

            accuracy, active_indices, _, is_converged = fl_sim.sync_run_step(beta=beta, rnd=rnd)

            lambda_m = np.zeros(int(self.cfg.M), dtype=float)
            lambda_m[active_indices] = 1.0

            e_total, _, t_total, _ = energy_model.compute_total_energy_from_latency(
                latency_model=latency_model,
                lambda_m=lambda_m,
                f_m=p_m * 0 + f_m,
                p_m=p_m,
                f_L=f_l,
                p_L=p_l,
            )

            communication_times += float(np.sum(lambda_m))
            total_time_consumption += float(t_total)
            total_energy_consumption += float(e_total)
            accumulated_cost += float(t_total + e_total)
            final_accuracy = float(accuracy)

            if rnd % 50 == 0 or rnd == self.rounds or is_converged:
                print(
                    f"    [{mode}] Round {rnd}/{self.rounds} | acc={final_accuracy:.4f} | "
                    f"comm={communication_times:.0f} | delay={t_total:.2f}s", flush=True
                )

            if is_converged:
                print(f"    [{mode}] [Early Stopping] Converged at round {rnd} with acc={final_accuracy:.4f}", flush=True)
                break

        return {
            "scheme": mode,
            "rounds": int(rnd),
            "communication_times": float(communication_times),
            "accuracy": float(final_accuracy),
            "time_consumption": float(total_time_consumption),
            "energy_consumption": float(total_energy_consumption),
            "accumulated_cost": float(accumulated_cost),
        }

    def run_scheme1_proposed(self) -> dict:
        return self._run_rounds(mode="scheme1")

    def run_scheme2_dynamic_beta_only(self) -> dict:
        return self._run_rounds(mode="scheme2")

    def run_scheme3_fixed_beta(self) -> dict:
        return self._run_rounds(mode="scheme3")

    def run_scheme4_lag(self) -> dict:
        return self._run_rounds(mode="scheme4")

    def run_scheme5_traditional_async(self) -> dict:
        return self._run_rounds(mode="scheme5")

    def run_all_schemes(self) -> list[dict]:
        return [
            self.run_scheme1_proposed(),
            self.run_scheme2_dynamic_beta_only(),
            self.run_scheme3_fixed_beta(),
            self.run_scheme4_lag(),
            self.run_scheme5_traditional_async(),
        ]


def save_results_csv(results: list[dict], output_csv: str) -> None:
    if not results:
        return

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        "scheme",
        "rounds",
        "communication_times",
        "accuracy",
        "time_consumption",
        "energy_consumption",
        "accumulated_cost",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 5 schemes over FL rounds.")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--m", type=int, default=9)
    parser.add_argument("--model-path", type=str, default="./models/ppo_auv_final")
    parser.add_argument("--lag-threshold", type=float, default=1e4)
    parser.add_argument("--beta-heuristic", type=str, choices=["linear", "constant"], default="linear")
    parser.add_argument(
        "--scheme",
        type=str,
        default="all",
        choices=["all", "scheme1", "scheme2", "scheme3", "scheme4", "scheme5"],
    )
    parser.add_argument("--out-csv", type=str, default="results/schemes/scheme_results.csv")
    args = parser.parse_args()

    cfg = SchemeEvaluator.build_config(m_value=args.m, eval_interval=1)
    evaluator = SchemeEvaluator(
        config=cfg,
        rounds=args.rounds,
        model_path=args.model_path,
        lag_threshold=args.lag_threshold,
        beta_heuristic=args.beta_heuristic,
    )

    if args.scheme == "all":
        results = evaluator.run_all_schemes()
    elif args.scheme == "scheme1":
        results = [evaluator.run_scheme1_proposed()]
    elif args.scheme == "scheme2":
        results = [evaluator.run_scheme2_dynamic_beta_only()]
    elif args.scheme == "scheme3":
        results = [evaluator.run_scheme3_fixed_beta()]
    elif args.scheme == "scheme4":
        results = [evaluator.run_scheme4_lag()]
    else:
        results = [evaluator.run_scheme5_traditional_async()]

    save_results_csv(results, args.out_csv)
    print("[DONE] Saved:", args.out_csv)
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
