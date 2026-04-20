import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.communication import CommunicationModel
from env.latency import LatencyModel
from env.energy import EnergyModel
from env.reward import RewardModel

class AUVSwarmEnv(gym.Env):
    """
    Môi trường RL điều phối AUV Swarm Federated Learning.
    State (Eq. 41): s(t) = (p_m(t-1), f_m(t-1), p_L(t-1), f_L(t-1), beta(t-1))
    Action (Eq. 42): a(t) = (p_m(t), f_m(t), p_L(t), f_L(t), beta(t))
    """
    def __init__(self, fl_sim, config):
        super().__init__()
        self.fl_sim = fl_sim
        self.cfg = config
        
        # Khởi tạo các module Vật lý
        self.comm_model = CommunicationModel(config)
        self.latency_model = LatencyModel(config, self.comm_model)
        self.energy_model = EnergyModel(config)
        self.reward_model = RewardModel(config)
        
        self.M = int(getattr(config, "M", 9))
        self.dim = 2 * self.M + 3  # (p_m, f_m) * M + p_L + f_L + beta
        
        # Giới hạn vật lý (Physical Bounds)
        self.p_min, self.p_max = 0.01, float(getattr(config, "p_max", 0.2))
        self.f_min, self.f_max = float(getattr(config, "f_min", 0.2e9)), float(getattr(config, "f_max", 0.4e9))
        
        # Action & Observation Space chuẩn hóa trong [-1, 1] cho PPO
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.dim,), dtype=np.float32)
        
        self.current_state_norm = None
        self.step_idx = 0
        self.max_steps = int(getattr(config, "max_fl_rounds", 1000))
        self.accumulated_cost = 0.0

    def _unscale_action(self, action_norm: np.ndarray):
        """Giải mã action từ [-1, 1] sang dải vật lý thực tế."""
        action_norm = np.clip(action_norm, -1.0, 1.0)
        
        # Mapping: y = (x + 1)/2 * (max - min) + min
        p_m = (action_norm[0 : self.M] + 1.0) / 2.0 * (self.p_max - self.p_min) + self.p_min
        f_m = (action_norm[self.M : 2*self.M] + 1.0) / 2.0 * (self.f_max - self.f_min) + self.f_min
        
        p_L = (action_norm[2*self.M] + 1.0) / 2.0 * (self.p_max - self.p_min) + self.p_min
        f_L = (action_norm[2*self.M + 1] + 1.0) / 2.0 * (self.f_max - self.f_min) + self.f_min
        
        beta = (action_norm[2*self.M + 2] + 1.0) / 2.0  # Beta thuộc [0, 1]
        
        return p_m, f_m, p_L, f_L, beta

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)
        self.step_idx = 0
        self.accumulated_cost = 0.0
        
        # Reset FL Simulator (Nếu có)
        if self.fl_sim:
            self.fl_sim.reset()
            
        # Khởi tạo trạng thái ban đầu ngẫu nhiên trong khoảng [-1, 1]
        self.current_state_norm = self.np_random.uniform(low=-1.0, high=1.0, size=(self.dim,)).astype(np.float32)
        
        return self.current_state_norm, {}

    def _apply_safety_layer(self, lambda_m, p_m, f_m, p_L, f_L):
        """
        Chiếu (project) các hành động để luôn thỏa mãn giới hạn năng lượng Eq. 39-40.
        Sử dụng cơ chế Binary Search để đưa các tham số phần cứng sát ngưỡng an toàn nhất có thể.
        """
        def check_safe(p_m_test, f_m_test, p_L_test, f_L_test):
            E_total, energy_details, T_total, latency_details = self.energy_model.compute_total_energy_from_latency(
                latency_model=self.latency_model,
                lambda_m=lambda_m,
                f_m=f_m_test,
                p_m=p_m_test,
                f_L=f_L_test,
                p_L=p_L_test,
            )
            T_safe = max(float(T_total), 1e-6)
            P_m_avg = (energy_details["E_Cp_m"] + energy_details["E_C_m"]) / T_safe
            P_L_avg = (energy_details["E_Cp_L"] + energy_details["E_C_L"]) / T_safe
            
            is_safe = not (np.any(P_m_avg > self.cfg.E_m_thd) or P_L_avg > self.cfg.E_L_thd)
            return is_safe, E_total, energy_details, T_total, latency_details

        # Step 2: Define the Absolute Minimum Bounds (Lower Bound)
        p_m_min = np.full_like(p_m, self.p_min)
        f_m_min = np.full_like(f_m, self.f_min)
        p_L_min = float(self.p_min)
        f_L_min = float(self.f_min)
        
        # Initialize the best known safe state with the lower bound
        _, best_E_total, best_energy_details, best_T_total, best_latency_details = check_safe(p_m_min, f_m_min, p_L_min, f_L_min)
        best_p_m, best_f_m, best_p_L, best_f_L = p_m_min.copy(), f_m_min.copy(), p_L_min, f_L_min

        # Step 3: Fast-Path the Target Action (Upper Bound)
        is_safe, E_total, energy_details, T_total, latency_details = check_safe(p_m, f_m, p_L, f_L)
        if is_safe:
            return p_m, f_m, p_L, f_L, E_total, energy_details, T_total, latency_details

        # Step 4: Execute the Binary Search Loop
        low = 0.0
        high = 1.0
        for _ in range(10):
            mid = (low + high) / 2.0
            
            # Interpolate test values
            p_m_test = p_m_min + mid * (p_m - p_m_min)
            f_m_test = f_m_min + mid * (f_m - f_m_min)
            p_L_test = p_L_min + mid * (p_L - p_L_min)
            f_L_test = f_L_min + mid * (f_L - f_L_min)
            
            is_safe, E_total, energy_details, T_total, latency_details = check_safe(p_m_test, f_m_test, p_L_test, f_L_test)
            if is_safe:
                low = mid
                best_p_m, best_f_m, best_p_L, best_f_L = p_m_test, f_m_test, p_L_test, f_L_test
                best_E_total, best_energy_details, best_T_total, best_latency_details = E_total, energy_details, T_total, latency_details
            else:
                high = mid

        # Step 5: Return the Optimized State
        return best_p_m, best_f_m, best_p_L, best_f_L, best_E_total, best_energy_details, best_T_total, best_latency_details

    def step(self, action: np.ndarray):
        self.step_idx += 1
        
        # 1. Giải mã Action
        p_m, f_m, p_L, f_L, beta = self._unscale_action(action)
        
        # 2. FL Simulator: Tính toán danh sách node tham gia dựa trên beta (Eq. 18-19)
        if self.fl_sim:
            accuracy, active_indices, _, is_converged = self.fl_sim.sync_run_step(beta, self.step_idx)
            timing_info = getattr(self.fl_sim, "last_timing_stats", {})
        else:
            probs = np.random.rand(self.M)
            active_indices = np.where(probs <= beta)[0]
            accuracy = np.nan
            is_converged = False
            timing_info = {}

        lambda_m = np.zeros(self.M, dtype=float)
        lambda_m[active_indices] = 1.0
        
        # 3. Tính Thời gian & Năng lượng kèm Safety Layer (kiểm duyệt năng lượng)
        p_m, f_m, p_L, f_L, E_total, energy_details, T_total, latency_details = self._apply_safety_layer(
            lambda_m, p_m, f_m, p_L, f_L
        )

        # 4. Tính Reward (Eq. 33 & 4.1)
        E_m_array = energy_details["E_Cp_m"] + energy_details["E_C_m"]
        E_L_val = energy_details["E_Cp_L"] + energy_details["E_C_L"]

        reward, cost, is_violated = self.reward_model.compute_reward(
            T_total=T_total,
            E_total=E_total,
            E_m=E_m_array,
            E_L=E_L_val,
        )

        self.accumulated_cost += float(cost)

        # 5. Check Termination: hết rounds HOẶC hội tụ sớm.
        terminated = bool(self.step_idx >= self.max_steps or is_converged)
        truncated = False

        # 5b. Phạt nặng nếu hết rounds mà chưa hội tụ (khuyến khích RL tìm hội tụ sớm).
        if terminated and not is_converged:
            reward -= 5000.0

        # 6. Cập nhật State (Eq. 41)
        self.current_state_norm = action.astype(np.float32)

        info = {
            "step_idx": int(self.step_idx),
            "max_steps": int(self.max_steps),
            "T_total": T_total,
            "E_total": E_total,
            "cost": cost,
            "accumulated_cost": float(self.accumulated_cost),
            "is_violated": is_violated,
            "is_converged": bool(is_converged),
            "active_nodes": int(np.sum(lambda_m)),
            "accuracy": float(accuracy),
            "timing": timing_info,
        }
        
        return self.current_state_norm, float(reward), terminated, truncated, info