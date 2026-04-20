import numpy as np

class RewardModel:
    """Section 3.6 & 4.1: Reward model theo đúng công thức của bài báo."""
    def __init__(self, config):
        self.cfg = config
        self.penalty = float(getattr(config, "penalty", 10.0))
        self.E_m_thd = float(getattr(config, "E_m_thd", 0.07))
        self.E_L_thd = float(getattr(config, "E_L_thd", 0.8))
        self.last_violation_detail = {
            "follower_violated": False,
            "leader_violated": False,
            "P_m_avg": None,
            "P_L_avg": None,
            "violations": [],
        }

    def compute_reward(
        self,
        T_total: float,
        E_total: float,
        E_m: np.ndarray,
        E_L: float,
    ) -> tuple[float, float, bool]:
        """
        Tính reward theo Eq. 33 và ràng buộc Section 3.6.

        Returns:
        - reward: giá trị reward cuối cùng cho agent
        - cost: tổng chi phí tại thời điểm t
        - is_violated: có vi phạm ràng buộc năng lượng hay không
        """
        E_m = np.asarray(E_m, dtype=np.float64)
        E_L = float(E_L)

        # Eq. 33 (theo yêu cầu hiện tại): Hàm Cost không trọng số, Cost(t) = T(t) + E(t).
        cost = float(T_total) + float(E_total)

        # Section 3.6 (điều chỉnh theo thực nghiệm tích hợp):
        # So sánh ngưỡng phần cứng theo công suất trung bình của vòng hiện tại.
        # P_m_avg = E_m / T_total, P_L_avg = E_L / T_total.
        T_safe = max(float(T_total), 1e-6)
        P_m_avg = E_m / T_safe
        P_L_avg = E_L / T_safe

        # Section 3.6: Ràng buộc follower - P_m_avg <= E_m_thd cho mọi m.
        follower_violated = bool(np.any(P_m_avg > self.E_m_thd))
        # Section 3.6: Ràng buộc leader - P_L_avg <= E_L_thd.
        leader_violated = bool(P_L_avg > self.E_L_thd)

        is_violated = follower_violated or leader_violated
        violations = []
        if follower_violated:
            violations.append("Follower average power constraint (P_m_avg > E_m_thd)")
        if leader_violated:
            violations.append("Leader average power constraint (P_L_avg > E_L_thd)")

        self.last_violation_detail = {
            "follower_violated": follower_violated,
            "leader_violated": leader_violated,
            "P_m_avg": P_m_avg,
            "P_L_avg": P_L_avg,
            "violations": violations,
        }

        if is_violated:
            pass # Removed log spamming

        # Bỏ đi việc áp dụng penalty do safety layer đã đảm bảo không bị quá mức năng lượng
        # Section 4.1 (MDP): Reward(t) = -Cost(t)
        reward = -cost

        return float(reward), float(cost), is_violated

    def calculate_cost_and_penalty(self, T_total: float, E_total: float, power_avg_m: np.ndarray, power_avg_L: float, dead_count: int, active_count: int) -> tuple[float, float, float]:
        """Hàm tương thích ngược: trả về (cost, penalty, violation_flag)."""
        reward, cost, is_violated = self.compute_reward(
            T_total=T_total,
            E_total=E_total,
            E_m=power_avg_m,
            E_L=power_avg_L,
        )
        _ = reward
        penalty = self.penalty if is_violated else 0.0
        return float(cost), float(penalty), float(is_violated)

    def compute_ar_gain(self, delta_acc: float, resource_cost: float, step: int) -> float:
        """Hàm tương thích ngược: AR-gain bị loại bỏ theo yêu cầu công thức paper."""
        _ = (delta_acc, step)
        return -float(resource_cost)


if __name__ == "__main__":
    import os
    import sys
    from types import SimpleNamespace

    # Cấu hình đường dẫn root để có thể thực thi file từ ngoài
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config.settings import ACOUSTIC_CFG, FL_CFG, HW_CFG, RL_CFG
    from env.communication import CommunicationModel
    from env.latency import LatencyModel
    from env.energy import EnergyModel

    rng = np.random.default_rng(42)

    # Gộp config để các module dùng đúng tập tham số cần thiết.
    full_cfg = SimpleNamespace(
        **vars(ACOUSTIC_CFG),
        **vars(FL_CFG),
        **vars(HW_CFG),
        **vars(RL_CFG),
    )

    M = 3
    comm_model = CommunicationModel(full_cfg)
    latency_model = LatencyModel(full_cfg, comm_model)
    energy_model = EnergyModel(full_cfg)
    reward_model = RewardModel(full_cfg)

    # Đồng bộ M test = 3 cho latency model.
    latency_model.M = M
    latency_model.N_m = latency_model._as_array(FL_CFG.N_m, M)
    latency_model.c_m = latency_model._as_array(FL_CFG.c_m, M)

    lambda_m = rng.integers(0, 2, size=M).astype(np.float64)
    f_m = rng.uniform(HW_CFG.f_min, HW_CFG.f_max, size=M)
    p_m = rng.uniform(0.01, HW_CFG.p_max, size=M)
    f_L = 0.5e9
    p_L = 0.1

    # Lấy latency từ latency.py (bên trong dùng rate từ communication.py).
    T_total, latency_details = latency_model.compute_total_time(
        active_mask=lambda_m,
        f_m=f_m,
        p_m=p_m,
        f_L=f_L,
        p_L=p_L,
    )

    # Lấy energy từ energy.py dựa trên latency_details thực.
    E_total, energy_details = energy_model.compute_total_energy(
        lambda_m=lambda_m,
        f_m=f_m,
        p_m=p_m,
        f_L=f_L,
        p_L=p_L,
        latency_details=latency_details,
    )

    # Tính reward theo Eq. 33 + ràng buộc Section 3.6 + Section 4.1.
    reward, cost, is_violated = reward_model.compute_reward(
        T_total=T_total,
        E_total=E_total,
        E_m=energy_details["E_Cp_m"] + energy_details["E_C_m"],
        E_L=energy_details["E_Cp_L"] + energy_details["E_C_L"],
    )

    print("=== Reward Test (Integrated Pipeline) ===")
    print("lambda_m:", lambda_m)
    print("T_total (s):", T_total)
    print("E_total (J):", E_total)
    print("E_m (J):", energy_details["E_Cp_m"] + energy_details["E_C_m"])
    print("E_L (J):", energy_details["E_Cp_L"] + energy_details["E_C_L"])
    print("cost:", cost)
    print("is_violated:", is_violated)
    print("violation_detail:", reward_model.last_violation_detail)
    print("reward:", reward)
