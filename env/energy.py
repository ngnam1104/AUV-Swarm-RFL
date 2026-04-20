import numpy as np

class EnergyModel:
    """Section 3.5: Energy Consumption Model - Tính E_total (Eq. 28 - 32)."""
    def __init__(self, config):
        self.cfg = config
        self.eps = 1e-12

        # Hệ số năng lượng theo phần cứng.
        self.k_cap = float(getattr(config, "k_cap", 1.25e-26))
        self.delta_pu = float(getattr(config, "delta_pu", 3.0))

        # Trọng số cân bằng trong tổng năng lượng.
        self.phi = float(getattr(config, "phi", 1.0))
        self.chi = float(getattr(config, "chi", 1.0))

    @staticmethod
    def _as_array(x) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    def compute_total_energy(
        self,
        lambda_m: np.ndarray,
        f_m: np.ndarray,
        p_m: np.ndarray,
        f_L: float,
        p_L: float,
        latency_details: dict,
    ) -> tuple[float, dict]:
        """
        Tính E_Cp, E_Cm (Follower) và E_CpL, E_CL (Leader) dựa trên latency_details.
        Trả về E_total và dictionary chứa các thành phần năng lượng để trừ pin.

        Eq. 28: E_Cp_m = k_cap * f_m^delta_pu * TLC_m
        Eq. 29: E_C_m = p_m * TLU_m
        Eq. 30: E_Cp_L = k_cap * f_L^delta_pu * (TGA_L + TGU_L)
        Eq. 31: E_C_L = p_L * max(TGD_m)
        Eq. 32: E_total = phi * (sum(E_Cp_m) + E_Cp_L) + chi * (sum(E_C_m) + E_C_L)
        """
        lambda_m = self._as_array(lambda_m)
        f_m = self._as_array(f_m)
        p_m = self._as_array(p_m)

        if not (lambda_m.shape == f_m.shape == p_m.shape):
            raise ValueError(
                "lambda_m, f_m, p_m phải cùng shape (M,)"
            )

        # Bóc tách các thành phần latency từ LatencyModel.
        TLC_m = self._as_array(latency_details["TLC_m"])
        TLU_m = self._as_array(latency_details["TLU_m"])
        TGA_L = float(latency_details["TGA_L"])
        TGU_L = float(latency_details["TGU_L"])
        TGD_m = self._as_array(latency_details["TGD_m"])

        if not (TLC_m.shape == TLU_m.shape == TGD_m.shape == lambda_m.shape):
            raise ValueError(
                "TLC_m, TLU_m, TGD_m phải cùng shape với lambda_m"
            )

        f_m_safe = np.clip(f_m, self.eps, None)
        f_L_safe = float(np.clip(f_L, self.eps, None))
        p_m_safe = np.clip(p_m, 0.0, None)
        p_L_safe = float(np.clip(p_L, 0.0, None))

        selected_m0 = latency_details.get("selected_m0")

        if selected_m0 is not None:
            # Extreme case: chỉ node m0 bị ép thức mới tiêu hao năng lượng follower.
            forced_mask = np.zeros_like(lambda_m, dtype=np.float64)
            forced_mask[int(selected_m0)] = 1.0
            eff_lambda = forced_mask
        else:
            eff_lambda = lambda_m

        # Eq. 28: Follower Computation Energy (node lười vẫn đã tốn local compute).
        E_Cp_m = self.k_cap * (f_m_safe ** self.delta_pu) * TLC_m

        # Eq. 29: Follower Communication Energy.
        E_C_m = eff_lambda * p_m_safe * TLU_m

        # Eq. 30: Leader Computation Energy.
        E_Cp_L = self.k_cap * (f_L_safe ** self.delta_pu) * (TGA_L + TGU_L)

        # Eq. 31: Leader Communication Energy (broadcast theo follower chậm nhất).
        E_C_L = p_L_safe * float(np.max(TGD_m))

        # Eq. 32: Total Energy Consumption (đã sửa typo paper).
        E_total = self.phi * (float(np.sum(E_Cp_m)) + E_Cp_L) + self.chi * (float(np.sum(E_C_m)) + E_C_L)

        details = {
            "E_Cp_m": E_Cp_m,
            "E_C_m": E_C_m,
            "E_Cp_L": E_Cp_L,
            "E_C_L": E_C_L,
            "phi": self.phi,
            "chi": self.chi,
            "selected_m0": selected_m0,
        }
        return float(E_total), details

    def compute_total_energy_from_latency(
        self,
        latency_model,
        lambda_m: np.ndarray,
        f_m: np.ndarray,
        p_m: np.ndarray,
        f_L: float,
        p_L: float,
    ) -> tuple[float, dict, float, dict]:
        """
        Pipeline tiện ích: lấy latency từ LatencyModel (và gián tiếp từ CommunicationModel),
        sau đó tính energy trên cùng action/state.

        Trả về:
        - E_total
        - energy_details
        - T_total
        - latency_details
        """
        T_total, latency_details = latency_model.compute_total_time(
            active_mask=lambda_m,
            f_m=f_m,
            p_m=p_m,
            f_L=f_L,
            p_L=p_L,
        )
        E_total, energy_details = self.compute_total_energy(
            lambda_m=lambda_m,
            f_m=f_m,
            p_m=p_m,
            f_L=f_L,
            p_L=p_L,
            latency_details=latency_details,
        )
        return E_total, energy_details, T_total, latency_details


if __name__ == "__main__":
    import sys
    import os
    from types import SimpleNamespace

    # Cấu hình đường dẫn root để có thể thực thi file từ ngoài
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.settings import HW_CFG, FL_CFG, ACOUSTIC_CFG
    from env.communication import CommunicationModel
    from env.latency import LatencyModel

    rng = np.random.default_rng(42)

    M = 3
    energy_model = EnergyModel(HW_CFG)

    # Gộp config để LatencyModel lấy đủ tham số FL + Acoustic.
    cfg = SimpleNamespace(
        **vars(ACOUSTIC_CFG),
        **vars(FL_CFG),
    )
    comm_model = CommunicationModel(cfg)
    latency_model = LatencyModel(cfg, comm_model)
    latency_model.M = M
    latency_model.N_m = latency_model._as_array(FL_CFG.N_m, M)
    latency_model.c_m = latency_model._as_array(FL_CFG.c_m, M)

    lambda_m = rng.integers(0, 2, size=M).astype(np.float64)
    f_m = rng.uniform(HW_CFG.f_min, HW_CFG.f_max, size=M)
    p_m = rng.uniform(0.01, HW_CFG.p_max, size=M)
    f_L = 0.5e9
    p_L = 0.1

    E_total, details, T_total, latency_details = energy_model.compute_total_energy_from_latency(
        latency_model=latency_model,
        lambda_m=lambda_m,
        f_m=f_m,
        p_m=p_m,
        f_L=f_L,
        p_L=p_L,
    )

    print("=== Energy Test (Eq. 28-32) ===")
    print("lambda_m:", lambda_m)
    print("f_m (Hz):", f_m)
    print("p_m (W):", p_m)
    print("f_L (Hz):", f_L)
    print("p_L (W):", p_L)

    print("\n=== Latency (from latency.py -> communication.py) ===")
    print("T_total (s):", T_total)
    print("TLC_m (s):", latency_details["TLC_m"])
    print("TLU_m (s):", latency_details["TLU_m"])
    print("TGA_L (s):", latency_details["TGA_L"])
    print("TGU_L (s):", latency_details["TGU_L"])
    print("TGD_m (s):", latency_details["TGD_m"])

    print("\n=== Components ===")
    print("E_Cp_m (J):", details["E_Cp_m"])
    print("E_C_m (J):", details["E_C_m"])
    print("E_Cp_L (J):", details["E_Cp_L"])
    print("E_C_L (J):", details["E_C_L"])
    print("phi:", details["phi"], "chi:", details["chi"])

    print("\nE_total (J):", E_total)
