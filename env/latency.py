import numpy as np
try:
    from env.communication import CommunicationModel
except ModuleNotFoundError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.communication import CommunicationModel

class LatencyModel:
    """Section 3.4: Latency Model - Tính T_total (Eq. 21 - 27)."""
    def __init__(self, config, comm_model: CommunicationModel):
        self.cfg = config
        self.comm = comm_model
        self.eps = 1e-12

        # Tham số cố định theo bài báo / config.
        self.M = int(getattr(config, "M", 1))
        self.N_m = self._as_array(getattr(config, "N_m", 1.0), self.M)
        self.c_m = self._as_array(getattr(config, "c_m", 1.0), self.M)

        # |w| = alpha * zeta. Nếu không có alpha/zeta riêng, suy từ w_size_bits.
        self.zeta = float(getattr(config, "zeta", 64.0))
        w_size_bits = float(getattr(config, "w_size_bits", 1594.0 * self.zeta))
        self.alpha = float(getattr(config, "alpha", w_size_bits / self.zeta))
        self.psi = float(getattr(config, "packet_overhead", 64.0))

        self.c0 = float(getattr(config, "c_0", 50.0))
        self.c_L_prime = float(getattr(config, "c_L_prime", 1.0))
        self.T_prime = float(getattr(config, "T_prime", 1.0))

    @staticmethod
    def _as_array(x, size: int) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(size, float(arr), dtype=np.float64)
        if arr.shape[0] != size:
            raise ValueError(f"Expected array size {size}, got {arr.shape[0]}")
        return arr

    def _model_size_bits(self) -> float:
        return float(self.alpha * self.zeta)

    def local_computation_latency(self, f_m: np.ndarray) -> np.ndarray:
        f_m = self._as_array(f_m, self.M)
        f_m_safe = np.clip(f_m, self.eps, None)
        # Eq. 21: Local Parameter Calculating Latency
        # TLC_m(t) = (N_m * c_m) / f_m(t)
        return (self.N_m * self.c_m) / f_m_safe

    def upload_latency(self, lambda_m: np.ndarray, R_U_m: np.ndarray) -> np.ndarray:
        lambda_m = self._as_array(lambda_m, self.M)
        R_U_m = self._as_array(R_U_m, self.M)
        R_U_safe = np.clip(R_U_m, self.eps, None)
        w_m = self._model_size_bits()
        # Eq. 22: Uploading Latency
        # TLU_m(t) = (|w_m| * lambda_m(t)) / R_U_m + (psi * lambda_m(t)) / R_U_m
        return (w_m * lambda_m) / R_U_safe + (self.psi * lambda_m) / R_U_safe

    def global_aggregation_latency(self, lambda_m: np.ndarray, f_L: float) -> float:
        lambda_m = self._as_array(lambda_m, self.M)
        f_L_safe = float(np.clip(f_L, self.eps, None))
        w_m = self._model_size_bits()
        active_count = float(np.sum(lambda_m))
        # Eq. 23: Global Parameter Aggregating Latency
        # TGA_L(t) = c0 * sum_m(|w_m|) / f_L(t), với tổng trên các node upload (lambda_m = 1)
        return self.c0 * (w_m * active_count) / f_L_safe

    def global_update_latency(self, f_L: float) -> float:
        f_L_safe = float(np.clip(f_L, self.eps, None))
        # Eq. 24: Global Parameter Updating Latency
        # TGU_L(t) = c'_L / f_L(t)
        return self.c_L_prime / f_L_safe

    def download_latency(self, R_D_m: np.ndarray) -> np.ndarray:
        R_D_m = self._as_array(R_D_m, self.M)
        R_D_safe = np.clip(R_D_m, self.eps, None)
        w = self._model_size_bits()
        # Eq. 25: Downloading Latency
        # TGD_m(t) = |w| / R_D_m + psi / R_D_m
        return w / R_D_safe + self.psi / R_D_safe

    def compute_total_latency(
        self,
        lambda_m: np.ndarray,
        f_m: np.ndarray,
        f_L: float,
        R_U_m: np.ndarray,
        R_D_m: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, dict]:
        """Tính tổng độ trễ Eq. 21-27 cho 1 vòng FL."""
        lambda_m = self._as_array(lambda_m, self.M)
        TLC_m = self.local_computation_latency(f_m)
        TLU_m = self.upload_latency(lambda_m, R_U_m)
        TGA_L = self.global_aggregation_latency(lambda_m, f_L)
        TGU_L = self.global_update_latency(f_L)
        TGD_m = self.download_latency(R_D_m)

        # Eq. 26: Total Latency (normal case)
        # T(t) = max_m{TLC_m + TLU_m} + max_m{TGA_L + TGU_L + TGD_m}
        # Với vế local chỉ xét node active theo lambda_m.
        max_local = float(np.max((TLC_m + TLU_m) * lambda_m))
        max_global = float(np.max(TGA_L + TGU_L + TGD_m))

        if np.sum(lambda_m) == 0:
            if rng is None:
                rng = np.random.default_rng()
            m0 = int(rng.integers(low=0, high=self.M))
            # Eq. 27: Extreme case (no upload)
            # T(t) = TLC_m0 + TLU_m0 + max_m{TGA_L + TGU_L + TGD_m} + T'
            total_latency = float(TLC_m[m0] + TLU_m[m0] + max_global + self.T_prime)
            selected_m0 = m0
        else:
            total_latency = float(max_local + max_global)
            selected_m0 = None

        details = {
            "TLC_m": TLC_m,
            "TLU_m": TLU_m,
            "TGA_L": TGA_L,
            "TGU_L": TGU_L,
            "TGD_m": TGD_m,
            "selected_m0": selected_m0,
            "sum_lambda": float(np.sum(lambda_m)),
            "T_prime": self.T_prime,
        }
        return total_latency, details

    def compute_total_time(self, active_mask: np.ndarray, f_m: np.ndarray, p_m: np.ndarray, f_L: float, p_L: float) -> tuple[float, dict]:
        """
        Tính T_LC, T_LU, T_GA, T_GU, T_GD.
        Trả về T_total và dictionary chứa các thành phần time để debug.
        """
        R_U_m = self.comm.uplink_rate(p_m, active_mask=active_mask)
        R_D_scalar = self.comm.downlink_rate(p_L)
        R_D_m = np.full(self.M, float(np.asarray(R_D_scalar).reshape(-1)[0]), dtype=np.float64)
        return self.compute_total_latency(
            lambda_m=active_mask,
            f_m=f_m,
            f_L=f_L,
            R_U_m=R_U_m,
            R_D_m=R_D_m,
        )


if __name__ == "__main__":
    import sys
    import os

    # Cấu hình đường dẫn root để có thể thực thi file từ ngoài
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.settings import ACOUSTIC_CFG, FL_CFG

    rng = np.random.default_rng(42)

    comm_model = CommunicationModel(ACOUSTIC_CFG)
    latency_model = LatencyModel(FL_CFG, comm_model)

    # Test case theo yêu cầu: M=3
    latency_model.M = 3
    latency_model.N_m = latency_model._as_array(FL_CFG.N_m, 3)
    latency_model.c_m = latency_model._as_array(FL_CFG.c_m, 3)

    lambda_m = rng.integers(0, 2, size=3).astype(np.float64)
    f_m = rng.uniform(0.2e9, 0.4e9, size=3)
    f_L = 0.5e9
    R_U_m = rng.uniform(1e3, 1e5, size=3)
    R_D_m = rng.uniform(1e3, 1e5, size=3)

    total_latency, details = latency_model.compute_total_latency(
        lambda_m=lambda_m,
        f_m=f_m,
        f_L=f_L,
        R_U_m=R_U_m,
        R_D_m=R_D_m,
        rng=rng,
    )

    print("=== Latency Test (Eq. 21-27) ===")
    print("lambda_m:", lambda_m)
    print("f_m (Hz):", f_m)
    print("f_L (Hz):", f_L)
    print("R_U_m (bps):", R_U_m)
    print("R_D_m (bps):", R_D_m)

    print("\n=== Components ===")
    print("TLC_m (s):", details["TLC_m"])
    print("TLU_m (s):", details["TLU_m"])
    print("TGA_L (s):", details["TGA_L"])
    print("TGU_L (s):", details["TGU_L"])
    print("TGD_m (s):", details["TGD_m"])
    if details["selected_m0"] is not None:
        print("selected_m0 (extreme case):", details["selected_m0"])
        print("T_prime (s):", details["T_prime"])

    print("\nTotal Latency (s):", total_latency)
