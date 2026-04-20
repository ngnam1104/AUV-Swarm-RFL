import numpy as np

class LazyNodeController:
    """Section 3.3: Control Model - Quản lý cơ chế Nút lười (Eq. 14 - 20)."""
    def __init__(self, M: int, N: float, N_m_dict: dict, lr: float, force_active_rounds: int = 5):
        self.M = M
        self.N = float(N)
        self.N_m_dict = dict(N_m_dict)
        self.lr = float(lr)
        self.force_active_rounds = int(force_active_rounds)
        self.lazy_consecutive = np.zeros(M, dtype=np.int32)
        
    def select_active_nodes(self, beta: float, local_grad_sq_norms: dict, global_diff_sq: float, rng: np.random.Generator) -> list[int]:
        """Chọn node active theo Eq. 19 và Eq. 20."""
        active_indices = []

        # Eq. 19: Self-Detection Threshold
        # ||N_m grad_m^{t-1}||^2 <= (N^2 / (gamma^2 M^2 beta)) * ||w(t-1)-w(t-2)||^2  => Lazy
        # Nếu > threshold => Active.
        if beta == 0:
            threshold = float("inf")
        else:
            threshold = (self.N ** 2) / ((self.lr ** 2) * (self.M ** 2) * float(beta)) * float(global_diff_sq)

        for m in range(self.M):
            if float(local_grad_sq_norms[m]) > threshold:
                active_indices.append(m)

        # Extreme case: nếu không có node active theo Eq. 19 thì random đúng 1 node.
        if len(active_indices) == 0:
            active_indices.append(int(rng.integers(low=0, high=self.M)))

        # Eq. 20: Force Active nếu ngủ liên tiếp >= tau rounds.
        for m in range(self.M):
            if self.lazy_consecutive[m] >= self.force_active_rounds:
                active_indices.append(m)

        return list(set(active_indices))
        
    def update_lazy_counters(self, active_indices: list[int]):
        """Tăng counter cho các node ngủ, reset về 0 cho node active."""
        self.lazy_consecutive += 1
        for m in active_indices:
            self.lazy_consecutive[m] = 0
