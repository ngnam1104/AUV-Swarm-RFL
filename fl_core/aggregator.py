import torch


class AsyncAggregator:
    """Xử lý logic nén và tổng hợp bất đồng bộ (Lazy Nodes)."""
    def __init__(self, num_workers: int, initial_weights: dict, N_m_dict: dict, N_total: float):
        self.num_workers = num_workers
        self.N_m_dict = dict(N_m_dict)  # {worker_id: N_m}
        self.N_total = float(N_total)
        # Cache giữ trọng số mới nhất của toàn bộ workers.
        self.worker_cache = {
            worker_id: {k: v.detach().cpu().clone() for k, v in initial_weights.items()}
            for worker_id in range(num_workers)
        }

    def update_and_aggregate(self, active_weights_dict: dict) -> dict:
        """Cập nhật cache từ các node active và tính w_global mới (Weighted FedAvg theo Eq. 2)."""
        # 1) Update cache của các workers active bằng weights mới nhận được.
        for worker_id, new_weights in active_weights_dict.items():
            self.worker_cache[worker_id] = {
                k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else torch.tensor(v))
                for k, v in new_weights.items()
            }

        if len(self.worker_cache) == 0:
            raise ValueError("worker_cache rỗng, không thể aggregate")

        # 2) Weighted FedAvg (Eq. 2): w_global = Σ (N_m / N) * w_m
        # Trọng số theo N_m/N thay vì chia đều.
        sample_key = list(list(self.worker_cache.values())[0].keys())
        w_global = {}

        for key in sample_key:
            acc = torch.zeros_like(list(self.worker_cache.values())[0][key])
            for worker_id, w in self.worker_cache.items():
                weight = self.N_m_dict[worker_id] / self.N_total
                acc += weight * w[key]
            w_global[key] = acc

        # 3) Trả về global weights mới.
        return w_global
