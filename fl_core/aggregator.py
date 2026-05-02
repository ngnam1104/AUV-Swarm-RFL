import torch
from fl_core.models import device

class AsyncAggregator:
    """Xử lý logic nén và tổng hợp bất đồng bộ (Lazy Nodes)."""
    def __init__(self, num_workers: int, initial_weights: dict, N_m_dict: dict, N_total: float):
        self.num_workers = num_workers
        self.N_m_dict = dict(N_m_dict)  # {worker_id: N_m}
        self.N_total = float(N_total)
        # Giữ danh sách keys đú ng thứ tự để stack/unstack đúng
        self._param_keys = list(initial_weights.keys())
        # Cache giữ trọng số mới nhất của toàn bộ workers (dưới dạng flat tensor).
        self.worker_cache: dict[int, dict] = {
            worker_id: {k: v.detach().clone() for k, v in initial_weights.items()}
            for worker_id in range(num_workers)
        }
        # Pre-compute weight scalars N_m/N cho từng worker
        self._w_scalars = {
            wid: self.N_m_dict[wid] / self.N_total
            for wid in range(num_workers)
        }

    def update_and_aggregate(self, active_weights_dict: dict) -> dict:
        """Cập nhật cache từ các node active và tính w_global mới (Weighted FedAvg theo Eq. 2)."""
        # 1) Update cache của các workers active (chỉ clone tensor, không wrap dict thừa).
        for worker_id, new_weights in active_weights_dict.items():
            cache = self.worker_cache[worker_id]
            for k, v in new_weights.items():
                cache[k] = v.detach() if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)

        if len(self.worker_cache) == 0:
            raise ValueError("worker_cache rỗng, không thể aggregate")

        # 2) Weighted FedAvg — vector hóa bằng torch.stack cho mỗi key.
        # Giảm số lần Python loop từ n_workers x n_keys xuống n_keys lần.
        w_global = {}
        scalars = torch.tensor(
            [self._w_scalars[wid] for wid in range(self.num_workers)],
            dtype=torch.float32, device=device
        )  # shape: (M,)

        for key in self._param_keys:
            # stack: (M, *param_shape)
            stacked = torch.stack(
                [self.worker_cache[wid][key].to(device) for wid in range(self.num_workers)]
            )
            # broadcast scalars: (M, 1, ...) x (M, *param_shape)
            view_shape = (self.num_workers,) + (1,) * (stacked.dim() - 1)
            w_global[key] = (scalars.view(view_shape) * stacked).sum(dim=0)

        return w_global
