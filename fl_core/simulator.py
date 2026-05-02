import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import DataLoader

from fl_core.aggregator import AsyncAggregator
from fl_core.control import LazyNodeController
from fl_core.dataset import DataManager
from fl_core.models import ModelUtils, SimpleNN, device
from fl_core.worker import LocalWorker
from fl_core.early_stopping import EarlyStopping


class FLSimulator:
    """Quản lý tiến trình học. Tích hợp LazyNodeController."""
    def __init__(self, config):
        cpu_threads = os.cpu_count() or 1
        # Giới hạn intra-op threads của PyTorch để tránh over-subscribe khi dùng ThreadPoolExecutor
        torch.set_num_threads(max(1, cpu_threads // 2))
        self._n_workers_parallel = min(cpu_threads, 8)  # số thread chạy song song local_train

        self.cfg = config
        self.M = int(getattr(config, "M", 9))
        self.lr = float(getattr(config, "lr", 0.01))
        self.num_epochs = int(getattr(config, "num_epochs", 1))
        self.batch_size = int(getattr(config, "batch_size", 64))
        self.force_active_rounds = int(getattr(config, "max_lazy_rounds", 5))
        self.eval_interval = int(getattr(config, "eval_interval", 10))
        self.rng = np.random.default_rng(seed=42)

        # Dữ liệu và chia user.
        dataset_size = int(getattr(config, 'dataset_size', 42000))
        self.train_dataset, self.test_dataset, self.dict_users = DataManager.get_mnist_data(
            num_users=self.M,
            iid=True,
            max_train_size=dataset_size,
        )
        self.N_m_dict = {m: float(len(self.dict_users[m])) for m in range(self.M)}
        self.N_total = float(sum(self.N_m_dict.values()))

        # Khởi tạo model toàn cục và các worker cục bộ.
        self.global_model = SimpleNN().to(device)
        self.w_global = ModelUtils.get_params(self.global_model)
        self.w_prev = None

        # Preload test set to GPU
        temp_loader = DataLoader(self.test_dataset, batch_size=2048, shuffle=False, num_workers=0)
        self.test_images = []
        self.test_labels = []
        for imgs, lbls in temp_loader:
            self.test_images.append(imgs)
            self.test_labels.append(lbls)
        if self.test_images:
            self.test_images = torch.cat(self.test_images, dim=0).to(device)
            self.test_labels = torch.cat(self.test_labels, dim=0).to(device)

        self.workers = [
            LocalWorker(worker_id=m, dataset=self.train_dataset, idxs=self.dict_users[m])
            for m in range(self.M)
        ]

        self.aggregator = AsyncAggregator(
            num_workers=self.M,
            initial_weights=self.w_global,
            N_m_dict=self.N_m_dict,
            N_total=self.N_total,
        )
        self.controller = LazyNodeController(
            M=self.M,
            N=self.N_total,
            N_m_dict=self.N_m_dict,
            lr=self.lr,
            force_active_rounds=self.force_active_rounds,
        )
        self.last_timing_stats = {}
        self.last_accuracy = 0.0
        self.early_stopping = EarlyStopping(patience=20, min_delta=0.001)

    def sync_run_step(self, beta: float, rnd: int) -> tuple[float, list[int], float, bool]:
        """
        1. Chạy local_train lấy trọng số mới.
        2. Gọi self.controller.select_active_nodes() lấy active_indices.
        3. Aggregation và tính Accuracy.
        4. Kiểm tra Early Stopping trên global accuracy.
        5. Trả về (accuracy, active_indices, global_diff_norm, is_converged).
        """
        t_step_start = time.perf_counter()

        # 1) Tính ||w(t-1) - w(t-2)||^2 bằng stacked tensor (nhanh hơn Python sum).
        t_global_diff_start = time.perf_counter()
        if self.w_prev is None:
            global_diff_sq = 0.0
        else:
            stacked_new = torch.cat([v.flatten() for v in self.w_global.values()])
            stacked_old = torch.cat([v.flatten() for v in self.w_prev.values()])
            global_diff_sq = float(((stacked_new - stacked_old) ** 2).sum().item())
        t_global_diff = time.perf_counter() - t_global_diff_start

        # 2) Local train song song cho TẤT CẢ workers bằng ThreadPoolExecutor.
        t_local_train_start = time.perf_counter()
        local_weights: dict[int, dict] = {}
        local_grad_sq_norms: dict[int, float] = {}
        worker_times: list[float] = []

        # Stacked global params một lần để tính grad norm nhanh
        stacked_global = torch.cat([v.flatten() for v in self.w_global.values()])
        N_m_values = torch.tensor(
            [self.N_m_dict[w.worker_id] for w in self.workers],
            dtype=torch.float32, device=stacked_global.device
        )

        def _train_one(worker):
            t_w = time.perf_counter()
            w_local, _ = worker.local_train(
                global_params=self.w_global,
                num_epochs=self.num_epochs,
                lr=self.lr,
                batch_size=self.batch_size,
            )
            elapsed = time.perf_counter() - t_w
            return worker.worker_id, w_local, elapsed

        with ThreadPoolExecutor(max_workers=self._n_workers_parallel) as pool:
            futures = {pool.submit(_train_one, w): w for w in self.workers}
            for fut in as_completed(futures):
                wid, w_local, elapsed = fut.result()
                local_weights[wid] = w_local
                worker_times.append(elapsed)

        # Tính grad norms trên GPU sau khi tất cả workers xong
        for worker in self.workers:
            wid = worker.worker_id
            w_local = local_weights[wid]
            N_m = self.N_m_dict[wid]
            stacked_local = torch.cat([v.flatten() for v in w_local.values()])
            grad_sq = float(((stacked_local - stacked_global) * (N_m / self.lr)).pow(2).sum().item())
            local_grad_sq_norms[wid] = grad_sq

        t_local_train = time.perf_counter() - t_local_train_start

        # 3) Chọn active node theo Lazy Node Controller.
        t_select_start = time.perf_counter()
        active_indices = self.controller.select_active_nodes(
            beta=beta,
            local_grad_sq_norms=local_grad_sq_norms,
            global_diff_sq=global_diff_sq,
            rng=self.rng,
        )
        t_select = time.perf_counter() - t_select_start

        # 4) Cập nhật bộ đếm lazy rounds.
        t_counter_start = time.perf_counter()
        self.controller.update_lazy_counters(active_indices)
        t_counter = time.perf_counter() - t_counter_start

        # 5) Chỉ giữ weights của các worker active.
        t_active_dict_start = time.perf_counter()
        active_weights_dict = {m: local_weights[m] for m in active_indices}
        t_active_dict = time.perf_counter() - t_active_dict_start

        # 6) Aggregation bất đồng bộ (cache-based FedAvg).
        t_agg_start = time.perf_counter()
        new_w_global = self.aggregator.update_and_aggregate(active_weights_dict)
        t_agg = time.perf_counter() - t_agg_start

        # 7) Lưu lịch sử global model cho vòng sau — dùng clone thay vì deepcopy.
        t_model_update_start = time.perf_counter()
        self.w_prev = {k: v.clone() for k, v in self.w_global.items()}
        self.w_global = {k: v.clone() for k, v in new_w_global.items()}
        ModelUtils.set_params(self.global_model, self.w_global)
        t_model_update = time.perf_counter() - t_model_update_start

        # 8) Đánh giá accuracy trên test set theo chu kỳ để giảm overhead.
        t_eval_start = time.perf_counter()
        should_eval = (rnd <= 1) or (self.eval_interval <= 1) or (rnd % self.eval_interval == 0)
        if should_eval:
            self.last_accuracy = self._evaluate_accuracy()
            # Cập nhật Early Stopping monitor tại Leader.
            self.early_stopping(self.last_accuracy)
        accuracy = self.last_accuracy
        enable_es = getattr(self.cfg, "enable_early_stopping", False)
        is_converged = self.early_stopping.early_stop if enable_es else False
        t_eval = time.perf_counter() - t_eval_start

        t_total_step = time.perf_counter() - t_step_start
        self.last_timing_stats = {
            "step_total_sec": float(t_total_step),
            "global_diff_sec": float(t_global_diff),
            "local_train_and_grad_sec": float(t_local_train),
            "select_active_sec": float(t_select),
            "update_counters_sec": float(t_counter),
            "build_active_dict_sec": float(t_active_dict),
            "aggregate_sec": float(t_agg),
            "model_update_sec": float(t_model_update),
            "evaluate_sec": float(t_eval),
            "should_evaluate": bool(should_eval),
            "eval_interval": int(self.eval_interval),
            "worker_train_mean_sec": float(np.mean(worker_times)) if worker_times else 0.0,
            "worker_train_max_sec": float(np.max(worker_times)) if worker_times else 0.0,
        }

        stage_times = {
            "global_diff_sec": t_global_diff,
            "local_train_and_grad_sec": t_local_train,
            "select_active_sec": t_select,
            "update_counters_sec": t_counter,
            "build_active_dict_sec": t_active_dict,
            "aggregate_sec": t_agg,
            "model_update_sec": t_model_update,
            "evaluate_sec": t_eval,
        }
        self.last_timing_stats["slowest_stage"] = max(stage_times, key=stage_times.get)

        # 9) Trả về (accuracy, active_indices, global_diff_sq, is_converged).
        return float(accuracy), list(active_indices), float(global_diff_sq), bool(is_converged)

    def _evaluate_accuracy(self) -> float:
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            if hasattr(self, 'test_images') and len(self.test_images) > 0:
                batch_size = 2048
                n_samples = len(self.test_labels)
                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    images = self.test_images[start_idx:end_idx]
                    labels = self.test_labels[start_idx:end_idx]
                    logits = self.global_model(images)
                    preds = torch.argmax(logits, dim=1)
                    correct += int((preds == labels).sum().item())
                    total += int(labels.size(0))
        return (correct / total) if total > 0 else 0.0

    def reset(self):
        self.global_model = SimpleNN().to(device)
        self.w_global = ModelUtils.get_params(self.global_model)
        self.w_prev = None
        self.aggregator = AsyncAggregator(
            num_workers=self.M,
            initial_weights=self.w_global,
            N_m_dict=self.N_m_dict,
            N_total=self.N_total,
        )
        self.controller.lazy_consecutive[:] = 0
        self.last_accuracy = 0.0
        self.early_stopping.reset()
