import copy
import os
import time

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
        torch.set_num_threads(cpu_threads)

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

        # 1) Tính ||w(t-1) - w(t-2)||^2. Nếu chưa đủ lịch sử (t < 2) thì đặt 0.0.
        t_global_diff_start = time.perf_counter()
        if self.w_prev is None:
            global_diff_sq = 0.0
        else:
            diffs = [(self.w_global[k] - self.w_prev[k]).pow(2).sum() for k in self.w_global.keys()]
            global_diff_sq = float(sum(diffs).item())
        t_global_diff = time.perf_counter() - t_global_diff_start

        # 2) Local train cho TẤT CẢ worker, sau đó tính ||N_m * grad_m^{t-1}||^2.
        t_local_train_start = time.perf_counter()
        local_weights = {}
        local_grad_sq_norms = {}
        worker_times = []

        for worker in self.workers:
            t_worker_start = time.perf_counter()
            w_local, _ = worker.local_train(
                global_params=self.w_global,
                num_epochs=self.num_epochs,
                lr=self.lr,
                batch_size=self.batch_size,
            )
            local_weights[worker.worker_id] = w_local

            N_m = self.N_m_dict[worker.worker_id]
            factor = N_m / self.lr
            grad_sq_tensor = sum(((w_local[k] - self.w_global[k]) * factor).pow(2).sum() for k in self.w_global.keys())
            local_grad_sq_norms[worker.worker_id] = float(grad_sq_tensor.item())
            worker_times.append(time.perf_counter() - t_worker_start)

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

        # 7) Lưu lịch sử global model cho vòng sau.
        t_model_update_start = time.perf_counter()
        self.w_prev = copy.deepcopy(self.w_global)
        self.w_global = copy.deepcopy(new_w_global)
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
        is_converged = self.early_stopping.early_stop
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
