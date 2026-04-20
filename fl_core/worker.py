import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fl_core.dataset import DatasetSplit
from fl_core.models import ModelUtils, SimpleNN


class LocalWorker:
    """Đại diện cho 1 Follower AUV thực hiện train cục bộ."""
    def __init__(self, worker_id: int, dataset, idxs: list):
        self.worker_id = worker_id
        self.idxs = idxs
        
        # Tiền xử lý dữ liệu và preload lên memory (CPU tensor) để không bị I/O bottleneck khi tạo DatasetSplit/DataLoader liên tục
        local_dataset = DatasetSplit(dataset, self.idxs)
        self.all_images = []
        self.all_labels = []
        
        # Load tất cả 1 lần và giải phóng I/O
        temp_loader = DataLoader(local_dataset, batch_size=128, shuffle=False, num_workers=0)
        for imgs, lbls in temp_loader:
            self.all_images.append(imgs)
            self.all_labels.append(lbls)
            
        if self.all_images:
            self.all_images = torch.cat(self.all_images, dim=0)
            self.all_labels = torch.cat(self.all_labels, dim=0)
            
        # Model dùng chung để tránh overhead cấp phát mỗi round
        self.model = SimpleNN()
        self.criterion = nn.CrossEntropyLoss()

    def local_train(self, global_params: dict, num_epochs: int, lr: float, batch_size: int) -> tuple[dict, float]:
        """Thực hiện SGD và trả về (w_local, loss)."""
        # Đổ trọng số từ global (w_t-1) xuống thiết bị local
        ModelUtils.set_params(self.model, global_params)
        self.model.train()
        
        # Khởi tạo optimizer cục bộ cho vòng này
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        total_loss = 0.0
        total_steps = 0

        # Mô phỏng DataLoader bằng tensor slicing (cực kỳ nhanh)
        n_samples = len(self.all_labels)
        indices = torch.randperm(n_samples)
        
        for _ in range(num_epochs):
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_idxs = indices[start_idx:end_idx]
                
                images = self.all_images[batch_idxs]
                labels = self.all_labels[batch_idxs]
                
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_steps += 1

        avg_loss = float(total_loss / total_steps) if total_steps > 0 else 0.0
        w_local = ModelUtils.get_params(self.model)
        return w_local, avg_loss
