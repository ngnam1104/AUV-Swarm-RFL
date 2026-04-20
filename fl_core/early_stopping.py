"""
Early Stopping for Federated Learning Global Model.

Standard PyTorch-style early stopping mechanism applied at the AUV Leader.
Monitors global test accuracy and signals convergence when accuracy plateaus,
allowing the FL process to terminate before the 1000-round maximum.
"""


class EarlyStopping:
    """
    Theo dõi accuracy toàn cục của Leader để phát hiện hội tụ sớm.

    Nếu accuracy không cải thiện thêm ít nhất `min_delta` sau `patience`
    lần đánh giá liên tiếp, đặt cờ `early_stop = True`.

    Parameters
    ----------
    patience : int
        Số lần đánh giá liên tiếp không cải thiện trước khi kích hoạt dừng sớm.
    min_delta : float
        Ngưỡng cải thiện tối thiểu để được tính là "có tiến bộ".
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        # State variables
        self.best_score: float | None = None
        self.counter: int = 0
        self.early_stop: bool = False

    def __call__(self, accuracy: float) -> None:
        """
        Cập nhật trạng thái early stopping dựa trên accuracy mới.

        Parameters
        ----------
        accuracy : float
            Accuracy toàn cục hiện tại (đánh giá trên test set tại Leader).
        """
        score = float(accuracy)

        if self.best_score is None:
            # Lần đánh giá đầu tiên — khởi tạo baseline.
            self.best_score = score
            return

        if score > self.best_score + self.min_delta:
            # Có cải thiện đáng kể → reset bộ đếm.
            self.best_score = score
            self.counter = 0
        else:
            # Không cải thiện → tăng bộ đếm.
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def reset(self) -> None:
        """Reset toàn bộ trạng thái cho lần chạy mới."""
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta}, "
            f"best_score={self.best_score}, counter={self.counter}, "
            f"early_stop={self.early_stop})"
        )
