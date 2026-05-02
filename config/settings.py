import os

# Lấy thư mục gốc của dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Định nghĩa đường dẫn trỏ tới thư mục data/
DATA_DIR = os.path.join(BASE_DIR, '.data')

from dataclasses import dataclass

@dataclass
class AcousticConfig:
    freq_acoustic: float = 30.0         # Tần số âm học f (kHz)
    s_factor: float = 0.5               # Hệ số tàu bè s (Dùng trong mô cảnh nhiễu)
    distance: float = 20.0          # Khoảng cách từ Follower đến Leader D (m) -> Quá ngắn
    B_Um: float = 10000.0                # Băng thông Uplink B_m^U (Hz) -> Quá cao
    B_D: float = 10000.0                # Băng thông Downlink B^D (Hz) ->  Quá cao
    w_speed: float = 0.0                # Tốc độ gió (m/s) - Mặc định 0.0 theo giả định biển lặng
    k_spread: float = 1.5               # Hệ số phân tán/lan truyền k (Eq.4 spreading factor, practical = 1.5, paper Table 1 chỉ liệt kê k_cap)

@dataclass
class HardwareConfig:
    k_cap: float = 1.25e-26             # Hệ số năng lượng CMOS k
    delta_pu: float = 3.0               # Bậc phụ thuộc năng lượng CPU (delta hoặc sigma)
    p_max: float = 0.2                  # Công suất phát tối đa P_max (W)
    f_max: float = 0.4e9                # Tần số CPU tối đa f_max (Hz)
    f_min: float = 0.2e9                # Tần số CPU tối thiểu f_min (Hz)
    E_m_thd: float = 0.07               # Ngưỡng công suất trung bình phạt Follower (W)
    E_L_thd: float = 0.8                # Ngưỡng công suất trung bình phạt Leader (W)
    phi: float = 1.0                    # Trọng số cân bằng Năng lượng Tính toán
    chi: float = 1.0                    # Trọng số cân bằng Năng lượng Truyền thông

@dataclass
class FLConfig:
    M: int = 9                          # Số lượng AUV Follower (M = 9, 16, 25, 36, 49)
    num_epochs: int = 1                 # Số epoch local train mỗi vòng FL
    lr: float = 0.01                    # Learning Rate của thuật toán học máy (gamma)
    batch_size: int = 256               # Batch size huấn luyện
    dataset_size: int = 42000           # Kích thước tập huấn luyện MNIST
    N_m: int = 4224                     # Số mẫu dữ liệu mỗi AUV (N_m)
    c_m: float = 10000.0                # Số chu kỳ CPU / 1 mẫu (c_m)
    c_0: float = 50.0                   # Độ phức tạp tổng hợp mô hình của Leader (c_0)
    w_size_bits: float = 1594 * 64.0    # Kích thước trọng số mô hình |w_m|, |w| (bits)
    packet_overhead: float = 64.0       # [bits] Kích thước dữ liệu vị trí/tốc độ (biến psi trong bài báo không có giá trị cụ thể, mình giả định 64 bits cho 2 float)
    max_fl_rounds: int = 1000           # Số vòng FL tối đa (Max rounds / Episodes)
    max_lazy_rounds: int = 5            # Ngưỡng bắt buộc Active (T = 5 vòng)
    c_L_prime: float = 50.0             # Eq. 24: Độ phức tạp cập nhật global model c'_L (paper không cho giá trị, đặt = c_0)
    T_prime: float = 1.0                # Eq. 27: Thời gian chờ nếu không có node nào upload (s)

@dataclass
class RLConfig:
    penalty: float = 10.0               # Penalty vi phạm ràng buộc năng lượng Eq. 39-40 (constrained RL)
    ppo_lr: float = 3e-4                # Learning rate của PPO
    ppo_n_steps: int = 1000             # Số step thu thập trước khi update PPO
    ppo_batch_size: int = 250           # Batch size của PPO
    ppo_n_epochs: int = 10              # Số epoch update PPO
    ppo_gamma: float = 0.99             # Eq. 44: Discount factor ξ (paper không cho giá trị cụ thể)

# Khởi tạo sẵn một object config mặc định để các file khác import
ACOUSTIC_CFG = AcousticConfig()
HW_CFG = HardwareConfig()
FL_CFG = FLConfig()
RL_CFG = RLConfig()
