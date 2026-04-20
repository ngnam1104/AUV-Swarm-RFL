import numpy as np

class CommunicationModel:
    """Section 3.2: Communication Model - Tính suy hao, nhiễu và tốc độ mạng thủy âm."""
    def __init__(self, config):
        self.cfg = config
        
        # 1. Trích xuất tham số vật lý (ưu tiên dùng thẳng đơn vị chuẩn)
        self.f = getattr(config, 'freq_acoustic', 30.0)         # Tần số f (kHz)
        self.s = getattr(config, 's_factor', 0.5)               # Hệ số tàu bè s
        self.w = getattr(config, 'w_speed', 0.0)                # Tốc độ gió w (m/s)
        self.k_spread = getattr(config, 'k_spread', 1.5)        # Hệ số spreading factor k
        self.D = getattr(config, 'distance', 20.0)            # Khoảng cách AUV tới Leader (m)
        
        # Đảm bảo băng thông lấy từ config đã là Hz (10kHz = 10000 Hz)
        # Theo paper, các node chia sẻ băng thông (FDMA).
        self.B_Um_total = getattr(config, 'B_Um', 10000.0)
        self.B_D_total = getattr(config, 'B_D', 10000.0)
        self.M = getattr(config, 'M', 9)
        
        # Băng thông dành cho MỖI AUV (Theo Table 1 và phân tích Figure 3 của paper)
        # B_m^U là 10kHz cho mỗi node, B^D là 10kHz cho kênh quảng bá chung.
        self.B_Um = self.B_Um_total
        self.B_D = self.B_D_total

        # Tiền tính toán channel gain gamma(D, f) vì D và f tĩnh
        (self.channel_gain, self.A_D_f, self.log_a, self.a_f_linear, 
         self.log_N_theta, self.log_N_s, self.log_N_w, self.log_N_th, 
         self.N_theta_linear, self.N_s_linear, self.N_w_linear, self.N_th_linear, 
         self.N_f_total_linear, self.N_f_total_standard) = self._calculate_channel_gain(self.D)

    def _calculate_channel_gain(self, D_meters: float) -> tuple:
        """Eq. 4-10: Tính Attenuation A(D,f) và Noise N(f), sau đó suy ra Gamma."""
        f = self.f
        
        # Eq. 5: Tính hệ số hấp thụ a(f) (đơn vị dB/km)
        term1 = (0.11 * f**2) / (1 + f**2)
        term2 = (44 * f**2) / (4100 + f**2)
        term3 = 2.75e-4 * f**2 + 0.003
        log_a = term1 + term2 + term3 
        
        # Chuyển đổi a(f) từ dB/km sang tuyến tính
        a_f_linear = 10 ** (log_a / 10.0)
        
        # Eq. 4: Độ suy hao A(D, f)
        # SỬA LỖI VẬT LÝ: Spreading loss tính theo mét (D^k), Absorption loss tính theo km (a_f^(D/1000))
        D_km = D_meters / 1000.0
        A_D_f = (D_meters ** self.k_spread) * (a_f_linear ** D_km)
        
        # Eq. 6-9: Tính các thành phần nhiễu (Turbulence, Shipping, Wind, Thermal) bằng dB
        log_N_theta = 17 - 30 * np.log10(f)
        log_N_s = 40 + 20 * (self.s - 0.5) + 26 * np.log10(f) - 60 * np.log10(f + 0.03)
        log_N_w = 50 + 7.5 * (self.w ** 0.5) + 20 * np.log10(f) - 40 * np.log10(f + 0.4)
        log_N_th = -15 + 20 * np.log10(f)
        
        # Chuyển đổi nhiễu từ dB sang tuyến tính
        N_theta_linear = 10 ** (log_N_theta / 10.0)
        N_s_linear = 10 ** (log_N_s / 10.0)
        N_w_linear = 10 ** (log_N_w / 10.0)
        N_th_linear = 10 ** (log_N_th / 10.0)
        
        # Eq. 10: Tổng nhiễu N(f)
        N_f_total_linear = N_theta_linear + N_s_linear + N_w_linear + N_th_linear #μPa2/Hz
        
        # Nhân với 1e-12 để khớp thang đo chuẩn từ microPascal sang W/Hz
        N_f_total_standard = N_f_total_linear * 1e-12

        
        # Eq. 11: SNR chuẩn hóa Gamma(D, f)
        gamma = 1.0 / (A_D_f * N_f_total_standard)
        return (gamma, A_D_f, log_a, a_f_linear, 
                log_N_theta, log_N_s, log_N_w, log_N_th, 
                N_theta_linear, N_s_linear, N_w_linear, N_th_linear, 
                N_f_total_linear, N_f_total_standard)

    def uplink_rate(self, p_m: np.ndarray, active_mask: np.ndarray | None = None) -> np.ndarray:
        """Eq. 12: Tốc độ R_U_m của Follower dùng FDMA (Orthogonal)."""
        # Băng thông B_Um đã được chia cho M ở hàm __init__
        p_m_clipped = np.clip(p_m, 1e-6, None)
        # SỬA LỖI: channel_gain đã bao gồm N(f), nên chỉ cần chia cho Băng thông B_Um
        snr = (p_m_clipped * self.channel_gain) / self.B_Um
        return self.B_Um * np.log2(1.0 + snr)

    def downlink_rate(self, p_L) -> np.ndarray: # Type có thể là float hoặc Numpy array
        """Eq. 13: Tốc độ R_D_m của Leader (Full Vectorized)."""
        # SỬA LỖI VECTOR: Sử dụng thư viện Numpy thay cho Python thuần
        p_L_clipped = np.clip(p_L, 0.0, None)
        sinr = (p_L_clipped * self.channel_gain) / self.B_D
        return self.B_D * np.log2(1.0 + sinr)
    
if __name__ == "__main__":
    import sys
    import os
    # Cấu hình đường dẫn root để có thể thực thi file từ ngoài
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.settings import ACOUSTIC_CFG

    comm_model = CommunicationModel(ACOUSTIC_CFG)
    print("=== Suy hao (Attenuation) ===")
    print(f"A_D_f (Độ suy hao tổng): {comm_model.A_D_f:.2e}")
    print(f"log_a (Suy hao hấp thụ H Hz dB/km): {comm_model.log_a:.2f}")
    print(f"a_f_linear (Suy hao tuyến tính): {comm_model.a_f_linear:.2e}")
    
    print("\n=== Nhiễu môi trường (dB) ===")
    print(f"N_theta (Rối thủy âm): {comm_model.log_N_theta:.2f} dB")
    print(f"N_s (Tàu bè): {comm_model.log_N_s:.2f} dB")
    print(f"N_w (Gió): {comm_model.log_N_w:.2f} dB")
    print(f"N_th (Nhiệt): {comm_model.log_N_th:.2f} dB")
    
    print("\n=== Nhiễu môi trường (Linear) ===")
    print(f"N_theta_linear: {comm_model.N_theta_linear:.2e}")
    print(f"N_s_linear: {comm_model.N_s_linear:.2e}")
    print(f"N_w_linear: {comm_model.N_w_linear:.2e}")
    print(f"N_th_linear: {comm_model.N_th_linear:.2e}")
    print(f"Tổng nhiễu (Linear): {comm_model.N_f_total_linear:.2e}")
    
    print("\n=== Các thông số chuẩn hóa ===")
    print(f"Tổng nhiễu (W/Hz - N_f_total_standard): {comm_model.N_f_total_standard:.2e}")
    print(f"Gain kênh truyền (Gamma): {comm_model.channel_gain:.2e}")
    
    print("\n=== Tốc độ mạng ===")
    # Test tốc độ uplink với một mảng công suất mẫu
    p_max = 0.2

    p_m_test = np.random.uniform(
        low=0.01,
        high=p_max,
        size=5
    )    
    uplink_rates = comm_model.uplink_rate(p_m_test)
    print("Uplink Rates (R_U_m):", uplink_rates)
    
    # Test tốc độ downlink với một công suất mẫu
    p_L_test = 0.1  # Công suất Leader (W)
    downlink_rate = comm_model.downlink_rate(p_L_test)
    print("Downlink Rate (R_D_m):", downlink_rate)