### 1. Hiện thực hóa các phương trình Vật lý (`env/`)
Phần này yêu cầu bạn chuyển đổi các công thức từ Section 3.2, 3.4 và 3.5 của bài báo[cite: 288, 290, 293]:
* **`communication.py`**: Viết hàm tính toán suy hao thủy âm $A(D,f)$ theo công thức (4, 5) [cite: 352, 355] và mô hình nhiễu tổng hợp $N(f)$ từ các nguồn nhiễu loạn, tàu bè, sóng biển và nhiệt theo công thức (6-10)[cite: 359, 367]. Từ đó tính SNR chuẩn hóa $\gamma(D,f)$ (11) và tốc độ truyền dữ liệu Uplink/Downlink (12, 13)[cite: 371, 373, 378].
* **`latency.py`**: Tính toán 5 thành phần thời gian trễ: tính toán cục bộ $T_{LC}$ (21), tải lên $T_{LU}$ (22), tổng hợp $T_{GA}$ (23), cập nhật $T_{GU}$ (24) và tải xuống $T_{GD}$ (25)[cite: 418, 426, 429, 435, 440]. Cuối cùng là tính tổng trễ $T(t)$ cho cả trường hợp thông thường và trường hợp cực đoan (26, 27)[cite: 445, 450].
* **`energy.py`**: Tính năng lượng tiêu thụ cho tính toán ($E_{Cp}$) và truyền thông ($E_{C}$) của cả Follower và Leader AUV theo các công thức (28-31) [cite: 458, 461, 466, 470], sau đó tổng hợp thành $E(t)$ (32)[cite: 471].

### 2. Hiện thực hóa lõi Federated Learning (`fl_core/`)
Phần này đảm nhận vai trò tính toán AI và điều khiển nút lười (Section 3.1 & 3.3)[cite: 334, 381]:
* **`models.py` & `worker.py`**: Định nghĩa mạng MLP 3 lớp (200-200-10) [cite: 546, 547] và hàm `local_train` thực hiện SGD trên thiết bị biên[cite: 336].
* **`control.py` (Lazy Node Logic)**: Đây là phần quan trọng. Bạn cần viết logic kiểm tra điều kiện nút lười theo bất đẳng thức (18) dựa trên biến $\beta$ do RL cung cấp[cite: 398, 404]. Đồng thời, hiện thực hóa cơ chế bắt buộc tham gia nếu AUV đã "ngủ" quá 5 vòng liên tiếp (20)[cite: 409, 414].
* **`aggregator.py`**: Viết cơ chế tổng hợp bất đồng bộ sử dụng bộ nhớ đệm (cache) để lưu trữ trọng số cũ của các nút lười, giúp mô hình toàn cục luôn được cập nhật đầy đủ từ $M$ nút[cite: 176, 177].

### 3. Thiết lập bộ máy Reward và Môi trường Gym (`env/auv_env.py`)
Đây là nơi kết nối giữa Vật lý và RL (Section 3.6 & 4.1)[cite: 472, 498]:
* **`reward.py`**: Chuyển đổi hàm mục tiêu tối thiểu hóa Chi phí ($Cost = T+E$) sang hàm tối đa hóa phần thưởng (Reward) cho RL[cite: 478, 513]. Bạn cần bê nguyên phần "Reward Reshaping" từ notebook vào đây, bao gồm việc sử dụng hàm `tanh` để nén các giá trị Reward và các hình phạt (Penalty) khi AUV hết pin hoặc vi phạm ngưỡng công suất[cite: 145].
* **`auv_env.py`**: Hoàn thiện các phương thức `reset()` (khởi tạo lại pin, trạng thái FL) [cite: 185] và `step()` (nhận Action từ PPO, gọi các module FL và Vật lý, trả về State tiếp theo)[cite: 500, 502, 506].

### 4. Tích hợp và Chạy thực nghiệm (`rl_agent/` & `Scripts`)
* **`ppo_trainer.py`**: Viết hàm khởi tạo mô hình PPO từ Stable-Baselines3, cấu hình các tham số như `learning_rate`, `n_steps`, `batch_size` và kiến trúc mạng MLP[cite: 521, 536].
* **`main_train.py`**: Script khởi tạo `FLSimulator` và `AUVSwarmEnv`, sau đó kích hoạt quá trình huấn luyện `model.learn()`[cite: 542].
* **`main_eval.py`**: Viết code để tải model đã lưu, chạy kiểm thử (inference) và vẽ các biểu đồ so sánh Hiệu năng (Cost, Accuracy, Communication Times) như trong Section 5 của bài báo[cite: 556, 598, 722].

**Tóm lại:** Bạn đã có "bản vẽ kiến trúc", công việc còn lại là bóc tách logic từ file `.ipynb` cũ và sắp xếp chúng vào đúng các "ngăn kéo" (classes/methods) trong cấu trúc OOP này. Bạn nên bắt đầu từ module **`communication.py`** vì nó là nền tảng để tính toán tất cả các giá trị về sau.

###  Giai đoạn 1: Khảo sát độ nhạy của $\beta$ (Figures 1, 2, 3)
*Mục tiêu:* Chứng minh $\beta$ là biến số cốt lõi ảnh hưởng đến sự đánh đổi giữa Tài nguyên và Độ chính xác.
* **TODO 1: Tạo script `scripts/eval_beta_sensitivity.py`**
  1. Cố định các thông số vật lý ($p_m, f_m, p_L, f_L$) ở một mức cố định (ví dụ: trung bình).
  2. Tạo vòng lặp biến $\beta$ chạy từ $0.1$ đến $0.9$ (step 0.1).
  3. Tại mỗi $\beta$, chạy `FLSimulator` đủ 1000 vòng FL (không dùng RL).
  4. Lặp lại bước 2-3 với các mức số lượng AUV $M \in \{9, 16, 25\}$.
  5. Thu thập dữ liệu và dùng `matplotlib` xuất ra 3 đồ thị:
     * **Figure 1:** Trục X là $\beta$, Trục Y là **Communication times** (Tổng `active_nodes` sau 1000 vòng).
     * **Figure 2:** Trục X là $\beta$, Trục Y là **Accuracy** (Độ chính xác ở vòng 1000).
     * **Figure 3:** Trục X là $\beta$, Trục Y là **Time consumption** (Tổng $T_{total}$ sau 1000 vòng).


###  Giai đoạn 2: Lập trình 5 Schemes cốt lõi
*Mục tiêu:* Số hóa định nghĩa 5 Schemes của bài báo thành các logic chạy môi trường.
* **TODO 2: Viết class `SchemeEvaluator` trong `scripts/eval_schemes.py`**
  Định nghĩa hàm chạy 1000 vòng FL cho từng Scheme:
  * **Scheme 1 (Proposed):** Load model PPO đã train (`model.predict`). Cả $\beta$ và các biến vật lý đều được tối ưu động.
  * **Scheme 2 (Dynamic $\beta$ only):** Cố định các biến vật lý ($f_m = max, p_m = min$). Chỉ cho phép PPO (hoặc một hàm heuristic) điều khiển $\beta$ biến thiên qua từng vòng.
  * **Scheme 3 (Fixed $\beta$):** Ép cứng hành động $\beta = 0.5$ trong toàn bộ 1000 vòng. Vật lý cố định.
  * **Scheme 4 (LAG - Lazy Aggregation Gradient):** Kế thừa `LazyNodeController`, ghi đè hàm `select_active_nodes`. Thay vì dùng Eq. 19, sử dụng điều kiện kiểm duyệt của thuật toán LAG truyền thống (Dựa trên chênh lệch gradient tuyệt đối lớn hơn hằng số).
  * **Scheme 5 (Traditional Async FL):** Ép $\beta = 1.0$ (hoặc threshold = -inf). **TẤT CẢ** các AUV đều Active trong mọi vòng. Không có Nút lười.


###  Giai đoạn 3: So sánh 5 Schemes theo Số lượng AUV (Figures 4, 5, 6)
*Mục tiêu:* Đánh giá khả năng mở rộng (Scalability) của mạng lưới.
* **TODO 3: Chạy thực nghiệm `scripts/run_fig_4_5_6.py`**
  1. Tạo vòng lặp số lượng AUV $M \in \{9, 16, 25, 36, 49\}$.
  2. Tại mỗi $M$, lần lượt gọi `SchemeEvaluator` để chạy cả 5 Schemes (mỗi Scheme 1000 vòng).
  3. Lưu lại kết quả cuối cùng của mỗi Scheme và vẽ 3 đồ thị (so sánh 5 đường cong trên mỗi hình):
     * **Figure 4:** Trục X là $M$, Trục Y là **Communication times** (Tổng số lần giao tiếp).
     * **Figure 5:** Trục X là $M$, Trục Y là **Accuracy** (Độ chính xác).
     * **Figure 6:** Trục X là $M$, Trục Y là **Energy consumption** (Tổng $E_{total}$). (Kỳ vọng: Đường của Scheme 5 sẽ dốc ngược lên cao nhất, Scheme 1 sẽ nằm dưới cùng).


###  Giai đoạn 4: So sánh Thuật toán RL (Figure 7)
*Mục tiêu:* Chứng minh PPO là thuật toán tìm kiếm Policy tốt nhất cho môi trường này so với các thuật toán khác. Cố định $M$ (ví dụ $M=9$), sử dụng cơ chế Scheme 1.
* **TODO 4: Tạo script huấn luyện Baseline `scripts/train_baselines.py`**
  1. **PPO (Proposed):** Chạy `main_train.py` để train 1000 Episodes. Lưu mảng `accumulated_cost`.
  2. **DDPG:** Import `DDPG` từ `stable_baselines3`. Setup môi trường y hệt PPO, train 1000 Episodes. Lưu mảng `accumulated_cost`.
  3. **Greedy Algorithm:** Viết vòng lặp 1000 Episodes không cần mạng Neural. Tại mỗi step, lấy action sinh ra giá trị $(T(t) + E(t))$ thấp nhất ngay lập tức (bỏ qua tính dài hạn).
  4. **Random:** Gọi `env.action_space.sample()` cho mỗi step.
* **TODO 5: Vẽ đồ thị hội tụ `scripts/plot_fig_7.py`**
  1. Đọc dữ liệu `accumulated_cost` (chi phí tích lũy Eq. 33) của cả 4 thuật toán trên.
  2. Dùng hàm `scipy.ndimage.gaussian_filter1d` hoặc smoothing để làm mượt các đường cong.
  3. Trục X là **Episodes (0 - 1000)**, Trục Y là **Accumulated Cost**.
  4. Xuất đồ thị Figure 7 (Kỳ vọng: Đường PPO sẽ hội tụ xuống mức thấp nhất và ổn định nhất).

