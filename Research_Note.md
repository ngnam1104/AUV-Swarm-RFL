
# Current Paper: 2022 - Efficient Asynchronous Federated Learning for AUV Swarm

## Problem Statement:

### 1. Tóm tắt Cấu trúc Mạng (Network Topology)
Hệ thống vận hành dựa trên một bầy AUV di chuyển theo đội hình cố định, bao gồm một AUV dẫn đường (Leader $L$) và $M$ AUV theo sau (Followers). Chức năng trong mạng được phân chia như sau: 
* Các AUV theo sau đóng vai trò là các nút cục bộ, sử dụng đường truyền lên (uplink) để gửi các tham số mô hình đã huấn luyện cho AUV dẫn đường. 
* AUV dẫn đường đóng vai trò là máy chủ trung tâm, có nhiệm vụ tổng hợp các tham số này và sử dụng đường truyền xuống (downlink) để phát sóng (broadcast) mô hình toàn cục cập nhật lại cho toàn bầy.
* Đặc thù truyền thông tin không đáng tin cậy và có độ trễ lớn dưới nước rất dễ gây ra hiệu ứng tụt hậu (straggler effect) nếu sử dụng các phương pháp học máy đồng bộ truyền thống.

### 2. Đầu vào và Đầu ra (Inputs & Outputs)
Quá trình tương tác dữ liệu và mô hình được xác định rõ ràng qua từng vòng lặp:
* **Đầu vào:** Dữ liệu đáy biển ($x$) được thu thập độc lập bởi các cảm biến của từng AUV theo sau cùng với nhãn tương ứng ($y$). Tập dữ liệu cục bộ này được sử dụng để huấn luyện và tính toán tham số mô hình cục bộ $w_{m}$. Ngoài ra, tốc độ và thông tin vị trí của mỗi AUV cũng là đầu vào cần thiết được tải lên để duy trì đội hình bay.
* **Đầu ra:** Một mô hình học máy toàn cục mạnh mẽ được tổng hợp (với tham số $w$) nhằm khai phá các thông tin có giá trị ẩn sau dữ liệu thô, phục vụ cho các nhiệm vụ phức tạp như nhận dạng mục tiêu, lập kế hoạch quỹ đạo hoặc mô hình hóa đặc điểm thủy văn.

### 3. Bài toán Tối ưu hóa (Optimization Formulation)
Vì năng lượng của AUV có hạn và giao tiếp dưới nước cực kỳ đắt đỏ, mục tiêu cốt lõi của bài toán (ký hiệu là P1) là **tối thiểu hóa tổng chi phí (Cost)**, được định nghĩa là tổng trọng số của độ trễ $T(t)$ và mức tiêu thụ năng lượng $E(t)$ trong suốt tổng số vòng lặp $N_{t}$. Công thức tối ưu hóa được thể hiện qua $Cost = \sum_{t=1}^{N_{t}}(T(t) + E(t))$.

Để đạt được mục tiêu này, hệ thống cần tối ưu hóa liên hợp 5 biến quyết định dưới các điều kiện ràng buộc khắt khe:

| Biến Tối Ưu | Ký Hiệu | Ý Nghĩa và Ràng Buộc (Constraints) |
| :--- | :--- | :--- |
| Công suất AUV theo sau | $p_{m}$ | Giới hạn công suất truyền tín hiệu của nút cục bộ, $0 \le p_{m} \le p_{max}$. |
| Tần số CPU theo sau | $f_{m}$ | Khả năng tính toán của nút cục bộ, $f_{min} \le f_{m} \le f_{max}$. |
| Công suất Leader | $p_{L}$ | Giới hạn công suất truyền tín hiệu của máy chủ, $0 \le p_{L} \le p_{max}$. |
| Tần số CPU Leader | $f_{L}$ | Khả năng tính toán của máy chủ tổng hợp, $f_{min} \le f_{L} \le f_{max}$. |
| Tỷ lệ nút "lười" | $\beta$ | Tỷ lệ các AUV bỏ qua vòng truyền tải (lazy nodes) để tiết kiệm giao tiếp, $0 \le \beta \le 1$. |

Bên cạnh giới hạn về phần cứng, bài toán còn yêu cầu tính khả thi về mặt năng lượng duy trì hoạt động:
* Tổng năng lượng tiêu thụ cho quá trình tính toán và giao tiếp của mỗi AUV theo sau không được vượt quá ngưỡng năng lượng cho phép $E_{m}^{thd}$.
* Tổng năng lượng tiêu thụ tương ứng của AUV dẫn đường cũng bị giới hạn và không được vượt quá ngưỡng $E_{L}^{thd}$.

## Methodology:

### 1. Federated Learning

**Biểu diễn toán học:**
* **Hàm mất mát cục bộ (Local Loss Function):** Trung bình các hàm mất mát trên bộ dữ liệu $N_{m}$ của từng AUV theo sau $m$:
    $$F_{m}(w_{m})=\frac{1}{N_{m}}\sum_{i=1}^{N_{m}}f(w_{m};x_{m,i}y_{m,i})$$
* **Hàm mất mát toàn cục (Global Loss Function):** Được tính bằng trung bình có trọng số của các hàm mất mát cục bộ:
    $$F(w)\triangleq\sum_{m=1}^{M}\frac{N_{m}F_{m}(w)}{N}$$
* **Mục tiêu của FL:** Tìm ra tham số mô hình $w$ tối ưu nhất giúp cực tiểu hóa hàm mất mát toàn cục:
    $$w = \arg\min F(w)$$
* **Điều kiện bỏ qua giao tiếp (Cơ chế nén Gradient):** Ở một vòng lặp $t$, AUV $m$ tự kiểm tra điều kiện dưới đây. Nếu thỏa mãn, nó sẽ trở thành "nút lười" và không gửi tham số lên để tiết kiệm tài nguyên:
    $$||N_{m}\nabla_{m}^{t-1}||^{2}\le\frac{N^{2}}{\gamma^{2}M^{2}\beta}||w(t-1)-w(t-2)||^{2}$$

### 2. Reinforcement Learning

**Biểu diễn toán học (Mô hình hóa MDP và PPO2):**
* **Không gian Trạng thái (State Space):** Trạng thái mạng $s(t)$ tại khe thời gian $t$ là tập hợp các quyết định tài nguyên từ bước thời gian trước đó $(t-1)$:
    $$s(t)=(p_{m}(t-1), f_{m}(t-1), p_{L}(t-1), f_{L}(t-1), \beta(t-1))$$
* **Không gian Hành động (Action Space):** Dựa vào trạng thái, tác tử xuất ra các thông số tối ưu cho vòng hiện tại $t$:
    $$a(t) = (p_{m}(t), f_{m}(t), p_{L}(t), f_{L}(t), \beta(t))$$
* **Hàm Phần thưởng (Reward Function):** Vì mục tiêu là tối thiểu hóa tổng của độ trễ $T(t)$ và năng lượng $E(t)$, phần thưởng được định nghĩa là giá trị âm của tổng chi phí này:
    $$r_{t}(s_{t},a_{t})=-(T(t)+E(t))$$
* **Hàm Lợi thế (Advantage Function):** Dùng để đánh giá hành động có tốt hơn mức trung bình hay không, giúp PPO2 cập nhật chính sách mà không làm giảm đột ngột hiệu suất:
    $$\hat{A}_{n}=R_{n}(T^{\prime})-V_{\phi}(s_{n})$$
* **Cập nhật Mạng Giá trị (Value Network Update trong PPO2):** Tìm tham số $\phi$ tối ưu để dự đoán giá trị trả về:
    $$\phi_{k+1}=\arg\min_{\phi}\frac{1}{|\mathcal{D}_{k}|T}\sum_{\tau\in\mathcal{D}_{k}}\sum_{t=0}^{T}(V_{\phi}(s_{t})-\hat{R}_{t})^{2}$$

## Results:

* **Cơ chế nén Gradient ($\beta$):** Trục X là $\beta$, Trục Y là số lần giao tiếp/độ chính xác/thời gian. Kết quả: Giảm $\beta$ giúp giảm mạnh số lần giao tiếp, độ chính xác giảm không đáng kể. Tuy nhiên, tăng $\beta$ lại giúp giảm tổng thời gian do mô hình hội tụ nhanh hơn khi được giao tiếp nhiều.
* **So sánh tổng thể (Scheme 1-5):** Trục X là số lượng AUV, Trục Y là số lần giao tiếp/độ chính xác/chi phí. Kết quả: Phương pháp đề xuất (Scheme 1) tối ưu nhất, giảm 710 lần giao tiếp (chỉ còn 21% so với FL truyền thống) và độ chính xác chỉ suy giảm 0.03%. Scheme 1 cũng tiêu thụ tổng chi phí (Cost) thấp nhất.
* **Mô hình điều khiển:** Trục X là số lượng AUV, Trục Y là độ chính xác. Kết quả: Thiết kế đề xuất mang lại độ chính xác cao hơn so với việc không dùng cơ chế kiểm soát.
* **Thuật toán tối ưu hóa:** Trục X là số vòng lặp (Episode), Trục Y là chi phí. PPO2 được đặt lên bàn cân với AC, DDPG, GA và PSO. Kết quả: PPO2 ưu việt hơn khi đạt mức chi phí thấp nhất trong cùng một số vòng lặp huấn luyện.

## Research Gaps:

### 1. Kênh truyền lý tưởng và Bỏ qua tỷ lệ mất gói (Packet Loss)

* **Lỗ hổng nghiên cứu:** Bài báo mô phỏng AUV di chuyển với khoảng cách cố định 20m và tính toán thông lượng lý thuyết thông qua công thức Shannon: $R_{m}^{U}=B_{m}^{U}\log_{2}(1+\frac{p_{m}\gamma(D,f)}{B_{m}^{U}})$. Tuy nhiên, môi trường âm thanh dưới nước thực tế chịu ảnh hưởng nặng nề bởi fading đa đường và hiệu ứng Doppler, khiến tỷ lệ rớt gói (Packet Loss Rate - $PLR$) rất cao, làm sai lệch kết quả mô phỏng.
* **Chứng minh toán học:** Độ trễ truyền tải thực tế khi có cơ chế truyền lại (ARQ) phải được biểu diễn theo phân phối hình học:
    $$\mathbb{E}[T_{m}^{LU}] = \frac{|w_{m}|}{R_{m}^{U} \cdot (1 - PLR)}$$
    Khi môi trường xấu đi, $PLR \to 1$, khiến $\mathbb{E}[T_{m}^{LU}] \to \infty$. Việc bài báo giả định $PLR = 0$ đã làm sai lệch hoàn toàn hàm mục tiêu tối ưu tổng độ trễ $T(t)$.
* **Đề xuất giải pháp:** Chuyển đổi từ mạng hình sao (Star Topology) sang **Học Liên kết Phân cụm (Clustered FL)**. Thay vì kết nối trực tiếp, các AUV ở gần nhau sẽ tự tạo thành các cụm (Cluster) và tiến hành tổng hợp cục bộ trước. Cluster Head sau đó mới truyền tham số lên Leader, giúp giảm suy hao đường truyền đa chặng và hạn chế tối đa rớt mạng.

### 2. Mô phỏng Năng lượng không toàn vẹn

* **Lỗ hổng nghiên cứu:** Mô hình năng lượng trong bài viết chỉ đưa ra ràng buộc cho mỗi vòng lặp đơn lẻ là $E_{m}^{Cp}+E_{m}^{C} \le E_{m}^{thd}$. Việc thiếu biến trạng thái về dung lượng pin tổng khiến hệ thống không thể dự báo rủi ro cạn kiệt năng lượng giữa chừng (dead nodes).
* **Chứng minh toán học:** Cần bổ sung biến dung lượng pin $B_m(t)$ cho AUV $m$ tại vòng $t$. Trạng thái năng lượng cập nhật theo:
    $$B_m(t) = B_m(t-1) - E_m(t)$$
    Hệ thống bắt buộc phải thỏa mãn ràng buộc sinh tồn: $B_m(N_t) > 0$. Sự thiếu vắng phương trình ràng buộc toàn cục $\sum_{t=1}^{N_t} E_m(t) \le B_{max}$ khiến thuật toán PPO2 chỉ tối ưu cận thị (myopic optimization) mà không thể giải quyết bài toán phân bổ năng lượng đường dài.
* **Đề xuất giải pháp:** Tích hợp **Tối ưu hóa Nhận thức Pin (Battery-Aware Optimization)**. Cần đưa biến trạng thái $B_m(t)$ vào mô hình Markov (MDP) và thiết kế lại hàm phần thưởng $r_t$. Thuật toán RL sẽ phạt cực nặng các hành động đẩy AUV vào rủi ro sập nguồn, đồng thời ưu tiên điều hướng tải trọng tính toán sang các AUV còn nhiều pin.

### 3. Nút thắt cổ chai tại Leader AUV (Bottleneck)

* **Lỗ hổng nghiên cứu:** Theo thiết kế, máy chủ (Leader L) phải gánh toàn bộ tải mạng: nhận, tổng hợp tham số từ $M$ follower với thời gian $T_{L}^{GA}(t)=\frac{c_{0}\sum_{m=1}^{M}|w_{m}|}{f_{L}(t)}$ và phát sóng lại. Khối lượng công việc khổng lồ này rất dễ gây tắc nghẽn nhưng chưa được phân tích trong thực nghiệm.
* **Chứng minh toán học:** Dữ liệu đẩy lên mạng bất đồng bộ tuân theo quá trình ngẫu nhiên Poisson (tốc độ đến $\lambda$). Nếu tốc độ xử lý mạng của Leader là $\mu$, theo lý thuyết hàng đợi M/M/1, thời gian chờ trung bình tại nút Leader sẽ là:
    $$W_L = \frac{1}{\mu - \lambda}$$
    Khi $\lambda \to \mu$, thời gian chờ $W_L \to \infty$. Việc bài báo định nghĩa hàm tổng độ trễ $T(t)$ mà không cộng thêm Queueing Delay là một thiếu sót lớn.
* **Đề xuất giải pháp:** Áp dụng **Phân chia Tác vụ Tính toán (Split Computing)** kết hợp với kiến trúc phân cấp. Bằng cách giảm tải khối lượng tính toán từ Leader xuống các cụm con (Cluster Heads), hệ thống sẽ triệt tiêu được điểm nghẽn cổ chai (bottleneck) và đảm bảo thời gian chờ $W_L$ luôn ở mức an toàn.

### 4. Giả định Dữ liệu Đồng nhất (IID) phi thực tế

* **Lỗ hổng nghiên cứu:** Phần thực nghiệm sử dụng bộ dữ liệu chuẩn MNIST và giả định hàm mất mát là trung bình trọng số $F(w) = \frac{1}{N}\sum_{m=1}^{M}N_mF_m(w)$. Tuy nhiên, dữ liệu thu thập từ các AUV ở các tọa độ khác nhau luôn đa phương thức và không đồng nhất (Non-IID).
* **Chứng minh toán học:** Với dữ liệu Non-IID, mức độ sai lệch gradient được chặn bởi tham số $\delta$:
    $$||\nabla F_m(w) - \nabla F(w)|| \le \delta$$
    Trong môi trường thực tế, $\delta$ rất lớn. Thuật toán trong bài sử dụng cơ chế "nút lười". Nếu một nút chứa nhãn dữ liệu hiếm vô tình bị bỏ qua, sự phân kỳ trọng số (Weight Divergence) sẽ phá vỡ giới hạn xấp xỉ $w(t)-w(t-1) \approx w(t-1)-w(t-2)$, làm mô hình không thể hội tụ.
* **Đề xuất giải pháp:** Sử dụng tập dữ liệu hải dương học thực tế (sonar, ảnh quang học dưới nước). Thay thế Federated Averaging bằng thuật toán **FedProx**. FedProx thêm thành phần điều chuẩn $\frac{\mu}{2}||w_m - w_t||^2$ vào quá trình huấn luyện cục bộ để ép các tham số không phân kỳ quá xa, giải quyết triệt để tính chất Non-IID.

### 5. Giới hạn Cơ chế Giảm tải (Thiếu Quantization & Knowledge Distillation)

* **Lỗ hổng nghiên cứu:** Hệ thống chỉ dừng ở việc dùng hàm kiểm tra $||N_{m}\nabla_{m}^{t-1}||^{2}$ để quyết định gửi hay đóng băng nguyên vẹn kích thước tham số $|w_m|$. Bài báo đã bỏ qua tiềm năng khổng lồ của các kỹ thuật nén mô hình hiện đại.
* **Chứng minh toán học:** * *Với Lượng tử hóa (Quantization):* Biến $w_m$ có thể ánh xạ vào không gian ít bit hơn qua toán tử $Q(w_{m}^i) = ||w_m|| \cdot \text{sgn}(w_{m}^i) \cdot \xi_k$. Kích thước gói tin giảm còn $|Q(w_m)| = \frac{|w_m|}{q}$ với ràng buộc phương sai $\mathbb{E}[||Q(w_m) - w_m||^2] \le \sigma_{quant}^2$.
    * *Với Chưng cất tri thức (KD):* Tối ưu hóa cực tiểu phân kỳ KL thay vì truyền tham số: $\mathcal{L}_{KD} = D_{KL}(q_{global} || q_{local}) = \sum q_{local} \log\left(\frac{q_{local}}{q_{global}}\right)$.
* **Đề xuất giải pháp:** Nâng cấp hệ thống với **Federated Knowledge Distillation (FedKD)** kết hợp **Extreme Quantization (INT8)**. Thay vì truyền các ma trận trọng số khổng lồ, các AUV chỉ gửi các dự đoán phân phối xác suất mềm (soft-logits) đã được lượng tử hóa. Điều này cho phép nén gói tin hàng vạn lần, hóa giải triệt để hạn chế băng thông dưới nước mà không làm mất đi năng lực biểu diễn tri thức.

<!-- # Topology FL (Kiến trúc Mạng Học Liên kết)

Nhóm phương pháp này tập trung định hình lại luồng giao tiếp mạng nhằm loại bỏ nút thắt cổ chai (bottleneck) tại AUV dẫn đường và giảm rủi ro rớt gói do khoảng cách truyền xa.

* **Hierarchical Federated Learning (HFL - Học liên kết phân cấp):**
    * **Cơ chế:** Chia bầy AUV thành các cụm (Clusters). Trong mỗi cụm có một Cluster Head (CH). Các AUV chỉ gửi tham số cho CH để tổng hợp cục bộ (Local Aggregation), sau đó các CH mới gửi tham số lên Leader.
    * **Hiệu quả:** Giảm độ trễ hàng đợi $W_L$ tại Leader, giảm khoảng cách vật lý giữa các node truyền dẫn, từ đó giảm tỷ lệ $PLR$.
* **Decentralized Federated Learning (DFL - Học liên kết phi tập trung):**
    * **Cơ chế:** Loại bỏ hoàn toàn nút Leader. Các AUV giao tiếp theo mô hình mạng lưới (Mesh/P2P) và sử dụng thuật toán đồng thuận (ví dụ: Gossip Averaging) để đồng bộ mô hình trực tiếp với các AUV lân cận.
    * **Hiệu quả:** Xóa bỏ hoàn toàn single-point-of-failure (Leader hỏng thì toàn bầy hỏng) và phân tán đều mức tiêu thụ năng lượng cho toàn mạng.
* **SplitFed Learning (Học phân tách kết hợp FL):**
    * **Cơ chế:** Cắt mô hình mạng nơ-ron thành hai phần. AUV chỉ tính toán một vài lớp đầu tiên (tạo ra các *smashed data*) và gửi lên Leader để Leader tính toán các lớp phức tạp còn lại.
    * **Hiệu quả:** Phù hợp nếu phần cứng CPU của AUV quá yếu, nhưng cần đánh đổi bằng việc truyền dữ liệu trung gian thay vì gradient.

# Communication Reduction: Knowledge Distillation & Quantization

Môi trường dưới nước có băng thông cực thấp. Nhóm phương pháp này không can thiệp vào cấu trúc mạng mà tác động trực tiếp vào "trọng lượng" của gói tin truyền tải.

* **Federated Knowledge Distillation (FedKD - Chưng cất tri thức liên kết):**
    * **Cơ chế:** AUV (Teacher) không gửi ma trận tham số $w_m$ khổng lồ. Thay vào đó, chúng gửi các đầu ra dự đoán xác suất (soft-logits) dựa trên một tập dữ liệu tham chiếu (Public Dataset/Dummy Data). Leader (Student) học cách bắt chước các phân phối này.
    * **Toán học:** Việc cập nhật mô hình được thực hiện bằng cách tối ưu hóa phân kỳ KL: $\mathcal{L}_{KD} = \sum q(x) \log\left(\frac{q(x)}{p(x)}\right)$.
    * **Hiệu quả:** Cho phép các AUV sử dụng kiến trúc mô hình hoàn toàn khác nhau (Heterogeneous Models) và nén dung lượng gói tin xuống mức siêu nhỏ.
* **Extreme Quantization (Lượng tử hóa sâu):**
    * **Cơ chế:** Chuyển đổi trọng số từ định dạng Float32 xuống INT8, INT4, hoặc thậm chí Binary (1-bit).
    * **Hiệu quả:** Tiết kiệm từ 75% đến 96% băng thông mạng. Thích hợp kết hợp với cơ chế sửa lỗi tiến (FEC - Forward Error Correction) để bù đắp cho kênh truyền nhiễu.
* **Sparsification (Làm thưa ma trận):**
    * **Cơ chế:** Chỉ gửi top-k các gradient có giá trị tuyệt đối lớn nhất, ép các giá trị nhỏ về 0 (như tác giả bài báo đang cố gắng làm nhưng ở mức sơ khai). Cần kết hợp với *Error Feedback* (tích lũy phần dư chưa gửi cho vòng sau) để mô hình không bị mất hội tụ.

# Multi-FL (Multitask, Multi-Objective, MARL) & Kỹ thuật hóa Môi trường

Nhóm này giải quyết sự phi thực tế của dữ liệu (Non-IID), tối ưu hóa đường dài (Năng lượng) và hiện thực hóa môi trường vật lý.

* **Xử lý Dữ liệu Non-IID (FedProx, SCAFFOLD):**
    * **Cơ chế:** Bổ sung các thuật ngữ toán học vào hàm mục tiêu cục bộ để căn chỉnh hướng cập nhật trọng số. Ví dụ, FedProx dùng Proximal term $\frac{\mu}{2}||w_m - w_t||^2$ để kéo gradient cục bộ không đi quá xa so với mô hình toàn cục.
    * **Hiệu quả:** Chống lại hiện tượng Weight Divergence khi AUV gặp phải vùng dữ liệu cực hiếm hoặc thiên lệch.
* **Multi-Agent Reinforcement Learning (MARL - Học tăng cường đa tác tử):**
    * **Cơ chế:** Thay vì PPO2 điều khiển tập trung, mỗi AUV là một Agent tự quyết định (Decentralized Actor - Centralized Critic). Agent tự học chiến lược: "Nếu pin của tôi $< 20\%$ và dữ liệu của tôi không mang thông tin mới, tôi sẽ im lặng".
    * **Hiệu quả:** Giải quyết triệt để bài toán Battery-Aware và tối ưu hóa năng lượng toàn vẹn đường dài.
* **Multi-Objective Optimization (Tối ưu hóa đa mục tiêu theo ranh giới Pareto):**
    * **Cơ chế:** Hàm mục tiêu không cộng dồn tuyến tính $T(t) + E(t)$ một cách thô sơ. Thiết kế hệ thống tìm điểm cân bằng Pareto giữa 3 mục tiêu đối kháng: Cực đại độ chính xác (Accuracy), Cực tiểu độ trễ (Latency), Cực tiểu tiêu thụ năng lượng (Energy).
* **Realistic Environment Modeling (Mô phỏng thực tế hóa):**
    * **Cơ chế:** Đưa kênh truyền Markov (Gilbert-Elliott model) vào để mô phỏng chuỗi rớt gói (bursty packet loss) của âm thanh dưới nước.
    * **Hiệu quả:** Tác tử RL (MARL) bị buộc phải học cách đối phó với môi trường xấu (ví dụ: tự động tăng mức độ nén hoặc chờ đến khi đội hình bơi vào vùng nước ít nhiễu mới truyền dữ liệu).


# Bảng Đối chiếu: Bài báo gốc vs. Môi trường Thực tế

| Khía cạnh Vật lý | Giả định trong Bài báo gốc | Cải tiến Môi trường Thực tế (Đề xuất) | Hậu quả nếu không cải tiến |
| :--- | :--- | :--- | :--- |
| **Khoảng cách & Đội hình** | Cố định hoàn hảo $D = 20\text{m}$, bay cùng tốc độ và độ sâu. | **Động lực học (Kinematics):** Khoảng cách $D_m(t)$ biến thiên liên tục do tác động của hải lưu và sai số điều hướng. | Thuật toán RL bị "overfit" với khoảng cách 20m, sẽ thất bại ngay khi đội hình bị xô lệch. |
| **Độ trễ truyền dẫn (Delay)** | Chỉ tính độ trễ phát tín hiệu $T_{m}^{LU} = \frac{w_m}{R_m^U}$. Bỏ qua thời gian sóng âm lan truyền. | **Trễ lan truyền (Propagation Delay):** Bắt buộc cộng thêm $T_{prop} = \frac{D_m(t)}{v_{sound}}$ (âm thanh chỉ đi được khoảng 1500 m/s). | Mô hình FL bị sai lệch nhịp đồng bộ, Leader tổng hợp sai tham số do nhận các gói tin ở các dòng thời gian khác nhau. |
| **Tỷ lệ rớt gói (Packet Loss)** | Kênh truyền hoàn hảo, tỷ lệ lỗi gói tin (PLR) bằng 0. Cứ phát là Leader nhận được. | **Mất gói theo chùm (Bursty Loss):** Sử dụng Chuỗi Markov (Mô hình Gilbert-Elliott) để tạo ra các khoảng đứt gãy tín hiệu ngẫu nhiên do Fading. | Hàm mục tiêu $T(t)$ tính sai hoàn toàn, hệ thống không biết cách phản ứng khi mất kết nối. |
| **Đụng độ & Can nhiễu (MAC)** | Bỏ qua hoàn toàn đụng độ gói tin. Coi như các AUV có băng thông độc lập tuyệt đối. | **Xác suất đụng độ (Collision Probability):** Đưa giao thức MAC (ví dụ CSMA/CA) vào. Khi nhiều AUV cùng truyền, nhiễu giao thoa sẽ phá hủy gói tin. | Việc các "nút lười" dừng truyền không phản ánh được lợi ích cốt lõi nhất: giảm đụng độ kênh truyền cho toàn bầy. | -->

# Câu hỏi cần giải đáp: 

1. Môi trường Truyền dẫn & Cấu hình Phần cứng (Vật lý & Năng lượng): Bài báo sử dụng mô hình kênh truyền âm thanh/vật lý nào? Có xét đến các yếu tố môi trường thực tế như Tỷ lệ mất gói (Packet Loss Rate - PLR), fading đa đường, hay hiệu ứng Doppler không? Về mặt phần cứng, cấu hình năng lượng của AUV được thiết lập ra sao: chỉ tối ưu tiêu hao tĩnh theo từng vòng (myopic) hay có theo dõi trạng thái dung lượng pin dài hạn (battery capacity over time)? (Yêu cầu: Trích xuất các phương trình thông lượng, độ trễ và ràng buộc năng lượng nếu có).

2. Cấu trúc Mạng (Topology) & Giao thức Truyền thông: Hệ thống sử dụng mạng Topology nào (Star, Mesh, Clustered, Hierarchical)? Quá trình giao tiếp giữa các AUV và nút trung tâm diễn ra qua phương thức nào (Single-hop, Multi-hop, Broadcast, Unicast)? Thiết kế Topology và cách thức truyền thông này có được lý giải là phù hợp với đặc thù hạn chế của môi trường nước không?

3. Phân bổ Tải truyền thông & Nút thắt cổ chai (Bottleneck): Dựa trên Topology được chọn, bài báo có phân tích hiện tượng thắt cổ chai hoặc độ trễ hàng đợi (Queueing delay) tại Server/Leader không? Để tối ưu hóa và giảm tải truyền thông, hệ thống có áp dụng các kiến trúc phân cấp (Tiered architecture), tổng hợp cục bộ (Local aggregation trong Cluster) hay tính toán chia nhỏ (Split Computing) không?

4. Dữ liệu & Hàm mục tiêu FL: Giả định phân bố dữ liệu thu thập được là IID hay Non-IID? Thuật toán Federated Learning cốt lõi được sử dụng để cập nhật mô hình là gì (FedAvg, FedProx, FedNova...)? (Yêu cầu: Trích xuất hàm mất mát cục bộ và hàm mục tiêu toàn cục).

5. Tối ưu Truyền thông qua Kỹ thuật Nén: Để tiết kiệm tối đa chi phí giao tiếp dưới nước, cơ chế nén/giảm tải tham số là gì? Bài báo có sử dụng các phương pháp nén tiên tiến như Lượng tử hóa (Quantization), Chưng cất tri thức (Knowledge Distillation) không, hay chỉ dùng kỹ thuật ngắt giao tiếp/chọn lọc (như Lazy nodes, Dropout, Gradient Sparsification)?

# Decentralized Federated Learning Papers 

Đặc điểm chung

* **1. Bài toán (Khử trung tâm & Vượt rào cản mạng):** Cả hai đều loại bỏ hoàn toàn máy chủ trung tâm để tránh "thắt cổ chai" và rủi ro hỏng hóc đơn điểm. Trọng tâm là duy trì việc học máy trong điều kiện mạng khắc nghiệt (đứt kết nối do rớt gói hoặc topology thay đổi do di chuyển) với băng thông cực kỳ hạn chế.
* **2. Hàm mục tiêu (Tối ưu hóa toàn cục):** Dù mạng lưới đứt gãy hay biến động, mục tiêu tối thượng không đổi là tìm ra bộ tham số $x^*$ giúp cực tiểu hóa hàm mất mát chung của toàn hệ thống (được tính bằng trung bình cộng các hàm mất mát cục bộ trên từng thiết bị):
    $$x^{*} = \arg\min_{x \in \mathbb{R}^d} \frac{1}{N} \sum_{i=1}^{N} f_i(x)$$
* **3. Phương pháp (Sự tất yếu của Gossip/Consensus):** Chính hàm mục tiêu (đòi hỏi tri thức toàn cục) đặt trong bối cảnh bài toán (không có máy chủ gom dữ liệu) đã **bắt buộc** cả hai nghiên cứu phải dùng giao tiếp ngang hàng (Gossip-based). Các node phải liên tục "trộn" mô hình với hàng xóm lân cận bằng các thuật toán đồng thuận (Consensus), kết hợp thêm cơ chế giảm xóc (như $\gamma_t$ suy giảm hoặc CECA) để ép mô hình hội tụ trên dữ liệu Non-IID.

## Paper 1: A Communication-Efficient Semi-Decentralized Approach for Federated Learning with Stragglers

**1. Môi trường Truyền dẫn & Cấu hình Phần cứng (Vật lý & Năng lượng):**

Tài liệu được cung cấp không đi sâu vào việc mô hình hóa kênh truyền âm thanh dưới nước, hiệu ứng Doppler, đa đường (multipath fading), hay các đặc thù của phương tiện tự hành dưới nước (AUV) như quản lý dung lượng pin dài hạn. Thay vào đó, bài báo tiếp cận vấn đề nhiễu kết nối và Tỷ lệ mất gói (PLR) thông qua một mô hình trừu tượng về các "nút trễ" (stragglers)

Việc mất gói/ngắt kết nối (stragglers) được mô hình hóa toán học cực kỳ đơn giản bằng một biến ngẫu nhiên Bernoulli. Xác suất thiết lập liên kết thành công từ thiết bị $i$ tới server ở vòng $t$ là $I_i^t$, với $I_i^t \sim Bernoulli(1-p)$.


**2. Cấu trúc Mạng (Topology) & Giao thức Truyền thông:**
* **Topology:** Hệ thống sử dụng mạng Bán phi tập trung (Semi-decentralized Federated Learning - SFL). Cấu trúc này kết hợp giữa mô hình Star (có Server trung tâm) và mạng Mesh/Ring (giao tiếp ngang hàng P2P giữa các thiết bị).
* **Phương thức giao tiếp:** * Server phát sóng (Broadcast) mô hình toàn cục xuống tất cả thiết bị.
    * Các thiết bị giao tiếp ngang hàng (P2P) với nhau trong một số vòng lặp nhất định để đạt sự đồng thuận (Multi-hop/Consensus).
    * Các thiết bị không bị lỗi mạng (non-stragglers) sẽ gửi kết quả đã tổng hợp cục bộ lên server.
* **Phù hợp với môi trường nước?** Không đề cập/Bỏ ngỏ. Bài báo giải quyết bài toán mạng không dây nói chung (Wireless Networks) có thiết bị kết nối chập chờn (stragglers), không nhắm tới môi trường IoUT. 

**3. Phân bổ Tải truyền thông & Nút thắt cổ chai (Bottleneck):**
* **Độ trễ hàng đợi (Queueing delay):** Không đề cập/Bỏ ngỏ.
* **Giảm tải & Kiến trúc:** Thay vì phân tích độ trễ hàng đợi tại server, bài báo tập trung giải quyết hiện tượng "thắt cổ chai dữ liệu" do các node bị rớt mạng (stragglers) không gửi được gradient lên server. Để khắc phục, hệ thống áp dụng cơ chế **Tổng hợp cục bộ (Local aggregation)** thông qua giao tiếp P2P. Các thiết bị khỏe (non-stragglers) sẽ "gánh" tải bằng cách gom gradient của các thiết bị yếu xung quanh (previous neighbors) và gửi đại diện lên server, thay vì để mất hoàn toàn dữ liệu của các thiết bị yếu đó. Không sử dụng Split Computing hay Tiered architecture.

**4. Dữ liệu & Hàm mục tiêu FL:**
* **Phân bố dữ liệu:** Giả định dữ liệu là Non-IID. Trong phần mô phỏng, tác giả cố tình chia tập dữ liệu MNIST sao cho chỉ có 2 thiết bị sở hữu nhãn cụ thể, tạo ra sự không đồng nhất. Về mặt toán học, sự không đồng nhất này (heterogeneity) được giới hạn bởi tham số $\beta^2$: 
    $$||\nabla f_i(x) - \frac{1}{N}\nabla f(x)||_2^2 \le \beta^2, \forall i, \forall x$$
* **Thuật toán FL cốt lõi:** Phương pháp đề xuất có tên là COFFEE, dựa trên Distributed/Federated SGD kết hợp với thuật toán đồng thuận CECA:
Thuật toán COFFEE ứng dụng trực tiếp CECA vào SFL qua 3 giai đoạn:

    * **Bước 1: Tính toán cục bộ (SGD):** Mỗi thiết bị tự tính gradient $g_i^t$ trên dữ liệu của mình.
    * **Bước 2: Trộn lẫn bằng CECA (Giao tiếp ngang hàng - P2P):** Các thiết bị liên lạc với hàng xóm (neighbor) trong $R$ bước lặp. Ở mỗi bước, thay vì gửi bản gốc, chúng đổ nước vào chung một bình, khuấy đều (tính trung bình cộng), rồi mỗi bên chia nhau cầm một nửa mang về. 
        * *Kết quả:* Sau $R$ bước, thiết bị $i$ sẽ cầm một hỗn hợp $q_i^R$, đại diện cho trung bình cộng gradient của chính nó và một số hàng xóm trước đó.
    * **Bước 3: Nộp bài (Giao tiếp Lên Server):** Bất kỳ thiết bị nào hiện đang có mạng kết nối với Server (non-stragglers) sẽ gửi bản "hỗn hợp" này lên.


* **Hàm mục tiêu toàn cục:** $$x^* = \arg\min_{x\in\mathbb{R}^d} f(x) \triangleq \arg\min_{x\in\mathbb{R}^d} \sum_{i=1}^N f_i(x)$$
* **Hàm mất mát cục bộ:** $$f_i(x) = \mathbb{E}_{\zeta \sim \mathcal{D}_i} F(x, \zeta)$$

**5. Tối ưu Truyền thông qua Kỹ thuật Nén (Compression):**
* **Cơ chế giảm tải/nén:** Không sử dụng Lượng tử hóa (Quantization) hay Chưng cất tri thức (Knowledge Distillation). 
* **Cách thức thực hiện:** Bài báo tối ưu truyền thông ở cấp độ "Topology routing" chứ không phải nén dữ liệu. Phương pháp baseline là Gradient Coding (GC) đòi hỏi mỗi thiết bị phải gửi gradient nguyên vẹn của nó cho một lượng lớn thiết bị khác, gây tràn băng thông. Phương pháp COFFEE giảm thiểu số vòng giao tiếp P2P (chỉ cần tối đa $\lceil \log_2 N \rceil$ bước) bằng thuật toán CECA (Communication-optimal exact consensus algorithm). Các thiết bị sẽ tính trung bình cộng gradient của bản thân và hàng xóm rồi gửi đi, thay vì phải gửi/nhận từng gradient gốc một cách dư thừa.


## Paper 2: Mobility-aware Decentralized Federated Learning for Autonomous Underwater Vehicles

**1. Môi trường Truyền dẫn & Cấu hình Phần cứng (Vật lý & Năng lượng):**
* **Kênh truyền & Suy hao:** Bài báo sử dụng mô hình kênh âm thanh nước nông (shallow-water acoustic channel). Nó xem xét các yếu tố môi trường như suy hao do khoảng cách và tần số (attenuation $A(d,f) = d^{A_0} a(f)^d$ trong đó $a(f)$ tính bằng công thức Thorp), cùng 4 loại nhiễu môi trường: nhiễu hỗn loạn (turbulence), nhiễu tàu thuyền (shipping), nhiễu sóng (wave) và nhiễu nhiệt (thermal) để tính SNR $\gamma(d,f)$. Tuy nhiên, nó không xem xét Tỷ lệ mất gói (PLR), fading đa đường hay hiệu ứng Doppler.
* **Thông lượng/Băng thông:** Phương trình công suất kênh truyền (channel capacity) được sử dụng là:
    $C(d)=B \log_{2}(1+\frac{\beta P_{T}\tilde{\gamma}(d)}{2\pi(1\mu Pa)B})$
* **Năng lượng:** **Không đề cập/Bỏ ngỏ**. Mặc dù có xét ràng buộc năng lượng tổng (Total Transmit Power $P_T = 300mW$) cho mỗi thiết bị trong phần mô phỏng, không có phương trình tối ưu năng lượng hay biến trạng thái dung lượng pin dài hạn nào được đưa ra.

**2. Cấu trúc Mạng (Topology) & Giao thức Truyền thông:**
* **Topology:** Hệ thống sử dụng mạng phi tập trung hoàn toàn (Decentralized Federated Learning - DFL) thông qua một đồ thị thay đổi theo thời gian (time-varying graph $\mathcal{G}^{(t)}$).
* **Phương thức giao tiếp:** Giao tiếp diễn ra qua kênh broadcast (Multicast/Broadcast), trong đó các AUV chỉ kết nối và đồng bộ mô hình (P2P) với các hàng xóm (neighbors) nằm trong một khoảng cách kết nối nhất định ($d_{con}$) và ngắt kết nối nếu vượt quá khoảng cách ($d_{disc}$).

**3. Phân bổ Tải truyền thông & Nút thắt cổ chai (Bottleneck):**
* **Hiện tượng thắt cổ chai:** Bài báo lập luận rằng DFL loại bỏ sự phụ thuộc vào máy chủ trung tâm, từ đó tự động giải quyết các nút thắt tính toán và truyền thông tại Server. Do đó, không có phân tích về độ trễ hàng đợi (Queueing delay) tại Server/Leader.
* **Giảm tải:** Hệ thống không áp dụng kiến trúc phân cấp (Tiered architecture) hay Split Computing. Khối lượng truyền thông được phân bổ qua việc trao đổi thông tin cục bộ (Local aggregation) giữa các neighbor bằng một ma trận trộn (mixing matrix $W^{(t)}$) dựa trên trọng số Metropolis.

**4. Dữ liệu & Hàm mục tiêu FL:**
* **Phân bố dữ liệu:** Giả định dữ liệu là Non-IID (không độc lập và phân phối đồng nhất). Sự không đồng nhất này được định lượng và giả định bị giới hạn bởi:
    $\sum_{i=1}^{n}||\nabla f_{i}(x^{*})-\nabla f(x^{*})||_{2}^{2}\le\zeta^{2}, \forall i$
* **Thuật toán FL cốt lõi:** Phương pháp là sự kết hợp của Gossip-based Decentralized SGD với **thước đo đồng thuận suy giảm theo thời gian (decaying consensus step $\gamma_t$)**. Thay vì dùng trọng số đồng thuận cố định như CHOCO-SGD, bước đồng thuận $\gamma_t$ được đặt ở mức lớn ban đầu để bù đắp sai lệch dữ liệu Non-IID, và giảm dần sau một số vòng lặp $t_d$ để hội tụ khi có nhiễu nén.
    * **Giai đoạn 1: Phá vỡ định kiến (Khởi đầu với $\gamma_t$ lớn)**
        * **Hành động:** Các AUV "tin tưởng tuyệt đối" và pha trộn mạnh mẽ mô hình của hàng xóm vào mô hình của mình.
        * **Mục đích:** Nhanh chóng thoát khỏi góc nhìn phiến diện cục bộ để hình thành cái nhìn tổng quan. Giai đoạn này chấp nhận hy sinh độ sắc nét và hoàn toàn phớt lờ nhiễu do nén.
    * **Giai đoạn 2: Tinh chỉnh hội tụ (Suy giảm $\gamma_t$ sau vòng $t_d$)**
        * **Hành động:** Các AUV giảm dần sự phụ thuộc vào thông tin từ bên ngoài, chuyển sang tự tinh chỉnh cục bộ nhiều hơn.
        * **Mục đích:** Triệt tiêu sự rung lắc do "nhiễu nén" từ hàng xóm gây ra. Việc bớt lắng nghe giúp mô hình từ từ ổn định và hội tụ thành một phiên bản hoàn chỉnh, chính xác nhất.

* **Cực tiểu hóa hàm mất mát toàn cục (Global Loss Function):** Mục tiêu cốt lõi là tìm ra bộ trọng số mô hình tối ưu ($x^*$) sao cho tổng các hàm mất mát cục bộ của tất cả các AUV là nhỏ nhất.
    * **Công thức:** $x^* \triangleq \arg\min_{x \in \mathbb{R}^d} f(x)$, trong đó $f(x) \triangleq \frac{1}{N} \sum_{i=1}^N f_i(x)$.

* **Hàm mất mát cục bộ:**
    $f_{i}(x_{i})=\mathbb{E}_{\xi_{i}\sim\mathcal{D}_{i}}[F_{i}(x_{i},\xi_{i})]$

* **Về cách di chuyển (Mobility):** Đây là một **đặc điểm môi trường** và là thách thức mà thuật toán phải thích nghi. Bài báo sử dụng đồ thị thay đổi theo thời gian (time-varying graph) để mô tả việc các liên kết truyền thông bị thay đổi do AUV di chuyển, chứ không nhằm mục đích tìm ra cách di chuyển tối ưu.
* **Về năng lượng:** Bài báo này **không đề cập** đến việc tối ưu hóa năng lượng trong hàm mục tiêu. Năng lượng chỉ được nhắc đến như một hạn chế chung của thiết bị đầu cuối (edge devices) hoặc được giả định cố định ($P_T$) để so sánh hiệu suất giữa các kịch bản.


**5. Tối ưu Truyền thông qua Kỹ thuật Nén (Compression):**
* **Cơ chế nén:** Bài báo áp dụng kỹ thuật nén với cơ chế bù lỗi (error-compensation technique). Mỗi thiết bị không gửi toàn bộ mô hình mà chỉ nén sự khác biệt (difference) giữa mô hình cục bộ mới nhất và mô hình ước tính của nó:
    $q_{i}^{(t)}=Q(x_{i}^{(t+\frac{1}{2})}-\hat{x}_{i}^{(t)})$
* **Phương pháp nén cụ thể:** Kỹ thuật **Sparsification (Top-k sparsification)** được sử dụng trong phần mô phỏng để nén thông tin, giúp giảm số lượng bit cần truyền sao cho phù hợp với sức chứa hạn chế của kênh truyền âm thanh $b_{c}l_{c}\le t_{c}C(d)$. Không có sử dụng Knowledge Distillation (Chưng cất tri thức) hay Lượng tử hóa (Quantization) thuần túy.

# Clustered Federated Learning Papers

Đặc điểm chung

1. **Bài toán (Sự cạn kiệt tài nguyên):** Môi trường làm việc khắc nghiệt (dưới nước hoặc mạng IoT mật độ cao) khiến năng lượng pin và băng thông bị giới hạn nghiêm ngặt, việc truyền dữ liệu thô đi xa về một máy chủ trung tâm là hoàn toàn bất thi.
2. **Hàm mục tiêu (Tối ưu hóa đa biến):** Không chỉ theo đuổi độ chính xác của AI, hệ thống bắt buộc phải giải bài toán đánh đổi (trade-off): Cân bằng giữa chất lượng học máy/định tuyến với việc tiết kiệm năng lượng và giảm độ trễ nhằm kéo dài tối đa tuổi thọ của mạng lưới.
3. **Phương pháp (Phân cụm & Trí tuệ biên):** Bị ép buộc bởi bài toán và hàm mục tiêu trên, cả 3 bài đều phải chuyển sang cấu trúc mạng Phân cấp/Phân cụm (Clustering/Hierarchical). Các thiết bị sẽ tự xử lý dữ liệu cục bộ, gom thành nhóm, bầu ra "thủ lĩnh" (Cluster Head / Fog Node) để nén thông tin rồi mới gửi đi, giúp triệt tiêu hoàn toàn nút thắt cổ chai truyền thông.

## 1. Optimizing Cluster Head Selection and Routing in 5G WSNs: A Reinforcement Learning and Deep Learning Approach

**1. Môi trường Truyền dẫn & Cấu hình Phần cứng**
* **Môi trường vật lý:** Bài báo sử dụng mô hình vô tuyến trong không gian tự do (Free space) và fading đa đường (Multipath fading), không sử dụng kênh truyền âm thanh dưới nước. Hiệu ứng Doppler không được nhắc đến trong nghiên cứu này.
* **Tỷ lệ mất gói (PLR):** Bài báo có đo lường tỷ lệ rớt gói (Packet drop ratio - PDR) để tính toán mật độ lưu lượng. Tỷ lệ phân phối gói tin thành công (Packet Delivery Ratio) của hệ thống đạt trên 99.5%.
* **Cấu hình năng lượng phần cứng:** Hệ thống không dùng AUV mà phân loại các nút cảm biến tĩnh thành ba loại: tiên tiến, trung gian và bình thường. Năng lượng được theo dõi liên tục theo dung lượng pin còn lại và khi pin giảm xuống 0%, nút đó được coi là nút chết (dead node).

**2. Cấu trúc Mạng (Topology) & Giao thức Truyền thông**
* **Topology:** Hệ thống áp dụng cấu trúc mạng phân cụm (Clustered topology). * **Phương thức giao tiếp:** Quá trình truyền thông diễn ra qua cơ chế một bước nhảy (single-hop) từ nút cảm biến đến Cụm trưởng (CH). Cụm trưởng sau đó gửi dữ liệu lên Trạm gốc (Base Station) trực tiếp hoặc qua thêm một bước nhảy phụ (multi-hop).
* **Độ phù hợp:** Thiết kế mạng lưới này được lý giải là phù hợp với các ứng dụng IoT đô thị, thành phố thông minh và tự động hóa công nghiệp (Industry 4.0) quy mô từ 100 đến 10.000 nút, hoàn toàn không phải thiết kế cho môi trường nước.


**3. Phân bổ Tải truyền thông & Nút thắt cổ chai (Bottleneck)**
* **Kiến trúc phân cấp & Tổng hợp cục bộ:** Để giảm tải truyền thông, hệ thống sử dụng một kiến trúc phân cấp rõ rệt thông qua việc tạo cụm. Cụm trưởng (CH) đóng vai trò tổng hợp cục bộ (aggregate) các dữ liệu thu thập được từ các nút thành viên trước khi gửi lên Trạm gốc.
* **Hiện tượng thắt cổ chai:** Bài báo không đo lường độ trễ hàng đợi tại Server. Tuy nhiên, để tránh tắc nghẽn cục bộ, hệ thống đánh giá mật độ lưu lượng (Traffic Density) dựa trên việc sử dụng bộ đệm (Buffer usage), tải kênh truyền (Channel load) và tỷ lệ rớt gói (Packet drop ratio) nhằm loại bỏ các nút đang quá tải khỏi danh sách bầu chọn Cụm trưởng.
* **Split Computing:** Bài báo không áp dụng kỹ thuật tính toán chia nhỏ (Split Computing).


**4. Dữ liệu & Hàm mục tiêu
* **Thuật toán cốt lõi:** Nghiên cứu này không sử dụng các thuật toán Federated Learning như FedAvg hay FedProx để cập nhật mô hình. Kiến trúc được đề xuất là sự kết hợp giữa Học tăng cường (RL) để phân cụm, thuật toán MRFO để chọn Cụm trưởng, và Mạng niềm tin sâu (DBN) để định tuyến.
* **Phân phối dữ liệu:** Không có bất kỳ giả định nào về phân phối dữ liệu IID hay Non-IID.
* **Hàm mục tiêu cục bộ (RL):** Tác nhân Học tăng cường tối đa hóa hàm giá trị tích lũy dựa trên phần thưởng: $V^{\pi}(S_{i})=r_{i}+yr_{i+1}+y^{2}r_{i+2}+...=$.
* **Hàm mất mát toàn cục (DBN):** Mạng DBN sử dụng sai số toàn phương trung bình (MSE) để tính toán lỗi trong pha huấn luyện: $E=\frac{1}{x}\times\sum_{v=1}^{x}(P_{z}^{v}-P_{nr}^{v})$.



## 2. Energy-Aware Clustered Federated Learning for Underwater Sensor Networks in Naval Surveillance

Note: Bài này ít biểu diễn toán học, chưa để chứng minh chính xác

### 1. Môi trường Truyền dẫn & Cấu hình Phần cứng (Vật lý & Năng lượng)
* **Môi trường vật lý:** Bài báo sử dụng mô hình kênh truyền âm thanh dưới nước (acoustic communication). Kênh truyền này được thiết lập với các yếu tố môi trường thực tế như Tỷ lệ mất gói (PLR) được mô phỏng ngẫu nhiên từ 5% - 10% (và lên đến 15% trong môi trường nhiễu cao). Bài báo cũng xem xét đến các hiệu ứng như fading đa đường (multipath fading) và hiệu ứng dịch chuyển tần số Doppler (Doppler shifts).
* **Cấu hình phần cứng & Năng lượng:** Hệ thống không đề cập trực tiếp đến AUV mà sử dụng các "nút cảm biến dưới nước" (underwater sensor nodes). Năng lượng không được tối ưu cận thị theo từng vòng mà được theo dõi trạng thái dung lượng pin dài hạn thông qua mức "năng lượng dư thừa" (residual energy) liên tục được cập nhật. Mức năng lượng ban đầu của các nút dao động từ 1000 - 1200 Joules.
* **Các phương trình liên quan:**
    * **Xác suất tham gia huấn luyện (dựa trên năng lượng):** $$P_{i}=\alpha\times(\frac{E_{i}}{E_{max}})+(1-\alpha)\times\eta_{i}$$ 
        Trong đó, $E_{i}$ là năng lượng còn lại của nút $i$, $E_{max}$ là năng lượng tối đa quan sát được, $\eta_{i}$ là hệ số ngẫu nhiên đảm bảo tính công bằng, và $\alpha$ kiểm soát trọng số ưu tiên năng lượng.
    * **Cập nhật năng lượng dư thừa (từ Thuật toán 1):** Cập nhật $E_{i}^{(t+1)}$ dựa trên chi phí tính toán (0.5 Joules/epoch) và truyền tải (2 Joules/KB).
    * **Độ trễ và Thông lượng:** Không có phương trình toán học cụ thể, nhưng cấu hình vật lý giới hạn Băng thông (Bandwidth) ở mức 10-20 kbps và Độ trễ (Latency) từ 0.5-2 giây cho mỗi tin nhắn tùy thuộc khoảng cách.


### 2. Cấu trúc Mạng (Topology) & Giao thức Truyền thông
* **Topology:** Hệ thống áp dụng cấu trúc mạng phân cấp theo cụm (Clustered Hierarchical).
* **Phương thức giao tiếp:** Quá trình giao tiếp là sự kết hợp giữa truyền thông nội bộ cụm (Intra-Cluster) và liên cụm (Inter-Cluster). Các nút trong cụm phát sóng (broadcast) bản cập nhật mô hình của chúng cho một Cụm trưởng (Cluster Head - CH). Sau đó, các Cụm trưởng thực hiện kết nối đa bước (multi-hop) hoặc trực tiếp đến Trạm gốc Hải quân (Naval Base Station).
* **Sự phù hợp với môi trường nước:** Thiết kế này được lý giải là cực kỳ phù hợp vì việc giao tiếp trực tiếp (end-to-end) từ mọi nút đến máy chủ trung tâm là không khả thi dưới nước do giới hạn băng thông hẹp và tỷ lệ lỗi cao của kênh truyền âm thanh. Mô hình phân cấp giúp giảm thiểu triệt để số lượng các kết nối tầm xa không cần thiết.


### 3. Phân bổ Tải truyền thông & Nút thắt cổ chai (Bottleneck)
* **Nút thắt cổ chai:** Bài báo không đi sâu phân tích bằng công thức toán học về độ trễ hàng đợi (Queueing delay) tại Server. Thay vào đó, để giải quyết vấn đề nghẽn cổ chai do độ trễ truyền tải không đồng đều, hệ thống áp dụng cơ chế truyền thông bất đồng bộ (asynchronous communication timing), cho phép các nút và cụm gửi bản cập nhật bất cứ lúc nào thay vì chờ đợi đồng bộ hóa toàn cục.
* **Tối ưu và Giảm tải truyền thông:** Hệ thống áp dụng triệt để kiến trúc phân cấp (Hierarchical/Tiered architecture) và tổng hợp cục bộ (Local aggregation) tại Cụm trưởng. Thuật toán Tính toán chia nhỏ (Split Computing) không được sử dụng trong nghiên cứu này.


### 4. Dữ liệu & Hàm mục tiêu FL
* **Phân phối dữ liệu:** Bài báo giả định phân bố dữ liệu thu thập được là Non-IID (dữ liệu không độc lập và không phân phối đồng nhất).
* **Thuật toán cốt lõi:** Hệ thống không dùng FedAvg truyền thống do dễ bị ảnh hưởng bởi nhiễu. Thuật toán được đề xuất là Học liên kết phân cụm nhận thức năng lượng (CFL) sử dụng chiến lược tổng hợp dựa trên trung vị (Median-based aggregation) thay vì tính trung bình có trọng số.
* **Các phương trình mục tiêu/cập nhật:**
    * **Cập nhật mô hình cục bộ (Local update):**
        $$w_{i}^{(t)}=w^{(t-1)}-\mu\nabla l_{i}(w^{(t-1)})$$
    * **Hàm tổng hợp trung vị tại Cụm (Cluster-level aggregation):**
        $$w_{\mathcal{C}_{k}}^{(t)} = \text{median}(\{w_{i}^{(t)} | i \in \mathcal{C}_{k}\})$$
    * **Hàm tổng hợp toàn cục tại Trạm gốc (Inter-cluster aggregation):**
        $$w^{(t)} = \text{median}(\{w_{\mathcal{C}_{k}}^{(t)} | k = 1, \dots, K\})$$
    * **Nguyên lý trung vị cho tham số tổng quát:**
        $$w_{agg} = \text{median}(w_{1}, w_{2}, \dots, w_{n})$$


### 5. Tối ưu Truyền thông qua Kỹ thuật Nén
* **Cơ chế nén tham số:** Hệ thống áp dụng các kỹ thuật nén bản cập nhật (update compression methods) trước khi truyền tải. Cụ thể, bài báo sử dụng phương pháp Lượng tử hóa 8-bit (8-bit quantization) đối với các trọng số mô hình. Chưng cất tri thức (Knowledge Distillation) không được sử dụng.
* **Cơ chế ngắt giao tiếp/chọn lọc:** Bài báo áp dụng phương pháp chọn lọc sự tham gia (Energy-aware participant selection - tương tự như Lazy nodes/Dropout ở cấp độ Client). Cụ thể, hệ thống sẽ sử dụng một ngưỡng năng lượng động $\theta^{(t)}$; chỉ những nút có mức năng lượng dư thừa $E_{i}^{(t)} \ge \theta^{(t)}$ mới được đưa vào danh sách ứng viên và sau đó được chọn theo xác suất dựa trên năng lượng để tham gia vào vòng huấn luyện.

## 3. Energy-Efficient Hierarchical Federated Anomaly Detection for the Internet of Underwater Things via Selective Cooperative Aggregation (Nên đọc)

### 1. Môi trường Truyền dẫn & Cấu hình Phần cứng (Vật lý & Năng lượng)
* **Môi trường vật lý:** Bài báo sử dụng mô hình kênh truyền âm thanh dưới nước (Underwater Acoustic - UWA) kết hợp giữa suy hao lan truyền hình học và hệ số hấp thụ Thorp. Tạp âm môi trường được tính toán qua mô hình mật độ phổ công suất (PSD) Wenz, bao gồm nhiễu động lực học, tàu thuyền, gió và nhiệt.
* **Các yếu tố nhiễu:** Các hiện tượng vật lý như fading đa đường (multipath fading) hoặc hiệu ứng Doppler không được đưa trực tiếp vào công thức mô hình hóa. Thay vào đó, độ tin cậy được thiết lập qua giới hạn mức nguồn phát tối đa (capped source level) và Tỷ số Tín hiệu trên Nhiễu (SNR) để đánh giá tính khả thi của liên kết. 
* **Cấu hình năng lượng:** Hệ thống không tối ưu cận thị theo từng vòng. Khối pin của các nút được theo dõi sát sao qua từng khoảng thời gian dài, cập nhật lượng điện năng tiêu hao sau mỗi lần tính toán và truyền tải, đồng thời đảm bảo không vượt quá giới hạn năng lượng an toàn tối thiểu.
* **Phương trình toán học:**
    * *Mức nguồn phát tối thiểu (Ràng buộc năng lượng):* $$SL_{u}^{min}(u,v)=\gamma_{tgt}+TL(d_{uv},f)+NL(f,B)+IL$$
        Ràng buộc: $SL_{u}^{min}(u,v)\le SL_{max}$.
    * *Thông lượng (Dung lượng Shannon):* $$R_{uv}=B~log_{2}(1+10^{\gamma_{tgt}/10})$$.
    * *Độ trễ toàn vòng (Tính dựa trên liên kết chậm nhất):* $$\tau_{round}^{t}=max\{max_{i}\tau_{i\rightarrow a_{i}^{t}},max_{m,j}\tau_{m\rightarrow j}^{t},max_{m}\tau_{m\rightarrow g}^{t}\}+\tau_{comp}^{t}$$.

### 2. Cấu trúc Mạng (Topology) & Giao thức Truyền thông
* **Topology:** Sử dụng cấu trúc phân cấp 3 tầng (Three-tier Hierarchical) bao gồm: Lớp cảm biến (dưới sâu), Lớp Fog là các AUV (ở tầng nước giữa), và Cổng kết nối (trên bề mặt).

    * **Phân tầng cố định:** Kiến trúc hệ thống được chia sẵn thành 3 lớp phần cứng khác biệt: Cảm biến (dưới sâu), nút Fog/AUV (tầng nước giữa), và Cổng kết nối (trên mặt nước). Hoàn toàn không có cơ chế các nút cảm biến đồng cấp tự bầu chọn cụm trưởng với nhau.
    * **Vai trò cụm trưởng mặc định:** Các nút Fog (AUV) luôn được giao phó vai trò làm trạm thu gom và tổng hợp dữ liệu (tương đương chức năng của cụm trưởng) do chúng có năng lực tính toán mạnh hơn.
    * **Kết nối linh hoạt:** Mặc dù kiến trúc tầng là cố định, việc một cảm biến kết nối vào nút Fog nào lại diễn ra rất linh động theo từng vòng huấn luyện. Cảm biến sẽ tự động ghép cặp với nút Fog gần nhất mà đường truyền sóng âm có thể kết nối thành công.

* **Giao thức truyền thông:** Sự giao tiếp diễn ra dưới dạng kết hợp: truyền một bước nhảy (single-hop) từ cảm biến lên các nút Fog, sau đó các nút Fog có thể hợp tác giao tiếp ngang hàng với nhau (cooperative mixing), và cuối cùng đẩy mô hình lên Trạm mặt nước.
* **Mức độ phù hợp:** Cách thiết kế này được lý giải là đặc biệt tối ưu cho môi trường nước. Truyền thông âm thanh tiêu tốn nhiều năng lượng và băng thông cực hẹp. Nếu dùng mạng phẳng (flat FL), chỉ có khoảng 48% cảm biến kết nối được tới mặt nước, dẫn tới bỏ lọt dữ liệu. Phân cấp giúp 100% cảm biến tham gia được mạng lưới thông qua các đường truyền khả thi ngắn hơn lên AUV.


### 3. Phân bổ Tải truyền thông & Nút thắt cổ chai (Bottleneck)
* **Thắt cổ chai độ trễ:** Bài báo không dùng mô hình hàng đợi (Queueing delay) phức tạp tại Server mà đánh giá độ trễ nút thắt bằng hàm `max` của tất cả các luồng xử lý song song, cộng gộp cả thời gian lan truyền sóng âm, thời gian truyền tải bit và thời gian tính toán của vi xử lý cục bộ.
* **Kiến trúc giảm tải:** Áp dụng triệt để kiến trúc phân cấp (Tiered architecture).
* **Tổng hợp cục bộ (Local aggregation):** Các cập nhật được tập hợp và tính trung bình tại các nút Fog trước khi đẩy lên trên. 
* **Thiết kế cốt lõi (Selective Cooperative Aggregation):** Hệ thống không dùng Split Computing, mà giảm tải bằng chiến lược Hợp tác có chọn lọc. Thay vì các Fog luôn trao đổi dữ liệu với nhau (tốn rất nhiều năng lượng), hệ thống thiết lập luật: chỉ các Fog có lượng cụm nhỏ giọt mới được trao đổi mô hình với cụm lân cận lớn hơn để bù đắp sự thiếu đa dạng dữ liệu. 


### 4. Dữ liệu & Hàm mục tiêu FL
* **Phân phối dữ liệu:** Mạng lưới đối mặt với dữ liệu có độ không đồng nhất cao (Non-IID), được hệ thống mô phỏng bằng quy luật Dirichlet với hệ số $\alpha=0.1$. 
* **Thuật toán cốt lõi:** Phương pháp là một dạng mở rộng của Hierarchical FedAvg được tăng cường thông qua phép chia sẻ đồng thuận ngang hàng (gossip-style averaging). (Có so sánh với cả FedAvg và FedProx cơ bản).
* **Hàm mất mát cục bộ:** Do tính chất phát hiện bất thường, bài báo dùng Autoencoder tối thiểu hóa sai số tái tạo (Reconstruction Error):
    $F_{i}(\theta)=\frac{1}{n_{i}}\sum_{n=1}^{n_{i}}||x_{i,n}-h_{\theta}(x_{i,n})||_{2}^{2}$
* **Hàm sai số toàn cục:** $$F(\theta)=\sum_{i\in\mathcal{S}}\frac{n_{i}}{\sum_{k\in\mathcal{S}}n_{k}}F_{i}(\theta)$$

1. **Sai số mô hình toàn cục ($F(\theta^{T})$):** Là sai số tái tạo (reconstruction loss) của mô hình Autoencoder sau tổng số $T$ vòng huấn luyện.
2. **Tổng năng lượng tiêu thụ ($\sum E_{round}^{t}$):** Bao gồm năng lượng truyền tải từ cảm biến lên Fog, từ Fog sang Fog, và từ Fog lên Trạm mặt nước qua tất cả các vòng.
3. **Tổng độ trễ ($\sum \tau_{round}^{t}$):** Thời gian trễ của toàn bộ mạng trong quá trình giao tiếp.

**Phương trình hàm mục tiêu tổng quát được trích xuất từ bài báo như sau:**

$$\min \left( F(\theta^{T}) + \lambda_{E}\sum_{t=0}^{T-1}E_{round}^{t} + \lambda_{\tau}\sum_{t=0}^{T-1}\tau_{round}^{t} \right)$$

*Trong đó:*
* $\lambda_{E}$ và $\lambda_{\tau}$ là các hệ số trọng số (weighting coefficients) dùng để cân bằng mức độ ưu tiên giữa việc giảm năng lượng và giảm độ trễ so với việc tối ưu sai số mô hình.

**Ngoài ra, hàm mục tiêu này còn bị ràng buộc khắt khe bởi các điều kiện vật lý thực tế của môi trường nước:**
* Năng lượng pin còn lại của nút phải lớn hơn mức năng lượng dự trữ tối thiểu ($E_{i}^{t+1}\ge E_{min}$).
* Thời gian hoàn thành mỗi vòng không được vượt quá thời hạn cho phép ($\tau_{round}^{t}\le\tau_{max}$).
* Tỷ số tín hiệu trên nhiễu của kênh truyền âm thanh phải đạt ngưỡng khả thi tối thiểu ($SNR_{l}^{t}\ge\gamma_{tgt}$)

### 5. Tối ưu Truyền thông qua Kỹ thuật Nén
Để giải quyết bài toán băng thông eo hẹp (dưới 40 kbit/vòng), bài báo sử dụng một đường ống nén 2 giai đoạn:
* **Làm thưa Top-K (Top-K Sparsification):** Chỉ giữ lại $K$ tọa độ có độ lớn (magnitude) cao nhất. Những tham số bị cắt bỏ không bị vứt đi mà được gom vào một bộ đệm lỗi cục bộ (error feedback) để cộng bù vào vòng lặp sau.
* **Lượng tử hóa (Quantization):** Các giá trị vượt qua bước lọc Top-K tiếp tục bị lượng tử hóa xuống chuẩn 8-bit, giúp nén payload xuống còn 3% so với nguyên bản. 
* **Cơ chế ngắt giao tiếp:** Không sử dụng Knowledge Distillation hay Dropout. Nền tảng tiết kiệm dựa trên việc *tắt hoàn toàn các kết nối không khả thi* thông qua liên kết nhận thức (feasibility-aware association) và *hạn chế hợp tác* bằng luật ngắt sự trao đổi giữa các Fog nếu kích thước cụm của chúng đã đủ lớn (hệ số $c_{m}^{t}\le max\{2,0.75\overline{c}^{t}\}$).

### 6. Evaluate & Benchmark

#### 1. Các tiêu chí đánh giá (Evaluation Metrics)
Bài báo không chỉ đo lường độ chính xác và năng lượng, mà còn đưa yếu tố khả năng tham gia vào làm một tiêu chí chính:
* **Chất lượng phát hiện (Detection Quality):** Sử dụng điểm **F1** cho các kịch bản tổng hợp và điểm **PA-F1** (Point-Adjusted F1) cho các bộ dữ liệu thực tế.
* **Năng lượng giao tiếp (Communication Energy):** Đo lường tổng năng lượng tiêu thụ (tính bằng Joules) cho toàn bộ quá trình truyền tải trong mạng (từ cảm biến lên Fog, Fog sang Fog, và Fog lên Gateway).
* **Tỷ lệ tham gia (Network Participation):** Đo lường tỷ lệ phần trăm số lượng cảm biến có khả năng kết nối và thực sự tham gia vào quá trình huấn luyện mô hình so với tổng số cảm biến được triển khai.


#### 2. Các kịch bản đánh giá (Evaluation Scenarios)

#### A. Nhóm kịch bản mô phỏng tổng hợp (Synthetic Scenarios)
Được sử dụng để đánh giá hành vi của hệ thống khi thay đổi quy mô và điều kiện mạng dưới nước:
* **Kịch bản mở rộng quy mô (Scalability Study):** Mạng được thay đổi số lượng cảm biến $N \in \{50, 100, 150, 200\}$ với số nút Fog tương ứng $M = N/10$, chạy trong 20 vòng. Kịch bản này kiểm tra xem khi mạng mở rộng, các mạng phẳng (phải truyền thẳng lên mặt nước) sẽ bị rớt kết nối nhiều như thế nào so với mạng phân cấp.
* **Kịch bản nhạy cảm với dữ liệu không đồng nhất (Non-IID Sensitivity):** Thử nghiệm tại quy mô $N=100$ với dữ liệu được phân chia theo phân phối Dirichlet ở hai mức: $\alpha=0.1$ (dữ liệu cực kỳ không đồng nhất - strongly non-IID) và $\alpha \approx 10^{4}$ (gần như đồng nhất - near-IID). Mục tiêu là kiểm tra xem sự hợp tác giữa các nút Fog có thực sự giúp bù đắp lại sự thiếu đa dạng dữ liệu hay không.
* **Kịch bản tác động của nén dữ liệu (Effect of Compressed Uploads):** Đối chiếu độ tiêu hao năng lượng giữa việc truyền bản cập nhật mô hình nguyên bản so với khi áp dụng kỹ thuật nén (Sparsification Top-K với tỷ lệ $\rho_s=0.05$ và lượng tử hóa 8-bit).

#### B. Nhóm kịch bản với bộ dữ liệu thực tế (Real Benchmarks)
Để đảm bảo các kết luận từ mô phỏng có giá trị thực tiễn, bài báo chạy thử nghiệm (30 vòng) trên 3 bộ dữ liệu phát hiện bất thường đa biến nổi tiếng:
* **SMD (Server Machine Dataset):** Dữ liệu máy chủ với 38 đặc trưng (features).
* **SMAP (Soil Moisture Active Passive):** Dữ liệu đo xa vệ tinh với 25 đặc trưng.
* **MSL (Mars Science Laboratory):** Dữ liệu đo xa từ trạm thám hiểm sao Hỏa với 55 đặc trưng.

#### 3. Các mô hình cơ sở để so sánh (Baselines)
Trong các kịch bản trên, khung HFL-Selective (Hợp tác có chọn lọc) đề xuất được đem ra so sánh với 5 phương pháp khác để làm rõ sự đánh đổi giữa năng lượng và hiệu suất:
1.  **Centralised (Học tập trung):** Thu thập toàn bộ dữ liệu thô về máy chủ (chỉ dùng làm mốc tham chiếu lý tưởng trên dữ liệu thực tế, không khả thi dưới nước).
2.  **FedAvg:** Mạng phẳng truyền thống, chỉ những cảm biến kết nối trực tiếp được với Gateway mới tham gia.
3.  **FedProx:** Biến thể mạng phẳng mạnh nhất, giải quyết dữ liệu Non-IID tốt hơn.
4.  **HFL-NoCoop:** Phân cấp nhưng các nút Fog không hề giao tiếp với nhau (tiết kiệm năng lượng nhất trong nhóm phân cấp).
5.  **HFL-Nearest:** Phân cấp mà trong đó các nút Fog luôn luôn trao đổi dữ liệu với nút Fog gần nhất (tiêu hao năng lượng lớn nhất).

# Knowledge Distillation in FL

### Đặc điểm chung:

* **Mô hình Teacher cá nhân hóa (Personalized Teacher):** Mỗi client đều sở hữu một mô hình Teacher lớn, được huấn luyện và tối ưu hóa dựa trên đặc điểm dữ liệu cục bộ riêng biệt của chính client đó nhằm mang lại khả năng cá nhân hóa cao.
* **Mô hình Student dùng chung (Shared Student):** Tồn tại một mô hình Student nhỏ gọn hơn, đóng vai trò làm trung gian thống nhất, được chia sẻ và cập nhật toàn cục giữa tất cả các client.
* **Trích xuất kiến thức thích ứng (Adaptive Distillation):** Cường độ truyền đạt kiến thức giữa Teacher và Student không cố định mà được điều chỉnh tự động dựa trên mức độ chính xác (prediction correctness) của các dự đoán.
* **Học từ các trạng thái ẩn (Hidden States):** Để tối ưu hóa hiệu năng, mô hình Student không chỉ học từ kết quả đầu ra (soft labels) mà còn tiếp thu các đặc trưng sâu thông qua trạng thái ẩn và bản đồ chú ý (attention heatmaps) của mô hình Teacher.

### 3. Chiến lược Tối ưu hóa Truyền thông
* **Chỉ giao tiếp mô hình Student:** Để giải quyết bài toán chi phí truyền tải, hệ thống từ chối việc gửi các mô hình lớn (Teacher).Thay vào đó, chúng chỉ giao tiếp các bản cập nhật của mô hình Student nhỏ gọn giữa client và server.
* **Kỹ thuật nén kép (Double Compression):** Việc dùng KD để thu nhỏ mô hình mới chỉ là bước một. Cả hai framework đều áp dụng thêm một lớp nén kỹ thuật số khác trong quá trình giao tiếp:
    * **FedKD:** Kết hợp KD với phương pháp phân tích giá trị suy biến (SVD) để xấp xỉ và nén gradient với độ chính xác động.
    * **FedDT:** Kết hợp KD với lượng tử hóa bậc ba (Ternary Quantization) để biến đổi các trọng số liên tục của mô hình Student thành không gian rời rạc (-1, 0, 1), giúp giảm mạnh dung lượng.

* **Môi trường giả định:** Cả hai bài báo đều giả định máy chủ đáng tin cậy và các kênh truyền thông an toàn thông thường. Không có phương trình về thông lượng, độ trễ mạng vật lý hay dung lượng pin. Độ phức tạp tính toán được đánh giá qua công thức: $O(RD\Theta^t+RD\Theta^s+RPQ^2)$.

* **Hàm mục tiêu:** Dựa trên độ chính xác và hàm mất mát huấn luyện

### Bảng So sánh FedKD và FedDT

| Tiêu chí So sánh | FedKD (Knowledge Distillation + SVD) | FedDT (Knowledge Distillation + Ternary Quantization) |
| :--- | :--- | :--- |
| **1. Môi trường Truyền dẫn & Phần cứng** | **Không áp dụng môi trường nước/AUV.** Bài báo giả định máy chủ đáng tin cậy và các kênh truyền thông an toàn thông thường. Không có phương trình về thông lượng, độ trễ mạng vật lý hay dung lượng pin. Độ phức tạp tính toán được đánh giá qua công thức: $O(RD\Theta^t+RD\Theta^s+RPQ^2)$. | **Không áp dụng môi trường nước/AUV.** Chỉ xét các thiết bị viễn thông, di động và biên (edge devices) trong mạng thông thường. Không có đánh giá vật lý về Fading hay tỷ lệ mất gói (PLR). |
| **2. Cấu trúc Mạng (Topology)** | **Star Topology (Client-Server):** Các thiết bị đầu cuối (Client) giao tiếp trực tiếp một trạm (Single-hop) với máy chủ trung tâm (Server). Không đánh giá sự phù hợp với môi trường nước. | **Star Topology (Client-Server):** Các client tải mô hình từ server và gửi bản cập nhật về server. Bài báo nhấn mạnh các client tuyệt đối không giao tiếp ngang hàng chia sẻ thông tin với nhau. Không đánh giá sự phù hợp với môi trường nước. |
| **3. Phân bổ Tải & Thắt cổ chai** | Không phân tích trực tiếp hiện tượng trễ hàng đợi (Queueing delay) tại server. Để giảm tải thắt cổ chai do mô hình khổng lồ, hệ thống chỉ cho phép tải lên các mô hình "Mentee" (Student) rất nhỏ. Không sử dụng Split Computing hay Tiered Architecture. | Không đề cập đến độ trễ hàng đợi. Hệ thống giảm tải mạng bằng cách giới hạn việc truyền tải ở quy mô mô hình Student cực nhẹ thay vì mô hình lớn. Không có cơ chế tổng hợp cục bộ (Local aggregation) hay cụm (Cluster). |
| **4. Dữ liệu & Hàm mục tiêu FL** | **Dữ liệu:** Tập trung giải quyết dữ liệu Non-IID (phân phối không đồng nhất).<br>**Thuật toán:** Dùng FedAvg để tổng hợp trên server.<br>**Hàm mất mát (Local Loss):** Tổng hợp từ Task loss $\mathcal{L}_{t,i}^{t}$, Distillation loss $\mathcal{L}_{t,i}^{d}$, và Hidden loss $\mathcal{L}_{t,i}^{h}$ thành: $\mathcal{L}_{t,i} = \mathcal{L}_{t,i}^{d} + \mathcal{L}_{t,i}^{h} + \mathcal{L}_{t,i}^{t}$. | **Dữ liệu:** Xử lý tốt cả hai trường hợp IID và Non-IID.<br>**Thuật toán:** Tổng hợp tỷ lệ thuận (tương tự cơ chế của FedAvg).<br>**Hàm mất mát (Local Loss):** Tương tự, kết hợp 3 thành phần: $F_{(s,i)} = F_{(s,i)}^{d} + F_{(s,i)}^{h} + F_{(s,i)}^{t}$. |
| **5. Cơ chế Nén Truyền thông** | **Knowledge Distillation + SVD Gradient Compression:** Giảm thiểu bằng mô hình Student, sau đó phân tích giá trị suy biến (SVD) để xấp xỉ/nén các ma trận gradient trước khi gửi đi. Sử dụng độ chính xác động (dynamic precision) thay đổi theo từng giai đoạn huấn luyện. | **Knowledge Distillation + Ternary Quantization:** Dùng mô hình Student, sau đó lượng tử hóa (Quantization) các trọng số của mô hình theo từng lớp thành mạng trọng số bậc ba (chỉ lấy 3 giá trị rời rạc là -1, 0, 1) để nén cực sâu dung lượng tham số truyền tải. |
| **6. Điểm khác biệt (Phương pháp & Usecase)** | **Đặc thù:** Nén trực tiếp trên *Gradient* (độ dốc) bằng SVD. Phù hợp cho mạng Cross-silo (nơi client có tài nguyên khá tốt).<br>**Usecase thử nghiệm:** Xử lý ngôn ngữ tự nhiên (NLP) như Đề xuất tin tức (MIND), Khai thác thực thể y tế (CADEC) và Phân loại văn bản (SMM4H). | **Đặc thù:** Nén trực tiếp trên *Trọng số mô hình* (Weights) bằng Ternary Quantization. Tối ưu cực mạnh cho thiết bị giới hạn băng thông.<br>**Usecase thử nghiệm:** Thị giác máy tính (Computer Vision) với bài toán phân loại hình ảnh trên bộ dữ liệu MNIST và CIFAR10. => Hợp với AUV |

# Multi-Objective Optimization FL

### Đặc điểm chung

Dù ứng dụng và thuật toán tối ưu khác nhau, cả ba nghiên cứu đều cùng giải quyết những "nỗi đau" lớn nhất của FL trong thực tế:

* **Khắc phục dữ liệu Non-IID:** Cả ba bài đều tìm cách giải quyết thách thức khi dữ liệu trên các thiết bị client có phân phối thống kê không đồng nhất (Non-IID).
* **Tối ưu hóa tài nguyên và hiệu năng:** Các bài báo đều hướng tới việc giảm tải chi phí truyền thông, giới hạn năng lượng tiêu thụ hoặc giảm độ trễ xử lý.
* **Chiến lược tham gia chọn lọc:** Thay vì yêu cầu tất cả các client đều phải tham gia (full-participation) một cách cứng nhắc, cả ba hệ thống đều sử dụng cơ chế chọn lọc linh hoạt: chọn lọc tác tử, gom cụm client, hoặc điều phối liên kết UAV-EV linh hoạt.

### Bóc tách cụ thể khái niệm "Multi" trong từng hệ thống

Dưới đây là lời giải đáp chi tiết cho câu hỏi "Multi" cụ thể là gì, Task là gì và Agent là gì trong từng bài báo:

#### 1. Bài Multi-Agent: FedMarl
* **Khái niệm cốt lõi:** Học tăng cường Đa tác tử (Multi-Agent Reinforcement Learning - MARL).
* **Agent (Tác tử) là gì?** Trong hệ thống này, các tác tử là các mạng nơ-ron đa lớp (MLP) nhỏ được đặt tại máy chủ trung tâm (Central Server).
* **Vai trò của Agent:** Mỗi một Agent sẽ đại diện và ra quyết định thay cho **một thiết bị Client**. Agent sẽ quan sát trạng thái của client đó (kích thước dữ liệu, độ trễ, tổn hao đào tạo) để đưa ra hành động (Action) là dạng nhị phân 0 hoặc 1. Hành động này quyết định việc client đó có bị loại sớm hay được phép tiếp tục tham gia vào vòng huấn luyện hiện tại, nhằm cân bằng giữa độ chính xác và độ trễ.

#### 2. Bài Multi-Task (Clustered FMTL)
* **Khái niệm cốt lõi:** Học đa nhiệm liên kết (Federated Multi-Task Learning) kết hợp phân cụm (Clustering). 
* **Task (Nhiệm vụ) là gì?** Trong bài báo này, khái niệm Task mang tính trừu tượng: **mỗi client (hoặc một cụm các client có phân phối dữ liệu giống nhau) được coi là một Task độc lập**.
* **Bản chất của Multi-Task:** Thay vì cố gắng tạo ra một mô hình chung (Global model) duy nhất cho tất cả, hệ thống huấn luyện các mô hình cá nhân hóa cho từng cụm. Quá trình "Multi" ở đây là việc tối ưu hóa song song các mô hình cụm này, đồng thời sử dụng thuật toán tối ưu hóa cận kề (proximal optimization) để chia sẻ kiến thức chéo và học hỏi lẫn nhau giữa các nhóm.

#### 3. Bài Multi-Task (UAV Swarm)
* **Khái niệm cốt lõi:** Học đa nhiệm với cơ chế chia sẻ kiến thức động dựa trên mức độ thân thiết (Task Affinity).
* **Task (Nhiệm vụ) là gì?** Khác hoàn toàn với bài trên, Task ở đây là **các mục tiêu suy luận Machine Learning thực tế, đa dạng** chạy trên cùng một bộ dữ liệu hình ảnh do UAV thu thập. Ví dụ trên tập dữ liệu FLAME, các Task cụ thể là: nhận diện khung cảnh (rừng, hồ nước, tuyết) và giám sát cháy rừng (có lửa hay không có lửa).
* **Bản chất của Multi-Task:** Các Task này chia sẻ chung một bộ trích xuất đặc trưng hình ảnh (Feature Extractor) ở các lớp đầu, nhưng có các lớp dự đoán (Predictor) riêng biệt ở cuối mô hình. Hệ thống sẽ đánh giá độ tương đồng giữa các nhiệm vụ để quyết định mức độ chia sẻ trọng số nhằm tránh hiện tượng xung đột (negative transfer).


## Bài báo 1: FedMarl (A Multi-Agent Reinforcement Learning Approach for Efficient Client Selection in Federated Learning)

**1. Dữ liệu & Hàm mục tiêu FL**

* **Giả định phân bố dữ liệu:** Bài báo tập trung giải quyết các thách thức trong môi trường dữ liệu **Non-IID** (không độc lập và phân phối đồng nhất). Trong các thiết lập thực nghiệm, dữ liệu Non-IID được mô phỏng bằng cách gán cho mỗi client 80% dữ liệu thuộc về một nhãn (label) ngẫu nhiên duy nhất, 20% còn lại được lấy mẫu đồng đều từ các nhãn khác. Kích thước tập dữ liệu trên mỗi client cũng không đồng đều và tuân theo phân phối power law.
* **Thuật toán cốt lõi:** Cơ sở tổng hợp mô hình vẫn dựa trên cơ chế trung bình trọng số của **FedAvg**. Tuy nhiên, khâu "chọn lọc client" (client selection) ngẫu nhiên truyền thống được thay thế hoàn toàn bằng thuật toán Học tăng cường Đa tác tử (MARL). Quá trình thực nghiệm đã so sánh hiệu năng trực tiếp với các thuật toán nền tảng khác như FedAvg, FedProx, FedNova, HeteroFL, Oort và CS.
* **Hàm mất mát cục bộ (Local Loss):** Được tối ưu dựa trên hàm mục tiêu thay thế (surrogate objective function) $F_n(\cdot)$ tại từng thiết bị. Điểm đặc biệt của bài báo là giới thiệu khái niệm **"probing loss"** (tổn hao thăm dò) ký hiệu là $L_n^t$. Đây là giá trị loss thu được chỉ sau một epoch huấn luyện đầu tiên (probing training), dùng làm thước đo mức độ chệch hướng (bias) của dữ liệu cục bộ so với mô hình toàn cục, từ đó giúp loại sớm các client có chất lượng kém.
* **Hàm mục tiêu toàn cục (Global Objective):** Bài toán không chỉ đơn thuần là giảm loss mô hình, mà là tối ưu hóa hệ thống FL đa mục tiêu: Tối đa hóa độ chính xác kiểm thử $Acc(T)$, tối thiểu hóa tổng độ trễ xử lý $\sum H_t$, và tối thiểu hóa chi phí truyền thông $\sum B_t$.
    $$\max_{A} \mathbb{E} [w_1 Acc(T) - w_2 \sum_{t \in T} H_t - w_3 \sum_{t \in T} B_t]$$
    *(Trong đó $A = [a_n^t]$ là ma trận quyết định chọn client, các tham số $w_1, w_2, w_3$ là trọng số ưu tiên cho từng mục tiêu do nhà phát triển thiết lập)*.

---

**2. Phương pháp Multi ở đây cụ thể là gì, biểu diễn cụ thể, công thức**

* **Định nghĩa phương pháp Multi:** Cốt lõi của hệ thống là **Multi-Agent Reinforcement Learning (MARL)** - Học tăng cường Đa tác tử hợp tác. Bài báo sử dụng kiến trúc Mạng Phân rã Giá trị (Value Decomposition Network - VDN) để huấn luyện tập thể các tác tử này.
* **Biểu diễn cụ thể:** Có một tập hợp gồm $N$ tác tử (MARL agents) được triển khai trên máy chủ trung tâm (Central Server). Mỗi tác tử được cấu tạo bởi một mạng nơ-ron Multi-layer Perceptron (MLP) hai lớp và sẽ chịu trách nhiệm ra quyết định cho **một thiết bị client tương ứng**.
    Tại vòng huấn luyện $t$, tác tử $n$ quan sát trạng thái hệ thống $s_n^t$ và đưa ra hành động nhị phân $a_n^t \in \{0, 1\}$. Nếu $a_n^t = 1$, client được phép hoàn thành huấn luyện cục bộ và tải trọng số lên server; nếu $a_n^t = 0$, client bị loại sớm (early rejected) ngay sau bước "thăm dò" (probing) để tiết kiệm tài nguyên mạng và thời gian.
* **Các công thức tính toán cốt lõi:**
    * **Trạng thái tác tử (State):** Véc-tơ trạng thái đầu vào của tác tử $n$ tại vòng $t$ kết hợp 6 yếu tố: tổn hao thăm dò $L_n^t$, độ trễ của bước thăm dò $H_{t,n}^p$, lịch sử độ trễ tải lên $H_{t,n}^u$, chi phí băng thông $B_n^t$, dung lượng dữ liệu cục bộ $D_n$, và chỉ số vòng $t$.
        $$s_n^t = [L_n^t, H_{t,n}^p, H_{t,n}^u, B_n^t, D_n, t]$$
    * **Độ trễ xử lý của toàn vòng (Processing Latency):** Thời gian của vòng $t$ bằng thời gian đợi tất cả client hoàn thành "thăm dò", cộng với thời gian đợi các client *được chọn* ($a_n^t=1$) hoàn thành phần huấn luyện còn lại ($H_{t,n}^{rest}$) và tải mô hình lên ($H_{t,n}^u$).
        $$H_t = \max_{1 \le n \le N} (H_{t,n}^p) + \max_{n: 1 \le n \le N, a_n^t = 1} (H_{t,n}^{rest} + H_{t,n}^u)$$
    * **Phần thưởng chung (Team Reward):** Các tác tử nhận chung một phần thưởng $r_t$ dựa trên việc cải thiện độ chính xác $U(Acc(t))$, trừ đi hình phạt về độ trễ và chi phí truyền thông.
        $$r_t = w_1 [U(Acc(t)) - U(Acc(t-1))] - w_2 H_t - w_3 B_t$$
    * **Hàm Q-function phân rã (VDN):** Hàm Q tổng hợp của toàn hệ thống là phép cộng tuyến tính của các hàm Q từ từng tác tử cá nhân, giúp tối ưu hóa lợi ích tập thể thay vì lợi ích cục bộ.
        $$Q_{tot}(s_t, a_t) = \sum_n Q_n^\theta(s_n^t, a_n^t)$$


### Bài báo 2: CFMTL/SCFMTL (Clustered Federated Multitask Learning on Non-IID Data With Enhanced Privacy)

### 1. Dữ liệu & Hàm mục tiêu FL

* **Giả định phân bố dữ liệu:** Bài báo tập trung hoàn toàn vào việc giải quyết thách thức của dữ liệu **Non-IID** (không đồng nhất). Tình trạng này được mô tả là dữ liệu thô trên mỗi thiết bị tương ứng với người dùng, vị trí địa lý hoặc khung thời gian cụ thể, dẫn đến sự khác biệt lớn về phân phối thống kê.
* **Thuật toán Federated Learning cốt lõi:** Hệ thống sử dụng phương pháp **Học đa nhiệm liên kết kết hợp phân cụm (Clustered Federated Multi-Task Learning - CFMTL)**. Cơ chế tổng hợp trong từng cụm (Intra-cluster aggregation) có nét tương đồng với **FedAvg**, nhưng điểm khác biệt lớn nhất là việc sử dụng **Tối ưu hóa cận kề (Proximal Optimization)** để thực hiện việc học hỏi tri thức lẫn nhau giữa các cụm khác nhau thay vì chỉ có một mô hình toàn cục.
* **Hàm mất mát cục bộ (Local Loss):** Mỗi client $i$ thuộc một nhóm (cluster) $t$ sẽ thực hiện huấn luyện cục bộ dựa trên mô hình nhóm $G_t$ được nhận từ server.
    $$w_i^{r+1} = G_t^r - \eta \nabla \sum_{j=1}^{n_i} F(y_{i,j}, f(x_{i,j}, G_t^r))$$
    *(Trong đó $w_i^{r+1}$ là trọng số cập nhật của client, $G_t^r$ là mô hình của nhóm $t$ tại vòng $r$)*.
* **Hàm mục tiêu toàn cục (Global Objective):** Mục tiêu là tối thiểu hóa tổng tổn hao huấn luyện trung bình của tất cả các mô hình nhóm trên toàn bộ client, đồng thời có thêm số hạng phạt (penalty term) $g(G)$ để điều chỉnh mối quan hệ giữa các nhiệm vụ:
    $$\min_{K, G} \left\{ \sum_{t=1}^l \sum_{i=1}^m k_{t,i} \sum_{j=1}^{n_i} F(y_{i,j}, f(x_{i,j}, G_t)) + \hat{\lambda} g(G) \right\}$$
    *(Trong đó $l$ là số lượng cụm, $m$ là số lượng client, và $k_{t,i}$ là biến nhị phân xác định client $i$ có thuộc nhóm $t$ hay không)*.

### 2. Phương pháp "Multi" cụ thể là gì?

* **Khái niệm Multi:** Ở đây là **Multi-Task Learning (MTL - Học đa nhiệm)**.
* **Định nghĩa "Task" (Nhiệm vụ):** Trong kiến trúc CFMTL, mỗi cụm (cluster) gồm các client có phân phối dữ liệu tương đồng được coi là một **nhiệm vụ (Task)** riêng biệt. Thay vì ép tất cả client học chung một mô hình (Single-task), hệ thống huấn luyện đồng thời nhiều "mô hình nhóm" cá nhân hóa cho từng cụm.
* **Biểu diễn cụ thể:** Bài báo áp dụng cơ chế **Chia sẻ tham số mềm (Soft parameter sharing)**. Theo cách này, mỗi nhiệm vụ (cụm) có các tham số riêng, nhưng khoảng cách giữa các tham số của các nhiệm vụ khác nhau được ràng buộc để chúng có thể học hỏi những đặc điểm chung từ nhau thông qua một thuật toán tối ưu hóa.
* **Công thức biểu diễn việc học đa nhiệm:**
    Việc chuyển giao tri thức giữa các nhiệm vụ (liên cụm) được thực hiện thông qua bài toán tối ưu hóa cận kề, trong đó mô hình của nhóm $t$ sẽ được cập nhật dựa trên khoảng cách với các mô hình của nhóm $j$ khác:
    $$Prox(G_{new}) = \arg \min_{G} \frac{1}{2\eta} \|G_{new} - G_{old}\|_F^2 + L \sum_{j=1, j \neq t}^l \lambda_{t,j} g(G_{t,new}; G_{j,old})$$
    Trong đó, mối quan hệ/độ tương đồng giữa các nhiệm vụ (groups) được biểu diễn bằng chỉ số $\lambda_{t,j}$:
    $$\lambda_{t,j} = e^{-dist_{t,j}}$$
    *(Với $dist_{t,j}$ là khoảng cách Euclide giữa các tham số của mô hình nhóm $t$ và nhóm $j$)*. Công thức này cho phép các nhiệm vụ có đặc điểm gần giống nhau sẽ chia sẻ nhiều kiến thức hơn, trong khi các nhiệm vụ khác biệt sẽ ít ảnh hưởng đến nhau hơn.

Dưới đây là phân tích chi tiết cho bài báo thứ ba theo yêu cầu của bạn.

### Bài báo 3: UAV Swarm Multi-Task FL (Efficient UAV Swarm-Based Multi-Task Federated Learning with Dynamic Task Knowledge Sharing)

### 1. Dữ liệu & Hàm mục tiêu FL

* **Giả định phân bố dữ liệu:** Hệ thống được thiết kế để xử lý linh hoạt cả dữ liệu **IID và Non-IID**. Để mô phỏng các điều kiện thực tế khắc nghiệt, bài báo sử dụng phân phối Dirichlet (với các tham số $\alpha_1, \alpha_2$) để kiểm soát sự mất cân bằng về số lượng mẫu và phân phối nhãn dữ liệu trên mỗi UAV.
* **Thuật toán Federated Learning cốt lõi:** Thuật toán dựa trên khung của **FedAvg** nhưng được tùy biến cho môi trường đa nhiệm. Điểm cốt lõi là việc tách mô hình thành bộ trích xuất đặc trưng chung và bộ dự đoán riêng cho từng nhiệm vụ, sau đó thực hiện tổng hợp gradient có chọn lọc thay vì trung bình hóa toàn bộ.
* **Hàm mất mát cục bộ (Local Loss):** Được tính trên mỗi UAV $n$ cho một nhiệm vụ $m$ cụ thể:   $F_{m,n}(w_{m,n}) = \frac{1}{D_n} \sum_{(x,y) \in \mathcal{D}_n} f_{m,n}(x,y; w_{m,n})$.
    Trong đó, tham số mô hình $w_{m,n}$ bao gồm bộ trích xuất đặc trưng hình ảnh $w_{m,n}^s$ và bộ dự đoán $w_{m,n}^u$.
* **Hàm mục tiêu toàn cục (Global Objective):** Bài báo hướng tới việc tối ưu hóa hiệu năng trung bình của tất cả $M$ nhiệm vụ đồng thời:
    $F(w) = \frac{1}{M} \sum_{m=1}^M F_m(w_m)$.
    Với $F_m(w_m)$ là tổn hao toàn cục của nhiệm vụ $m$ trên toàn bộ mạng lưới. Ngoài ra, bài báo còn kết hợp tối ưu hóa năng lượng thông qua lý thuyết **Lyapunov** để cân bằng giữa hiệu năng và thời gian sử dụng pin của UAV.

### 2. Phương pháp "Multi" cụ thể là gì?


* **Khái niệm Multi:** Ở đây là **Multi-Task Federated Learning (MTFL)** với cơ chế **Chia sẻ tri thức nhiệm vụ động (Dynamic Task Knowledge Sharing)**.
* **Định nghĩa "Task" (Nhiệm vụ):** Khác với bài báo trước (nơi coi cụm client là Task), bài này định nghĩa Task là các nhiệm vụ phân tích dữ liệu khác nhau trên cùng một dòng dữ liệu hình ảnh. Ví dụ thực tế từ bộ dữ liệu FLAME bao gồm: (1) Nhận diện cảnh vật (rừng, hồ, tuyết) và (2) Giám sát hỏa hoạn (phát hiện lửa).
* **Biểu diễn cụ thể:** Hệ thống sử dụng kiến trúc mạng nơ-ron tách lớp. Các nhiệm vụ khác nhau chia sẻ chung kiến trúc bộ trích xuất đặc trưng (Feature Extractor) ở các lớp đầu để tận dụng thông tin thị giác chung, nhưng sở hữu các bộ dự đoán (Predictor) riêng biệt cho mục tiêu cụ thể.
* **Công thức và cơ chế chia sẻ tri thức:**
    * **Chỉ số Thân thiết Nhiệm vụ (Task Affinity - TA):** Đo lường mức độ ảnh hưởng của nhiệm vụ $i$ đối với nhiệm vụ $j$. Nếu việc sử dụng gradient của nhiệm vụ $i$ giúp giảm loss cho nhiệm vụ $j$, chúng được coi là có quan hệ tích cực:
        $\Theta_{i \to j}^t = 1 - \frac{L_j(w_{j,t}^s, G_{i,t}, G_{j,t})}{L_j(w_{j,t}^s, G_{j,t})}$.
    * **Cập nhật trích xuất đặc trưng (Knowledge Sharing):** Lớp trích xuất đặc trưng của nhiệm vụ $m$ sẽ được cập nhật dựa trên gradient tổng hợp từ chính nó và các nhiệm vụ khác có độ thân thiết dương ($S_{m,t}$):
        $w_{m,t+1}^s = w_{m,t}^s - \eta \frac{\sum_{i \in S_{m,t}} D_{i,t} G_{i,t}^s}{\sum_{i \in S_{m,t}} D_{i,t}}$.
    * **Cơ chế Chú ý Nhiệm vụ (Task Attention):** Sử dụng giá trị **Task Shapley Value (TSV)** để đo lường đóng góp của từng nhiệm vụ vào việc cải thiện hiệu năng chung, từ đó gán trọng số ưu tiên $\alpha_m^t$ nhằm phân bổ tài nguyên (băng thông, năng lượng) tối ưu cho các nhiệm vụ quan trọng hoặc khó học hơn.
