# BÁO CÁO ĐỀ XUẤT NGHIÊN CỨU: TỐI ƯU HÓA HỌC LIÊN KẾT PHÂN CỤM DỰA TRÊN TRI THỨC TRONG MÔI TRƯỜNG INTERNET VẠN VẬT DƯỚI NƯỚC (IoUT)

# THÊM REFERENCE

## ABSTRACT
Mạng lưới vạn vật dưới nước (IoUT) đang đối mặt với những thách thức nghiêm trọng về băng thông thấp, độ trễ lan truyền cao và nguồn năng lượng hạn chế của các thiết bị tự hành (AUV). Mặc dù Học liên kết (Federated Learning - FL) cho phép huấn luyện mô hình mà không cần chia sẻ dữ liệu thô, các kiến trúc hình sao truyền thống thường thất bại trong việc xử lý tỷ lệ mất gói cao và tính chất không đồng nhất (Non-IID) của dữ liệu hải dương. Báo cáo này đề xuất một khung mô hình cải tiến kết hợp **Học liên kết phân cụm (Clustered FL)** dựa trên khoảng cách **Earth Mover’s Distance (EMD)** và **Chưng cất tri thức (Knowledge Distillation)**. Hệ thống sử dụng thuật toán lai giữa **Học tăng cường (RL)** và **Tối ưu hóa bầy đàn Manta Ray (MRFO)** để điều phối tài nguyên và lựa chọn nút Fog động. Mục tiêu tối thượng là tối ưu hóa đồng thời độ chính xác, năng lượng và độ trễ dưới các ràng buộc vật lý thực tế của kênh truyền âm thanh.


## I. RESEARCH GAPS (LỖ HỔNG NGHIÊN CỨU)

Bài báo cơ sở: Efficient Asynchronous Federated Learning for AUV Swarm

### 1. Kênh truyền lý tưởng và Bỏ qua tỷ lệ mất gói (Packet Loss)
* **Lỗ hổng:** Các nghiên cứu trước đây thường giả định khoảng cách truyền tin cố định và thông lượng lý thuyết, bỏ qua ảnh hưởng của Fading đa đường và hiệu ứng Doppler trong môi trường âm thanh dưới nước. Điều này dẫn đến việc đánh giá sai lệch hiệu suất thực tế khi tỷ lệ rớt gói ($PLR$) tăng cao.
* **Giải pháp đề xuất:** Chuyển đổi sang kiến trúc **Clustered FL**. Các AUV gần nhau sẽ tự tổ chức thành các cụm dựa trên tính khả thi của liên kết (SNR). Việc tổng hợp tại các nút Fog (AUV trung tâm cụm) giúp giảm suy hao đường truyền đa chặng và hạn chế tác động của rớt mạng cục bộ.

### 2. Mô phỏng năng lượng không toàn vẹn
* **Lỗ hổng:** Các mô hình hiện tại thường chỉ đưa ra ràng buộc năng lượng cho từng vòng lặp riêng lẻ mà thiếu biến trạng thái về dung lượng pin tổng thể, dẫn đến rủi ro cạn kiệt năng lượng đột ngột (dead nodes).
* **Giải pháp đề xuất:** Tích hợp **Battery-Aware Optimization**. Đưa biến trạng thái pin $B_i^{(t)}$ của từng AUV $i$ vào mô hình Markov (MDP). Thuật toán RL sẽ thiết kế hàm phần thưởng $r^{t+1}$ để phạt nặng các hành động gây rủi ro sập nguồn, đồng thời ưu tiên điều phối tải tính toán sang các AUV có mức năng lượng cao.

### 3. Nút thắt cổ chai tại Leader AUV (Bottleneck)
* **Lỗ hổng:** Trong cấu trúc tập trung, Leader AUV phải gánh chịu toàn bộ tải trọng nhận và tổng hợp tham số từ tất cả các nút con, gây ra hiện tượng tắc nghẽn và tăng độ trễ hệ thống.
* **Giải pháp đề xuất:** Sử dụng kiến trúc 3 tầng (**Sensor - Fog - Surface**). Việc phân cấp tổng hợp tại các Cluster Head (AUV Fog) giúp giảm tải trọng tính toán và truyền tin cho trạm nổi (Surface Gateway).

### 4. Giả định dữ liệu đồng nhất (IID) phi thực tế
* **Lổng hổng:** Việc sử dụng các bộ dữ liệu chuẩn như MNIST với giả định phân phối dữ liệu giống nhau giữa các AUV là không thực tế, vì dữ liệu thu thập tại các tọa độ khác nhau dưới đại dương luôn có tính Non-IID cực cao.
* **Giải pháp đề xuất:** Nhóm các AUV có phân phối dữ liệu tương đồng bằng khoảng cách **EMD**. Sử dụng **Proximal Optimization** để ép các tham số mô hình cụm không phân kỳ quá xa, giúp giải quyết triệt để tính chất Non-IID của dữ liệu sonar và ảnh quang học dưới nước.

### 5. Giới hạn cơ chế giảm tải (Knowledge Distillation)
* **Lỗ hổng:** Các cơ chế hiện tại chỉ dừng lại ở việc gửi hoặc đóng băng tham số dựa trên ngưỡng gradient đơn giản, bỏ qua tiềm năng nén mô hình của kỹ thuật chưng cất tri thức.
* **Giải pháp đề xuất:** Nâng cấp hệ thống với **Federated Knowledge Distillation (FedKD)** kết hợp **Ternary Compression**. Thay vì truyền toàn bộ trọng số, các AUV chỉ gửi các dự đoán xác suất mềm (soft-logits) đã được lượng tử hóa về mức {-1, 0, 1}, giúp giảm dung lượng gói tin hàng vạn lần.

## II. Các nghiên cứu liên quan:

### 1. Decentralized Federated Learning

* **Bài toán (Khử trung tâm & Vượt rào cản mạng):** Cả hai đều loại bỏ hoàn toàn máy chủ trung tâm để tránh "thắt cổ chai" và rủi ro hỏng hóc đơn điểm. Trọng tâm là duy trì việc học máy trong điều kiện mạng khắc nghiệt (đứt kết nối do rớt gói hoặc topology thay đổi do di chuyển) với băng thông cực kỳ hạn chế.
* **Hàm mục tiêu (Tối ưu hóa toàn cục):** Dù mạng lưới đứt gãy hay biến động, mục tiêu tối thượng không đổi là tìm ra bộ tham số $x^*$ giúp cực tiểu hóa hàm mất mát chung của toàn hệ thống (được tính bằng trung bình cộng các hàm mất mát cục bộ trên từng thiết bị):
    $$x^{*} = \arg\min_{x \in \mathbb{R}^d} \frac{1}{N} \sum_{i=1}^{N} f_i(x)$$
* **Phương pháp (Sự tất yếu của Gossip/Consensus):** Chính hàm mục tiêu (đòi hỏi tri thức toàn cục) đặt trong bối cảnh bài toán (không có máy chủ gom dữ liệu) đã **bắt buộc** cả hai nghiên cứu phải dùng giao tiếp ngang hàng (Gossip-based). Các node phải liên tục "trộn" mô hình với hàng xóm lân cận bằng các thuật toán đồng thuận (Consensus), kết hợp thêm cơ chế giảm xóc (như $\gamma_t$ suy giảm hoặc CECA) để ép mô hình hội tụ trên dữ liệu Non-IID.

* **Vấn đề:** Các nghiên cứu DFL tập trung cải thiện độ chính xác của mô hình, tuy nhiên công suất, năng lượng, số lần giao tiếp lớn, không phù hợp với môi trường dưới nước.

### 2. Clustered Federated Learning

Đặc điểm chung

* **Bài toán (Sự cạn kiệt tài nguyên):** Môi trường làm việc khắc nghiệt (dưới nước hoặc mạng IoT mật độ cao) khiến năng lượng pin và băng thông bị giới hạn nghiêm ngặt, việc truyền dữ liệu thô đi xa về một máy chủ trung tâm là hoàn toàn bất thi.
* **Hàm mục tiêu (Tối ưu hóa đa biến):** Không chỉ theo đuổi độ chính xác của AI, hệ thống bắt buộc phải giải bài toán đánh đổi (trade-off): Cân bằng giữa chất lượng học máy/định tuyến với việc tiết kiệm năng lượng và giảm độ trễ nhằm kéo dài tối đa tuổi thọ của mạng lưới.
* **Phương pháp (Phân cụm & Trí tuệ biên):** Bị ép buộc bởi bài toán và hàm mục tiêu trên, cả 3 bài đều phải chuyển sang cấu trúc mạng Phân cấp/Phân cụm (Clustering/Hierarchical). Các thiết bị sẽ tự xử lý dữ liệu cục bộ, gom thành nhóm, bầu ra "thủ lĩnh" (Cluster Head / Fog Node) để nén thông tin rồi mới gửi đi, giúp triệt tiêu hoàn toàn nút thắt cổ chai truyền thông.

### 3. Knowledge Distillation in FL

* **Mô hình Teacher cá nhân hóa (Personalized Teacher):** Mỗi client đều sở hữu một mô hình Teacher lớn, được huấn luyện và tối ưu hóa dựa trên đặc điểm dữ liệu cục bộ riêng biệt của chính client đó nhằm mang lại khả năng cá nhân hóa cao.
* **Mô hình Student dùng chung (Shared Student):** Tồn tại một mô hình Student nhỏ gọn hơn, đóng vai trò làm trung gian thống nhất, được chia sẻ và cập nhật toàn cục giữa tất cả các client.
* **Trích xuất kiến thức thích ứng (Adaptive Distillation):** Cường độ truyền đạt kiến thức giữa Teacher và Student không cố định mà được điều chỉnh tự động dựa trên mức độ chính xác (prediction correctness) của các dự đoán.
* **Học từ các trạng thái ẩn (Hidden States):** Để tối ưu hóa hiệu năng, mô hình Student không chỉ học từ kết quả đầu ra (soft labels) mà còn tiếp thu các đặc trưng sâu thông qua trạng thái ẩn và bản đồ chú ý (attention heatmaps) của mô hình Teacher.

### 4. Multi-Objective Optimization FL

* **Multi-Agent**: Mỗi một Agent sẽ đại diện và ra quyết định thay cho một thiết bị Client. Agent sẽ quan sát trạng thái của client đó (kích thước dữ liệu, độ trễ, tổn hao đào tạo) để đưa ra hành động (Action)

* **Multi-Task**: 
    * Hệ thống huấn luyện các mô hình cá nhân hóa cho từng cụm và tối ưu hóa song song các mô hình cụm này
    * Tối ưu hóa đa mục tiêu (Multi-objective Optimization): Đa mục tiêu suy luận Machine Learning thực tế, đa dạng chạy trên cùng một bộ dữ liệu hình ảnh do UAV thu thập. Ví dụ trên tập dữ liệu FLAME, các Task cụ thể là: nhận diện khung cảnh (rừng, hồ nước, tuyết) và giám sát cháy rừng (có lửa hay không có lửa).

* **Vấn đề**: Khá sát thực tế, nhưng thời hạn đồ án và giới hạn phần cứng không đủ để tiến hành

## III. Xây dựng bài toán

Dự kiến sẽ biểu diễn toán học nhằm xây dựng các mô hình:

### 1. Mô hình truyền tin âm thanh thực tế (Channel and Energy Model)

Môi trường vật lý dưới nước được mô hình hóa chặt chẽ theo chuẩn IEEE để phản ánh chính xác chi phí năng lượng và độ tin cậy của liên kết. Luồng mô hình hóa đi theo trình tự: **Vật lý âm thanh** $\rightarrow$ **Tốc độ mạng** $\rightarrow$ **Năng lượng điện tiêu thụ** $\rightarrow$ **Đồ thị kết nối khả thi**.

#### a. Vật lý âm thanh: Suy hao và Nhiễu nền

* **Suy hao truyền dẫn (Transmission Loss - $TL$):**
    Tính toán theo mô hình kết hợp lan truyền hình học và hấp thụ nhiệt theo công thức Thorp:
    $$TL(d, f) = 10 k \log_{10}(d) + \alpha(f) \frac{d}{1000}$$
    Trong đó $d$ là khoảng cách (m), $f$ là tần số sóng mang (kHz), $k$ là hệ số lan truyền (thường là 1.5). Hệ số hấp thụ $\alpha(f)$ được tính bằng công thức Thorp:
    $$\alpha(f) = \frac{0.11f^2}{1+f^2} + \frac{44f^2}{4100+f^2} + 2.75 \times 10^{-4}f^2 + 0.003 \quad \text{(dB/km)}$$

* **Nhiễu nền đại dương (Ambient Noise - $NL$):**
    Mật độ phổ công suất nhiễu $N_0(f)$ là tổng của 4 thành phần theo mô hình Wenz: nhiễu do xoáy ($N_{turb}$), tàu bè ($N_{ship}$), gió ($N_{wind}$) và nhiệt ($N_{therm}$). Mức nhiễu tổng hợp trong băng thông $B$ (Hz) là:
    $$NL(f, B) = N_0(f) + 10 \log_{10}(B) \quad \text{(dB re } \mu\text{Pa)}$$

#### b. Mức nguồn phát tối thiểu và Tốc độ truyền dữ liệu

* **Mức nguồn phát tối thiểu ($SL^{min}$):**
    Để đảm bảo SNR tại máy thu đạt ngưỡng mục tiêu $\gamma_{tgt}$ (dB), mức nguồn phát từ AUV $u$ đến AUV $v$ phải thỏa mãn phương trình cân bằng liên kết (link budget):
    $$SL_u^{min}(u, v) = \gamma_{tgt} + TL(d_{uv}, f) + NL(f, B) + IL$$
    Trong đó $IL$ là tổn hao chèn (Insertion Loss) do nhiễu xuyên âm và hiệu ứng Doppler. Đây là *lượng năng lượng âm thanh tối thiểu* AUV phải phát đi để đảm bảo liên kết đáng tin cậy.

* **Tốc độ truyền dữ liệu (Data Rate - $R_{uv}$):**
    Biến $R$ được sử dụng tính độ trễ truyền tải $\tau = L/R$ không thể chọn tùy ý — trong môi trường dưới nước, $R$ phụ thuộc trực tiếp vào SNR thực tế của kênh. Áp dụng định lý Shannon, tốc độ truyền dữ liệu tối đa giữa cặp AUV $(u, v)$ khi hệ thống duy trì SNR mục tiêu $\gamma_{tgt}$ là:
    $$R_{uv} = B \log_2\!\left(1 + 10^{\gamma_{tgt}/10}\right) \quad \text{(bps)}$$
    Điều này thể hiện rằng AUV đang thực hiện **điều khiển công suất (power control)**: bằng cách giữ $SL_u^{min}$ đủ để đạt $\gamma_{tgt}$, hệ thống duy trì một tốc độ $R$ ổn định và có thể dự đoán, làm cơ sở cho tính toán độ trễ và năng lượng ở các bước tiếp theo.

#### c. Chuyển đổi Công suất: Từ $SL^{min}$ sang Năng lượng điện tiêu thụ

Trong đề xuất cũ, $E_{tx}$ được sử dụng nhưng chưa chỉ rõ làm sao từ mức nguồn phát $SL^{min}$ (đơn vị dB re $\mu$Pa@1m) ra được số Joules tiêu thụ. Hệ phương trình chuyển đổi được xây dựng qua hai bước:

* **Bước 1 — Công suất âm thanh (Acoustic Power - $P_{ac}$):**
    Chuyển từ mức âm $SL^{min}$ (thang logarithmic) sang công suất bức xạ âm thanh thực tế (Watt):
    $$P_{ac} = \frac{4\pi\, p_{ref}^2}{\rho_w\, c_s}\, 10^{SL^{min}/10} \quad \text{(W)}$$
    Trong đó: $p_{ref} = 10^{-6}$ Pa là áp suất âm tham chiếu dưới nước, $\rho_w \approx 1025\ \text{kg/m}^3$ là mật độ nước biển, và $c_s \approx 1500\ \text{m/s}$ là vận tốc âm thanh.

* **Bước 2 — Công suất điện thực tế tại modem ($P_{tx}$):**
    Do bộ biến đổi điện-âm không có hiệu suất hoàn hảo, công suất điện modem phải tiêu thụ là:
    $$P_{tx} = \frac{P_{ac}}{\eta_{ea}} \quad \text{(W)}$$
    Với $\eta_{ea} \in (0, 1]$ là hiệu suất chuyển đổi điện-âm (electro-acoustic efficiency) của modem. Từ đó, năng lượng tiêu thụ để truyền một gói tin kích thước $L$ (bit) là:
    $$E_{tx}(L, d_{uv}) = \left(\frac{P_{ac}}{\eta_{ea}} + P_{c,tx}\right) \frac{L}{R_{uv}}$$
    Trong đó $P_{c,tx}$ là công suất mạch điện tử (circuit power) và $L/R_{uv}$ là thời gian truyền.

#### d. Đồ thị Khả thi (Feasible Communication Graph - $\mathcal{G}^t$)

Thay vì giả định mọi cặp AUV đều có thể kết nối với nhau, chúng tôi định nghĩa toán học tập hợp các liên kết *vật lý khả thi* tại mỗi vòng lặp $t$:

$$\mathcal{G}^t = \bigl\{(u, v) \;\big|\; SL_u^{min}(u, v) \le SL_{max}\bigr\}$$

Một cạnh $(u, v)$ **chỉ tồn tại** trong $\mathcal{G}^t$ nếu và chỉ nếu mức công suất phát cần thiết không vượt quá giới hạn phần cứng $SL_{max}$ của modem. Đây là **nền tảng toán học** cho thiết kế kiến trúc Clustered FL: chính ràng buộc $\mathcal{G}^t$ làm cho cấu trúc FL hình sao (truyền thẳng mọi AUV về trạm nổi) trở nên bất khả thi về mặt năng lượng khi khoảng cách lớn, từ đó **bắt buộc** hệ thống phải áp dụng kiến trúc phân cụm đa chặng (Clustered Multi-hop FL) để duy trì liên lạc trong phạm vi năng lượng cho phép.


### 2. Xây dựng mô hình Học máy (Thành phần $\text{Loss}_{model}$)

Thay vì dùng một hàm Loss tĩnh với hệ số $\alpha$ cố định, hệ thống áp dụng **Cơ chế Chưng cất Tri thức Thích ứng (Adaptive Knowledge Distillation)** dựa trên **Ref 6 (He et al., 2025)** và **Ref 4 (Jiang et al., 2025)**. Mỗi AUV $i$ vừa sở hữu mô hình **Teacher** cá nhân hóa (lớn, được huấn luyện trên dữ liệu cục bộ), vừa duy trì mô hình **Student** dùng chung (nhỏ gọn, được đồng bộ liên kết).

#### a. Xác suất mềm với tham số Nhiệt độ (Temperature Scaling)

Trước khi tính Loss, đầu ra logits của cả hai mô hình được làm trơn bằng hàm Softmax với tham số nhiệt độ $T > 1$ để tạo ra phân phối xác suất mềm giàu thông tin hơn:

$$p_j = \frac{\exp(z_j / T)}{\sum_k \exp(z_k / T)}, \qquad q_j = \frac{\exp(v_j / T)}{\sum_k \exp(v_k / T)}$$

Trong đó $z_j$ và $v_j$ lần lượt là logits đầu ra của **Teacher** và **Student** cho lớp $j$. Khi $T > 1$, phân phối trở nên phẳng hơn, giúp Student học được tương quan giữa các lớp — thông tin mà nhãn cứng (one-hot) không truyền tải được.

#### b. Các thành phần Hàm Mất mát Thích ứng

Hàm $\text{Loss}_{model}$ (ký hiệu $F_{(s,i)}$ cho Student tại AUV $i$) được phân rã thành 3 thành phần có khả năng tự điều chỉnh trọng số:

**① Task Loss ($F^t$) — Học từ nhãn thực tế:**

Đo lường sai số giữa dự đoán và nhãn gốc (Ground Truth) bằng Cross-Entropy (CE) cho cả hai mô hình:
$$F_{(s,i)}^t = \mathrm{CE}(y_i,\, y_i^s), \qquad F_{(t,i)}^t = \mathrm{CE}(y_i,\, y_i^t)$$
Trong đó $y_i$ là nhãn thực, $y_i^s$ và $y_i^t$ lần lượt là dự đoán của Student và Teacher trên tập dữ liệu cục bộ của AUV $i$.

**② Adaptive Distillation Loss ($F^d$) — Học từ Teacher:**

Sử dụng phân kỳ Kullback-Leibler (KL) để ép phân phối xác suất của Student bám sát Teacher. Điểm đột phá là mẫu số $F_{(t,i)}^t + F_{(s,i)}^t$ đóng vai trò bộ điều chỉnh trọng số tự động: khi cả hai mô hình đều dự đoán sai nhiều (Task Loss cao), trọng số chưng cất tự giảm để tập trung học nhãn thực; khi đã hội tụ tốt, trọng số tăng để khai thác tối đa tri thức mềm từ Teacher:
$$F_{(s,i)}^d = \frac{KL(y_i^t,\, y_i^s)}{F_{(t,i)}^t + F_{(s,i)}^t + \varepsilon}$$

*(Trong đó $\varepsilon \approx 10^{-8}$ là hằng số ổn định số học để tránh chia cho 0 khi cả hai mô hình hội tụ hoàn hảo)*.

**③ Adaptive Hidden Loss ($F^h$) — Học đặc trưng sâu:**

Để Student không chỉ học kết quả đầu ra mà còn bắt chước *tư duy nội tại* của Teacher, ta đối chiếu trực tiếp các **trạng thái ẩn** ($H$) và **bản đồ chú ý** ($A$) giữa hai mô hình qua MSE, cũng được tự chuẩn hóa bởi Task Loss:
$$F_{(s,i)}^h = \frac{\mathrm{MSE}(H_i^t,\, W_i^h H_i^s) + \mathrm{MSE}(A_i^t,\, A_i^s)}{F_{(t,i)}^t + F_{(s,i)}^t}$$
Trong đó $W_i^h$ là ma trận biến đổi tuyến tính (projection matrix) giúp căn chỉnh kích thước vector đặc trưng giữa Teacher và Student khi hai mô hình có kiến trúc khác nhau (heterogeneous models).

#### c. Hàm Mục tiêu Tổng hợp và Cơ chế Lượng tử hóa Ternary (TTQ)

Hàm tổng tổn hao cục bộ mà mô hình Student tại AUV $i$ cần cực tiểu hóa là sự kết hợp thích ứng của 3 thành phần:
$$\text{Loss}_{model} = F_{(s,i)} = F_{(s,i)}^t + F_{(s,i)}^d + F_{(s,i)}^h$$

Gradient để cập nhật bộ tham số (trọng số) $\Theta^s$ của mô hình Student được tính bằng:
$$g_i = \frac{\partial F_{(s,i)}}{\partial \Theta^s}$$

Để vượt qua nút thắt cổ chai của băng thông âm thanh dưới nước, toàn bộ trọng số của mô hình Student sau khi cập nhật không được truyền đi nguyên bản (full-precision) mà phải trải qua cơ chế **Lượng tử hóa Ternary (Ternary Quantization)**[cite: 4]. Quá trình này diễn ra như sau:

*   **Chuẩn hóa và tính ngưỡng:** Trọng số $\Theta^s$ được chuẩn hóa, sau đó hệ thống tính toán một ngưỡng cắt $\Delta$ thích ứng dựa trên độ thưa của mạng:
    $$\Delta = \frac{\rho_k}{D^2} \sum |\Theta_j^s|$$
    *(Trong đó $\rho_k \in (0,1)$ là tỷ lệ sparsity cục bộ của lớp $k$, $D$ là tổng số lớp của mô hình Student — ký hiệu này phân biệt với nhiệt độ $T$ và khoảng cách $d_{uv}$)*[cite: 4].
*   **Ánh xạ Ternary:** Các giá trị trọng số liên tục được ánh xạ về không gian rời rạc qua hàm bước nhảy, tạo ra các ma trận chỉ thị dương ($I_p$) và âm ($I_n$)[cite: 4]:
    $$I_p = \{j \mid \Theta_j^s > \Delta\}, \quad I_n = \{j \mid \Theta_j^s < -\Delta\}$$
*   **Khôi phục trọng số nén:** Trọng số cuối cùng được biểu diễn bằng các hệ số lượng tử hóa $\omega_p, \omega_n$ (được huấn luyện song song) nhân với ma trận chỉ thị:
    $$\tilde{\Theta}^s = \omega_p \times I_p - \omega_n \times I_n$$

Nhờ cơ chế này, thay vì truyền các giá trị float 32-bit, AUV chỉ cần đóng gói các trạng thái rời rạc $\{-1, 0, 1\}$ (tương đương 2-bit), giúp nén kích thước gói tin ($L$) xuống xấp xỉ 16 lần[cite: 4], giảm triệt để năng lượng tiêu thụ $E_{tx}$ và độ trễ truyền tải $\tau_{link}$.

> **Lưu ý phân tầng:** Phân cụm EMD là thao tác của **Lớp Mạng** — diễn ra *trước* khi Lớp AI bắt đầu huấn luyện. Nó được trình bày ở đây vì phụ thuộc vào gradient mô hình để tính khoảng cách, nhưng về mặt kiến trúc hệ thống, nó là đầu vào cho bước gom cụm RL-MRFO ở Section IV.2.

#### d. Phân cụm dựa trên Earth Mover's Distance (EMD) và Proximal Optimization

Dữ liệu hải dương thu thập bởi các AUV mang tính chất Non-IID cực cao (ví dụ: AUV quét rạn san hô sẽ có phân phối dữ liệu khác hoàn toàn AUV quét đáy cát). Nếu áp dụng FedAvg truyền thống, sự khác biệt này gây ra hiện tượng phân kỳ trọng số (weight divergence), làm mô hình suy giảm độ chính xác nghiêm trọng.

Để giải quyết, hệ thống nhóm các AUV có phân phối dữ liệu tương đồng thành các cụm. Khoảng cách **EMD** được sử dụng để đo lường độ lệch trọng số giữa mô hình cục bộ và mô hình cụm lý tưởng, với bài toán tối ưu phân cụm:
$$\arg\min_{K} \sum_{t=1}^{l} \frac{\sum_{i=1}^{m} k_{t,i} n_i}{N} \frac{\|G_t - H_t\|}{\|H_t\|}$$
Trong đó[cite: 4]:
*   $G_t$: Mô hình tổng hợp của cụm $t$.
*   $H_t$: Mô hình lý tưởng nếu được huấn luyện tập trung trên toàn bộ dữ liệu của cụm $t$.
*   $k_{t,i} \in \{0,1\}$: Biến chỉ thị AUV $i$ thuộc cụm $t$.
*   $n_i / N$: Tỷ lệ dữ liệu của AUV $i$ so với toàn mạng.

**Cách thức thực thi:** Vì không thể chia sẻ dữ liệu thô để tìm $H_t$, hệ thống sử dụng thuật toán **Hierarchical Clustering (HC)** với khoảng cách Euclid giữa các tham số mô hình làm thước đo xấp xỉ[cite: 4]. Ma trận tương đồng này sau đó được ánh xạ qua đồ thị liên kết khả thi $\mathcal{G}^t$ (ràng buộc bởi SNR âm thanh). 

Hệ quả là, các AUV không chỉ được nhóm lại vì chúng "gần nhau về mặt vật lý" (đáp ứng SNR), mà còn vì chúng "gần nhau về mặt tri thức" (cùng phân phối dữ liệu). Các mô hình cụm ($G_t$) sau đó sẽ được đồng bộ chéo bằng **Proximal Optimization** nhằm chia sẻ các đặc trưng chung mà không làm mất đi tính cá nhân hóa của từng cụm[cite: 4].

### 3. Xây dựng Mô hình Vật lý & Năng lượng ($E_{round}$)

Thành phần này mô phỏng chi phí sinh tồn của mạng IoUT tại mỗi vòng lặp giao tiếp, được toán học hóa thành các hệ phương trình sau:

#### a. Năng lượng truyền tin ($E_{tx}$)

Các công thức vật lý âm thanh ($SL^{min}$, $P_{ac}$, $P_{tx}$, $R_{uv}$) đã được xây dựng đầy đủ tại **Mục III.1.b–c**. Tóm tắt kết quả cuối cùng:

$$E_{tx}(L;\, u, v) = \left(\frac{P_{ac}}{\eta_{ea}} + P_{c,tx}\right) \frac{L}{R_{uv}}$$

Trong đó $L$ là kích thước gói tin (bits) sau nén Ternary, $R_{uv}$ là tốc độ kênh Shannon phụ thuộc SNR mục tiêu $\gamma_{tgt}$. Nhờ TTQ giảm $L$ xuống $\approx \frac{1}{16}$, năng lượng $E_{tx}$ giảm tương ứng [cite: 3, 6].

#### b. Năng lượng tính toán cục bộ ($E_{comp}$)
Chi phí năng lượng tính toán tại mỗi AUV được nội suy từ số lượng phép toán dấu phẩy động (FLOPs) [cite: 3]. Trong cơ chế Chưng cất tri thức (Knowledge Distillation), số lượng FLOPs $\Phi_i$ của AUV $i$ bao gồm chi phí lan truyền xuôi (forward pass) của mô hình Teacher cộng với chi phí lan truyền xuôi/ngược của mô hình Student.

Năng lượng tính toán được mô hình hóa bằng [cite: 3]:

$$E_{comp, i} = \epsilon_{op} \Phi_i$$

*(Trong đó $\epsilon_{op}$ là hệ số năng lượng tiêu thụ trên mỗi phép tính FLOP của bộ vi xử lý trên AUV)*.

#### c. Chiến lược nén mô hình (Giảm dung lượng $L$)
Nhìn vào phương trình của $E_{tx}$, năng lượng truyền tin tỷ lệ thuận với kích thước gói tin $L$ [cite: 3]. Để giảm $L$, hệ thống áp dụng kỹ thuật Lượng tử hóa Ternary (Trained Ternary Quantization - TTQ) [cite: 6]. 

Bộ tham số của Student model $\Theta^s$ (vốn là các số thực nổi 32-bit) được ánh xạ về không gian rời rạc $\{-1, 0, 1\}$ qua ngưỡng $\Delta$ [cite: 6]. Do chỉ có 3 trạng thái, mỗi trọng số chỉ cần 2-bit để biểu diễn. 

Kích thước gói tin thực tế $L$ truyền đi sau khi nén được tính bằng:

$$L = 2 \times |\Theta^s| + L_{meta}$$

*(Trong đó $|\Theta^s|$ là tổng số lượng tham số của mô hình Student, $L_{meta}$ là phần header/metadata của gói tin truyền thông)*. Việc nén từ 32-bit xuống 2-bit giúp giảm $L$ xấp xỉ 16 lần, qua đó triệt tiêu sự bùng nổ của $E_{tx}$.

#### d. Tổng năng lượng vòng lặp ($E_{round}$)
Tổng năng lượng mà AUV $i$ bị tiêu hao trong một vòng lặp $t$ là tổng hợp của các thành phần trên [cite: 3]:

$$E_{round, i}^{(t)} = E_{comp, i} + E_{tx}(L; i, \text{Fog}_k) + E_{rx}$$

*(Với $E_{rx} = P_{c,rx} \frac{L}{R}$ là năng lượng tiêu hao để nhận các phản hồi từ Cluster Head/Fog)*.

### 4. Xây dựng mô hình Mạng & Độ trễ (Thành phần $\tau_{round}$)

Môi trường âm thanh dưới nước đặc trưng bởi vận tốc lan truyền rất chậm (chỉ khoảng $1500$ m/s, chậm hơn sóng vô tuyến $2 \times 10^5$ lần), khiến độ trễ trở thành một ràng buộc khắt khe. Tổng độ trễ trong một vòng huấn luyện liên kết ($\tau_{round}$) được mô hình hóa qua các thành phần sau:

#### a. Độ trễ tính toán cục bộ (Computation Latency)
Trước khi truyền tin, mỗi AUV $i$ phải thực hiện huấn luyện mô hình Teacher-Student trên tập dữ liệu cục bộ. Thời gian tính toán phụ thuộc vào số lượng mẫu dữ liệu ($N_i$), độ phức tạp của mô hình (số chu kỳ CPU cần cho mỗi mẫu - $c_0$) và năng lực tính toán của vi xử lý trên AUV ($h_i$, tính bằng Hz) [cite: 7]:

$$\tau_{comp, i} = \frac{N_i \cdot c_0}{h_i}$$

#### b. Độ trễ giao tiếp (Communication Latency)
Thời gian để truyền một gói tin tham số $\Theta^s$ đã được nén (kích thước $L$) từ nút $u$ sang nút $v$ được cấu thành từ hai yếu tố: độ trễ truyền tải (Transmission delay) và độ trễ lan truyền (Propagation delay) [cite: 3].

Ký hiệu $\tau_{comm}(u, v)$ là tổng độ trễ giao tiếp trên liên kết $(u, v)$:

$$\tau_{comm}(u, v) = \tau_{trans} + \tau_{prop} = \frac{L}{R_{uv}} + \frac{d_{uv}}{c_s}$$

Trong đó:
*   $L / R_{uv}$: Thời gian bơm gói tin lên kênh truyền, với $R_{uv}$ là tốc độ kênh truyền đã xác định. Nhờ nén Ternary, $L$ giảm mạnh giúp triệt tiêu thành phần này.
*   $d_{uv} / c_s$: Thời gian tín hiệu di chuyển trong nước, với $d_{uv}$ là khoảng cách vật lý và $c_s \approx 1500$ m/s là vận tốc âm thanh dưới nước.

#### c. Thành phần thắt cổ chai (Bottleneck Effect) và Tổng độ trễ
Trong kiến trúc Clustered FL, hệ thống hoạt động theo cơ chế đồng bộ tại các điểm tổng hợp. Do đó, thời gian hoàn thành của toàn mạng bị chi phối bởi "liên kết chậm nhất" (Straggler Effect) [cite: 3, 7].

Tổng độ trễ của một vòng lặp ($t$) được chia làm 2 chặng (Sensor $\rightarrow$ Fog $\rightarrow$ Server), và được toán học hóa bằng hàm $\max$ lồng nhau như sau [cite: 3, 7]:

$$\tau_{round}^{(t)} = \max_{k \in \mathcal{K}} \left\{ \max_{i \in C_k} \left( \tau_{comp, i} + \tau_{comm}(i, CH_k) \right) + \tau_{comm}(CH_k, Server) \right\} + \tau_{agg}$$

Trong đó:
*   $\max_{i \in C_k} \left( \tau_{comp, i} + \tau_{comm}(i, CH_k) \right)$: Độ trễ thắt cổ chai nội cụm (Intra-cluster bottleneck), bị quyết định bởi AUV tính toán chậm nhất hoặc ở xa Cluster Head nhất trong cụm $C_k$.
*   $\tau_{comm}(CH_k, Server)$: Độ trễ truyền tải từ cụm $k$ về Trạm nổi trung tâm.
*   $\tau_{agg}$: Thời gian tổng hợp tham số tại Server (mang tính chất hằng số $\varpi$).

### 5. Hệ thống Ký hiệu và Biến số Thống nhất (Unified Notation Reference)

Để phục vụ cho bài toán tối ưu hóa xen kẽ giữa Lớp Mạng và Lớp AI, toàn bộ ký hiệu được chuẩn hóa và phân loại theo 5 nhóm sau:

#### A. Biến Mạng & Tô-pô (Network & Topology)

| Ký hiệu | Kiểu | Định nghĩa |
| :--- | :---: | :--- |
| $\mathcal{N} = \{1,\dots,N\}$ | Hằng | Tập hợp toàn bộ $N$ AUV trong mạng |
| $\mathcal{S}^{(t)} \subseteq \mathcal{N}$ | Trạng thái | Tập AUV đủ điều kiện tham gia vòng lặp $t$ (đủ pin & liên kết) |
| $\mathcal{K} = \{C_1,\dots,C_K\}$ | Trạng thái | Tập $K$ cụm được hình thành tại vòng lặp $t$ |
| $C_k \subseteq \mathcal{S}^{(t)}$ | Trạng thái | Tập thành viên AUV của cụm $k$ |
| $CH_k$ | Trạng thái | Cluster Head (nút Fog) được bầu chọn trong cụm $k$ |
| $\mathcal{G}^t = \{(u,v) \mid SL_u^{min} \le SL_{max}\}$ | Trạng thái | Đồ thị kết nối khả thi tại vòng $t$ (ràng buộc bởi SNR) |
| $a_i^{(t)} \in \mathcal{F}$ | Quyết định | Quyết định gán AUV $i$ vào Cluster Head $k$ tại vòng $t$ |
| $\alpha_{k,j}^{(t)}$ | Quyết định | Trọng số lai ghép mô hình (mixing weight) từ cụm $k$ sang cụm $j$; $\sum_j \alpha_{k,j}^{(t)} = 1$ |

#### B. Biến Kênh Âm Thanh & Năng lượng (Acoustic Channel & Energy)

| Ký hiệu | Đơn vị | Định nghĩa |
| :--- | :---: | :--- |
| $d_{uv}$ | m | Khoảng cách Euclid giữa AUV $u$ và $v$ |
| $f$ | kHz | Tần số sóng mang âm thanh (mặc định: $12$ kHz) |
| $B$ | Hz | Băng thông kênh truyền |
| $TL(d,f)$ | dB | Suy hao truyền dẫn (Thorp model) |
| $NL(f,B)$ | dB re $\mu$Pa | Mức nhiễu nền đại dương (Wenz model) |
| $SL_u^{min}(u,v)$ | dB re $\mu$Pa@1m | Mức nguồn phát tối thiểu để đạt $\gamma_{tgt}$ |
| $SL_{max}$ | dB re $\mu$Pa@1m | Giới hạn công suất phát cứng của modem ($= 140$ dB) |
| $\gamma_{tgt}$ | dB | Ngưỡng SNR mục tiêu tại máy thu |
| $R_{uv} = B\log_2(1+10^{\gamma_{tgt}/10})$ | bps | Tốc độ truyền dữ liệu theo Shannon |
| $P_{ac}$ | W | Công suất âm thanh bức xạ thực tế |
| $P_{tx} = P_{ac}/\eta_{ea}$ | W | Công suất điện tiêu hao tại mạch phát |
| $\eta_{ea}$ | — | Hiệu suất chuyển đổi điện–âm của modem |
| $P_{c,tx},\, P_{c,rx}$ | W | Công suất mạch điện tử tĩnh khi phát / nhận |
| $E_{tx}(L;u,v) = (P_{tx}+P_{c,tx})\frac{L}{R_{uv}}$ | J | Năng lượng truyền gói tin $L$ bit từ $u$ sang $v$ |
| $E_{comp,i} = \epsilon_{op}\,\Phi_i$ | J | Năng lượng tính toán cục bộ tại AUV $i$ |
| $\epsilon_{op}$ | J/FLOP | Hệ số năng lượng tiêu thụ mỗi phép tính |
| $\Phi_i$ | FLOPs | Tổng phép tính (forward Teacher + forward/backward Student) |
| $E_{round,i}^{(t)} = E_{comp,i} + E_{tx} + E_{rx}$ | J | Tổng năng lượng AUV $i$ tiêu thụ trong vòng lặp $t$ |
| $B_i^{(t)}$ | J | Dung lượng pin còn lại của AUV $i$ tại đầu vòng lặp $t$ |
| $E_{min}$ | J | Ngưỡng pin an toàn tối thiểu (AUV bị loại nếu $B_i^{(t)} < E_{min}$) |

#### C. Biến Độ trễ (Latency)

| Ký hiệu | Định nghĩa |
| :--- | :--- |
| $\tau_{comp,i} = \frac{N_i \cdot c_0}{h_i}$ | Độ trễ tính toán cục bộ tại AUV $i$ |
| $\tau_{comm}(u,v) = \frac{L}{R_{uv}} + \frac{d_{uv}}{c_s}$ | Tổng độ trễ truyền tin (transmission + propagation) |
| $c_s \approx 1500\text{ m/s}$ | Vận tốc âm thanh dưới nước |
| $\tau_{agg}$ | Thời gian tổng hợp mô hình tại Server (hằng số $\varpi$) |
| $\tau_{round}^{(t)} = \max_{k}\{\max_{i \in C_k}(\tau_{comp,i}+\tau_{comm}(i,CH_k))+\tau_{comm}(CH_k,\text{Server})\}+\tau_{agg}$ | Tổng độ trễ một vòng lặp (dominated by straggler) |
| $\tau_{max}$ | Ngưỡng độ trễ tối đa cho phép (ràng buộc C3) |

#### D. Biến Mô hình Học máy (ML Model Variables)

| Ký hiệu | Định nghĩa |
| :--- | :--- |
| $\Theta^{(t)}$ | Trọng số mô hình toàn cục (Server) tại đầu vòng lặp $t$ |
| $\Theta_i^s$ | Trọng số mô hình **Student** của AUV $i$ (nhỏ, $\approx 5{,}000$ params) |
| $\Theta_i^t$ | Trọng số mô hình **Teacher** cá nhân hóa của AUV $i$ (cố định) |
| $\tilde{\Theta}_i^s$ | Trọng số Student **sau khi** nén Ternary: $\tilde{\Theta}^s = \omega_p I_p - \omega_n I_n$ |
| $T$ | Tham số nhiệt độ (Temperature) cho Softmax làm trơn soft-label ($T=2$) |
| $y_i^s,\; y_i^t$ | Dự đoán xác suất mềm của Student và Teacher tại AUV $i$ |
| $F_{(s,i)} = F_{(s,i)}^t + F_{(s,i)}^d + F_{(s,i)}^h$ | Tổng hàm mất mát Student tại AUV $i$ |
| $F_{(s,i)}^t = \mathrm{CE}(y_i, y_i^s)$ | Task Loss (Cross-Entropy với nhãn thực) |
| $F_{(s,i)}^d = \frac{KL(y_i^t, y_i^s)}{F_{(t,i)}^t + F_{(s,i)}^t}$ | Adaptive Distillation Loss (KL có trọng số tự thích ứng) |
| $F_{(s,i)}^h = \frac{MSE(H_i^t, W_i^h H_i^s)+MSE(A_i^t,A_i^s)}{F_{(t,i)}^t+F_{(s,i)}^t}$ | Adaptive Hidden Loss (trạng thái ẩn & attention map) |
| $\Delta = \frac{T_k}{d^2}\sum|\Theta_j^s|$ | Ngưỡng Ternary Quantization thích ứng |
| $L = 2\times|\Theta^s| + L_{meta}$ | Kích thước gói tin sau nén 2-bit (đơn vị: bits) |
| $EMD_{i,j}$ | Earth Mover's Distance giữa phân phối dữ liệu AUV $i$ và $j$ |
| $\eta$ | Learning rate Gradient Descent cục bộ tại AUV |

#### E. Biến Tác tử RL & Hàm mục tiêu (RL Agent & Objective)

| Ký hiệu | Định nghĩa |
| :--- | :--- |
| $S_t$ | Trạng thái RL (mức pin các AUV lân cận + chất lượng kênh) |
| $a_t \in \mathcal{F}$ | Hành động RL (gán cụm / chọn định tuyến) |
| $r^{t+1}$ | Phần thưởng RL (tiết kiệm năng lượng, giảm trễ; phạt khi rớt gói) |
| $\beta$ | Tốc độ học Q-learning |
| $\gamma$ | Hệ số chiết khấu phần thưởng tương lai |
| $\theta^{(t)}$ | Ngưỡng pin động tại vòng $t$ (tính từ trung bình & phương sai $B_i^{(t)}$) |
| $\text{Fitness}_i = w_1 F_i^\epsilon + w_2 F_i^d + w_3 F_i^\delta + w_4 F_i^f$ | Hàm thích nghi MRFO cho ứng viên Cluster Head $i$ |
| $F_i^\epsilon, F_i^d, F_i^\delta, F_i^f$ | 4 thành phần Fitness: năng lượng, khoảng cách, độ trễ, lưu lượng |
| $\lambda_E,\; \lambda_\tau$ | Trọng số chuẩn hóa thứ nguyên trong hàm mục tiêu $\mathcal{J}$ |
| $\mathcal{J} = \text{Loss}_{model}(\Theta^T) + \lambda_E\sum\bar{E}_{round}^{(t)} + \lambda_\tau\sum\bar{\tau}_{round}^{(t)}$ | Hàm chi phí tổng hợp toàn mạng (cực tiểu hóa) |

---

### 6. Bài toán Tối ưu hóa Tổng hợp và Ràng buộc (Joint Optimization Problem)

Mục tiêu của nghiên cứu là tìm ra bộ tham số mô hình AI tối ưu ($\Theta$) và các quyết định điều phối mạng $\{a, \mathcal{N}\}$ nhằm cực tiểu hóa chi phí tổng hợp toàn mạng. Bài toán tối ưu hóa đa mục tiêu (Multi-Objective Optimization) được phát biểu dưới dạng toán học như sau [cite: 3]:

# Cân nhắc có đưa Loss vào hàm mục tiêu không, giải thích ý nghĩa xuất hiện, Loss liên quan gì tới E và T, liên quan gì tới hệ thống ?

$$\min_{(\Theta, a, \mathcal{N})} \mathcal{J} = \text{Loss}_{model}(\Theta^T) + \lambda_E \sum_{t=0}^{T-1} \overline{E}_{round}^{(t)} + \lambda_\tau \sum_{t=0}^{T-1} \overline{\tau}_{round}^{(t)}$$

*(Trong đó, $\overline{E}$ và $\overline{\tau}$ là các giá trị năng lượng và độ trễ đã được chuẩn hóa về thang $[0,1]$ để tránh sai khác thứ nguyên với $\text{Loss}_{model}$)*.

**Phụ thuộc vào các điều kiện ràng buộc vật lý (Subject to):**

**C1. Ràng buộc Khả thi Kênh truyền (Acoustic Link Feasibility):**
Mọi liên kết sử dụng phải thuộc đồ thị khả thi $\mathcal{G}^t$, đồng thời SNR thực tế tại đầu thu phải đạt ngưỡng mục tiêu $\gamma_{tgt}$ [cite: 3]:

$$a_i^{(t)} \in \mathcal{G}^t \quad \text{và} \quad SNR_{uv}^{(t)} \ge \gamma_{tgt}, \quad \forall (u,v) \text{ được chọn}$$

*(Ràng buộc này đảm bảo cả tính khả thi vật lý — liên kết nằm trong $\mathcal{G}^t$ — lẫn chất lượng truyền thông tối thiểu để mô hình Student được truyền không bị hỏng)*.

**C2. Ràng buộc Năng lượng Sinh tồn (Battery Preservation):**
Năng lượng tiêu hao trong một vòng lặp ($E_{round}$) không được làm cạn kiệt pin của bất kỳ AUV nào dưới mức an toàn $E_{min}$ để tránh đứt gãy mạng [cite: 3].

$$B_i^{(t+1)} = B_i^{(t)} - E_{round, i}^{(t)} \ge E_{min}, \quad \forall i \in \mathcal{S}^{(t)}, \forall t$$

**C3. Ràng buộc Độ trễ Tối đa (Latency Deadline):**
Tổng độ trễ chênh lệch giữa liên kết chậm nhất (straggler) và thời gian tính toán không được vượt quá ngưỡng chịu đựng của hệ thống thời gian thực $\tau_{max}$ [cite: 3].

$$\tau_{round}^{(t)} \le \tau_{max}, \quad \forall t$$

**C4. Ràng buộc Cấu trúc Liên kết (Association & Mixing Constraints):**
Mỗi AUV $i$ chỉ được phép kết nối với một Cluster Head hợp lệ thuộc tầng Fog, và tổng trọng số lai ghép mô hình giữa các cụm phải bằng 1 để đảm bảo mô hình hội tụ không bị bùng nổ gradient [cite: 3].

$$a_i^{(t)} \in \mathcal{F}, \quad \sum_{j \in \{k\} \cup \mathcal{N}_k^{(t)}} \alpha_{k,j}^{(t)} = 1, \quad \forall k, t$$

**Chiến lược giải quyết bài toán:**
Vì phương trình $\mathcal{J}$ là một bài toán tối ưu phi tuyến tính (Non-convex) và tổ hợp NP-Hard (do sự pha trộn giữa biến liên tục $\Theta$ và biến rời rạc $a, \mathcal{N}$), hệ thống áp dụng chiến lược **Tối ưu hóa xen kẽ (Alternating Optimization)**:
1. **Bước 1 (Lớp Mạng):** Cố định cấu trúc AI, Agent sử dụng thuật toán lai MRFO và RL để giải quyết cấu trúc liên kết $\{a, \mathcal{N}\}$ nhằm cực tiểu hóa chi phí $E$ và $\tau$ [cite: 8].
2. **Bước 2 (Lớp AI):** Dựa trên topology mạng vừa cố định, thực hiện huấn luyện mô hình thông qua Chưng cất tri thức cục bộ và cập nhật trọng số $\Theta$ qua gradient lượng tử hóa (Ternary) [cite: 6].

## IV. PHƯƠNG PHÁP NGHIÊN CỨU ĐỀ XUẤT

### 1. Khung mô hình CFL-KDT (Clustered Federated Learning with Knowledge Distillation & Ternary Quantization)

Khung mô hình đề xuất giải quyết bài toán dữ liệu phi đồng nhất (Non-IID), kiến trúc mô hình dị biệt và nút thắt cổ chai truyền thông thông qua một quy trình 3 giai đoạn tích hợp:

**a. Phân cụm dựa trên Earth Mover’s Distance (EMD) và Hierarchical Clustering:**
Để khắc phục sự phân kỳ trọng số do dữ liệu Non-IID, hệ thống không huấn luyện một mô hình toàn cục duy nhất mà chia các AUV thành các cụm có phân phối dữ liệu tương đồng. Khoảng cách EMD được sử dụng để đo lường độ lệch trọng số giữa mô hình cục bộ và mô hình lý tưởng trên dữ liệu toàn cụm [cite: 5]. Bài toán phân cụm được định nghĩa là:

$$\arg \min_{K} \sum_{t=1}^{l} \frac{\sum_{i=1}^{m} k_{t,i} n_i}{N} \frac{\|G_t - H_t\|}{\|H_t\|}$$

Trong đó:
*   $G_t$: Mô hình được tổng hợp của cụm $t$ [cite: 5].
*   $H_t$: Mô hình lý tưởng được huấn luyện tập trung trên dữ liệu của toàn cụm $t$ [cite: 5].
*   $k_{t,i} \in \{0, 1\}$: Biến chỉ thị AUV $i$ thuộc cụm $t$ [cite: 5].

Hệ thống áp dụng thuật toán **Hierarchical Clustering (HC)** (Phân cụm phân cấp) để giải bài toán này, nhóm các AUV dựa trên khoảng cách Euclid của tham số mô hình cục bộ, từ đó tạo ra cấu trúc liên kết nội cụm ổn định [cite: 5].

**b. Cơ chế Chưng cất Tri thức Thích ứng (Adaptive Federated Knowledge Distillation):**
Để giải quyết sự không đồng nhất về phần cứng (Model Heterogeneity), mỗi AUV duy trì một mô hình "Teacher" cá nhân hóa (được huấn luyện trước và giữ cố định) và một mô hình "Student" nhỏ gọn dùng chung [cite: 6]. Thay vì chưng cất với trọng số cố định, hệ thống áp dụng cơ chế **thích ứng (Adaptive)** dựa trên độ chính xác dự đoán của Teacher và Student [cite: 6].

Hàm mất mát cục bộ của Student trên AUV $i$ là sự kết hợp của ba thành phần:

$$F_{(s,i)} = F_{(s,i)}^{t} + F_{(s,i)}^{d} + F_{(s,i)}^{h}$$

*   **Task Loss ($F_{(s,i)}^t$):** Sai số Cross-Entropy giữa dự đoán của Student ($y_i^s$) và nhãn thực tế ($y_i$) [cite: 6].
*   **Adaptive Distillation Loss ($F_{(s,i)}^d$):** Đo lường độ chệch Kullback-Leibler (KL) giữa xác suất mềm của Teacher ($y_i^t$) và Student ($y_i^s$). Trọng số của KL được điều chỉnh động bởi tổng sai số Task Loss, giúp hệ thống tập trung vào nhãn thực khi dự đoán của mô hình còn yếu:
    $$F_{(s,i)}^d = \frac{KL(y_i^t, y_i^s)}{F_{(t,i)}^t + F_{(s,i)}^t}$$
*   **Adaptive Hidden Loss ($F_{(s,i)}^h$):** Ép Student học các đặc trưng sâu thông qua trạng thái ẩn ($H$) và bản đồ chú ý ($A$) của Teacher bằng sai số toàn phương trung bình (MSE):
    $$F_{(s,i)}^h = \frac{MSE(H_i^t, W_i^h H_i^s) + MSE(A_i^t, A_i^s)}{F_{(t,i)}^t + F_{(s,i)}^t}$$

**c. Lượng tử hóa Ternary cấp mạng (Ternary Quantization - TTQ):**
Mô hình Student, mặc dù đã được thu gọn, vẫn chứa các tham số số thực 32-bit gây tốn kém năng lượng khi truyền qua kênh âm thanh [cite: 6]. Khung mô hình áp dụng kỹ thuật **Lượng tử hóa Ternary** để nén trực tiếp trọng số $\Theta^s$ của Student thành các giá trị rời rạc $\{-1, 0, 1\}$ trước khi đưa vào modem phát [cite: 6]. 

*   **Ngưỡng lượng tử hóa thích ứng ($\Delta$):** Được tính toán dựa trên độ phân tán của trọng số tại từng lớp:
    $$\Delta = \frac{T_k}{d^2} \sum |\Theta_i^s|$$
*   **Ánh xạ tham số:** Các trọng số lớn hơn ngưỡng được giữ lại qua ma trận chỉ thị dương ($I_p$) và âm ($I_n$), các trọng số nhỏ bị triệt tiêu về 0 [cite: 6]. Trọng số nén $\tilde{\Theta}^s$ được truyền đi với dung lượng giảm xấp xỉ 16 lần [cite: 6]:
    $$\tilde{\Theta}^s = \omega_p \times I_p - \omega_n \times I_n$$

Sự kết hợp giữa **(a) Phân cụm**, **(b) Chưng cất thích ứng**, và **(c) Lượng tử hóa Ternary** chính là lõi công nghệ giúp hệ thống duy trì độ chính xác cao trên dữ liệu Non-IID trong khi vẫn đáp ứng được các ràng buộc vật lý khắc nghiệt của môi trường dưới nước.

### 2. Thuật toán Tối ưu hóa Hybrid (RL & MRFO) điều phối mạng

Môi trường đại dương biến động không ngừng (dòng chảy, fading đa đường) khiến các thuật toán định tuyến tĩnh bị thất bại. Hệ thống sử dụng một khung tối ưu hóa lai kết hợp giữa Học tăng cường (RL) ở cấp vĩ mô và Tối ưu hóa bầy đàn Manta Ray (MRFO) ở cấp vi mô [cite: 8].

#### a. Học tăng cường (RL) cho quyết định gom cụm và định tuyến
Mỗi nút (hoặc trạm điều khiển) đóng vai trò là một Agent học cách thiết lập liên kết mạng hiệu quả năng lượng. 

*   **Không gian trạng thái ($S_t$):** Bao gồm mức năng lượng hiện tại của các AUV lân cận và chất lượng kênh truyền âm thanh [cite: 8].
*   **Hành động ($a_t$):** Quyết định gán một AUV vào một cụm cụ thể hoặc chọn đường định tuyến [cite: 8].
*   **Phần thưởng ($r^{t+1}$):** Được tính toán dựa trên mức độ tiết kiệm năng lượng và chi phí giao tiếp; agent bị phạt nếu hành động dẫn đến rớt gói hoặc tạo ra liên kết có SNR thấp [cite: 8].
*   **Hàm cập nhật (Bellman Equation):** Ma trận giá trị Q được cập nhật liên tục thông qua kỹ thuật Temporal-Difference để Agent rút kinh nghiệm từ quá trình tương tác với môi trường đại dương [cite: 8]:

$$Q_{t+1}(S_t, a_t) = (1-\beta)\,Q_t(S_t, a_t) + \beta \left[ r^{t+1} + \gamma \max_{a'} Q_t(S_{t+1}, a') - Q_t(S_t, a_t) \right]$$

*(Trong đó $\beta$ là tốc độ học Q-learning, $\gamma$ là hệ số chiết khấu; ký hiệu $\beta$ thay cho $\alpha$ để tránh nhầm với mixing weight $\alpha_{k,j}$)*.

#### b. Lựa chọn Cluster Head bằng Tối ưu hóa bầy đàn Manta Ray (MRFO)
Sau khi RL định hình sơ bộ các cụm, thuật toán sinh học MRFO được kích hoạt nội bộ để bầu chọn ra nút làm Cluster Head (CH) [cite: 8]. Việc chọn CH không chỉ dựa trên năng lượng mà là một bài toán tối ưu hóa đa mục tiêu (Multi-Objective Optimization) nhằm thỏa mãn 4 tiêu chí cốt lõi:

Hàm độ thích nghi (Fitness Function) cho một ứng viên $i$ được định nghĩa là sự tối ưu hóa đồng thời [cite: 8]:

$$\text{Fitness}_i = w_1 F_i^\epsilon + w_2 F_i^d + w_3 F_i^\delta + w_4 F_i^f$$

Cụ thể, các thành phần vật lý được toán học hóa như sau:
*   **Năng lượng ($F_i^\epsilon$):** Ưu tiên nút có phần trăm năng lượng dư thừa cao nhất để gánh vác việc nhận/gửi và tổng hợp mô hình (Aggregating), tránh làm chết nút [cite: 8].
*   **Khoảng cách ($F_i^d$):** Tối thiểu hóa khoảng cách Euclid tổng hợp. Nghĩa là CH phải nằm ở trung tâm của các thành viên trong cụm và càng gần trạm nổi (Sink) càng tốt để giảm suy hao lan truyền [cite: 8].
*   **Độ trễ ($F_i^\delta$):** Tỷ lệ thuận với số lượng thành viên trong cụm. Cụm càng đông, nguy cơ nghẽn cổ chai và độ trễ chờ đợi càng lớn, do đó thuật toán ưu tiên cân bằng kích thước cụm [cite: 8].
*   **Mật độ lưu lượng ($F_i^f$):** Đo lường áp lực mạng tại nút $i$, được tính bằng trung bình của 3 chỉ số: mức sử dụng bộ đệm (Buffer utilization - $B_{ut}$), tỷ lệ rớt gói (Packet drop ratio - $P_{dr}$), và tải kênh (Channel load - $C_l$) [cite: 8]:

$$F_i^f = \frac{1}{3} \left[ B_{ut} + P_{dr} + C_l \right]$$

Thuật toán MRFO sẽ mô phỏng 3 hành vi tìm mồi của cá đuối (Chain foraging, Cyclone foraging, Somersault foraging) để nhanh chóng tìm ra nút có chỉ số $\text{Fitness}$ tốt nhất, đảm bảo tính hội tụ tốc độ cao và không bị kẹt ở cực tiểu cục bộ [cite: 8].

### 3. Giả mã Thuật toán Tổng thể (Algorithm Pseudocode)

Để hệ thống hóa luồng hoạt động của khung mô hình **CFL-KDT** (Clustered Federated Learning with Knowledge Distillation and Ternary Quantization), toàn bộ quá trình tương tác giữa Trạm nổi (Server), Cluster Head (Fog) và AUV (Client) được tóm tắt trong **Algorithm 1**. Quá trình này được thiết kế theo dạng tối ưu hóa xen kẽ (Alternating Optimization): Lớp mạng thiết lập đồ thị và phân cụm trước, sau đó lớp AI tiến hành huấn luyện, nén và truyền tải [cite: 3].

**Algorithm 1: Học liên kết phân cụm lai (RL-MRFO) kết hợp Chưng cất tri thức và Nén Ternary**

```
Input : Tập AUV N = {1,...,N}; Teacher weights Θ_i^t; Server init Θ^0;
        Số vòng lặp T; Ngưỡng pin E_min.
Output: Mô hình toàn cục tối ưu Θ^T.

 1: Khởi tạo: Trạm nổi Broadcast Θ^0 tới toàn bộ AUV; khởi tạo Q-Table.
 2: for t = 1, 2, ..., T do
 3:   // ── GIAI ĐOẠN 1: QUẢN LÝ NĂNG LƯỢNG & PHÂN CỤM (LỚP MẠNG) ──
 4:   Tính ngưỡng năng lượng động θ^(t) theo trung bình & phương sai pin mạng.
 5:   for mỗi AUV i ∈ N do
 6:     Xác định xác suất tham gia P_i theo pin còn lại B_i^(t).
 7:     if B_i^(t) ≥ θ^(t)  AND  SL_u^min ≤ SL_max then
 8:       Đưa AUV i vào danh sách đủ điều kiện S^(t).
 9:     end if
10:   end for
11:   Tác tử RL quan sát S_t, quyết định a_t → gom nhóm sơ bộ S^(t).
12:   for mỗi cụm C_k vừa hình thành do
13:     Chạy MRFO nội cụm đánh giá 4 mục tiêu: F^ε, F^d, F^δ, F^f.
14:     Bầu chọn AUV có Fitness cao nhất làm Cluster Head (CH_k).
15:   end for

16:   // ── GIAI ĐOẠN 2: HUẤN LUYỆN CỤC BỘ & LƯỢNG TỬ HÓA (LỚP AI) ──
17:   for mỗi AUV i ∈ S^(t) [song song] do
18:     Đồng bộ Student model: Θ_i^s ← Θ^(t-1).
19:     Chạy Teacher cục bộ → lấy soft-labels (xác suất mềm).
20:     Tính tổng tổn hao: F_(s,i) = F^t_(s,i) + F^d_(s,i) + F^h_(s,i).
21:     Cập nhật: Θ_i^s ← Θ_i^s − η · ∇F_(s,i).  [Gradient Descent]
22:     Ternary Quantization (TTQ): tính Δ, ánh xạ Θ_i^s → {−1, 0, +1}.
23:     Cập nhật pin: B_i^(t+1) ← B_i^(t) − (E_comp + E_tx).
24:     Truyền gói nén Θ̃_i^s tới CH_k qua kênh âm thanh.
25:   end for

26:   // ── GIAI ĐOẠN 3: TỔNG HỢP VÀ PHẢN HỒI (FOG & SERVER) ──
27:   for mỗi cụm C_k do
28:     CH_k tổng hợp nội cụm: Θ_k^(t) = Aggregate({ Θ̃_i^s | i ∈ C_k }).
29:     Gửi Θ_k^(t) về Trạm nổi qua liên kết đường dài.
30:   end for
31:   Trạm nổi giải lượng tử hoá + Proximal Optimization → cập nhật Θ^(t).
32:   RL nhận thưởng r^(t+1) (dựa trên τ_round^(t), E_round^(t)) → cập nhật Q-Table.
33: end for
34: return Θ^T
```
---

## V. THIẾT LẬP MÔ PHỎNG VÀ KẾ HOẠCH ĐÁNH GIÁ (SIMULATION SETUP & EVALUATION PLAN)

Để kiểm chứng tính đúng đắn và hiệu năng của khung mô hình **CFL-KDT** đề xuất, một môi trường mô phỏng độ chân thực cao (High-fidelity simulation) sẽ được thiết lập, phản ánh sát thực tế các hạn chế về phần cứng, năng lượng và kênh truyền âm thanh dưới nước.

---

### 5.1. Nền tảng Mô phỏng và Yêu cầu Phần cứng (Hardware & Software Setup)

**Môi trường Phần mềm:** Mô phỏng được triển khai trên ngôn ngữ **Python (phiên bản 3.12+)**, sử dụng thư viện **PyTorch** và **TensorFlow Federated (TFF)** để điều phối các vòng lặp học liên kết phi tập trung. Các thư viện **NumPy** và **SciPy** được sử dụng để lập trình lớp vật lý của kênh truyền âm thanh (suy hao Thorp, nhiễu Wenz).

**Phần cứng giả lập tại AUV (Client):** Vi xử lý của các AUV được mô phỏng tương đương với các dòng vi điều khiển tiêu thụ điện năng thấp (ví dụ: dòng **ARM Cortex-M4/M7**), phù hợp với giới hạn RAM/CPU khắt khe. Năng lượng tiêu thụ tính toán được gán ở mức $0.5$ Joules cho mỗi epoch cục bộ.

**Phần cứng tại Trạm nổi (Server/Gateway):** Các thao tác tổng hợp mô hình và Proximal Optimization phức tạp sẽ được chạy mô phỏng trên các máy chủ có GPU hỗ trợ CUDA (ví dụ: **NVIDIA GeForce RTX 2080Ti** hoặc **RTX 4060**) để tăng tốc độ xử lý.

---

### 5.2. Cấu hình Mạng & Môi trường Biển (Network & Acoustic Parameters)

Kịch bản mô phỏng giả lập một vùng biển giám sát với các thông số vật lý chặt chẽ như sau:

| Thông số | Giá trị thiết lập | Ghi chú |
|---|---|---|
| Khu vực hoạt động | $2000\text{m} \times 2000\text{m} \times 1000\text{m}$ | Vùng không gian 3D dưới nước |
| Quy mô mạng lưới ($N$) | $50$ đến $200$ AUVs | Đánh giá khả năng mở rộng (Scalability) |
| Tần số sóng mang ($f$) | $12$ kHz | Chuyên dụng cho modem âm thanh IoUT |
| Băng thông kênh truyền ($B$) | $10\text{–}20$ kbps | Giới hạn băng thông cực hẹp |
| Tỷ lệ rớt gói ngẫu nhiên | $5\%\text{–}10\%$ | Bắt chước hiện tượng fading đa đường và nhiễu |
| Năng lượng pin khởi tạo | $1000\text{–}1200$ Joules | Khởi tạo ngẫu nhiên, tạo độ lệch về tuổi thọ |
| Giới hạn công suất phát ($SL_{max}$) | $140$ dB re $1\mu\text{Pa @ 1m}$ | Ràng buộc vật lý modem âm thanh |

---

### 5.3. Bộ Dữ liệu và Mô hình Máy học (Dataset & ML Models)

**Bộ dữ liệu (Dataset):** Sử dụng bộ dữ liệu **Fish4Knowledge** (bao gồm khoảng 27,370 hình ảnh của 23 loài sinh vật biển, kích thước $64 \times 64$). Bộ dữ liệu này sẽ được xáo trộn và chia cho các AUV theo phân phối **Non-IID** với hệ số Dirichlet $\alpha = 0.1$, ép buộc mỗi AUV chỉ nhận được $N_c = 2$ hoặc $N_c = 5$ lớp/AUV để phản ánh thực tế sự di chuyển ở các độ sâu khác nhau.

**Mô hình Máy học:**
- **Teacher Model:** Sử dụng các mạng nơ-ron sâu như **CNN** hoặc **ResNet50** (được huấn luyện trước và cá nhân hóa cho từng AUV).
- **Student Model:** Sử dụng kiến trúc **Lightweight CNN/MLP** hai lớp với khoảng $\approx 5{,}000$ tham số, đảm bảo vừa vặn với bộ nhớ RAM cực nhỏ của ARM Cortex-M.

---

### 5.4. Thuật toán Cơ sở để so sánh (Baselines)

Để chứng minh sự vượt trội của hệ thống đề xuất, kết quả sẽ được đối chiếu với các thuật toán học liên kết hiện hành:

| Thuật toán | Mô tả | Vai trò |
|---|---|---|
| **Centralized Learning (CL)** | Tập trung toàn bộ dữ liệu thô về trạm nổi | Giới hạn trần (Upper-bound) — "Oracle" lý tưởng |
| **FedAvg** | Học liên kết cơ bản, cấu trúc hình sao, cập nhật đồng bộ | Baseline chuẩn |
| **FedProx** | Biến thể FedAvg với Proximal term xử lý Non-IID | Baseline mạnh nhất (Flat FL) |
| **FedKD / HFL-NoCoop** | KD hoặc phân cụm nhưng không trao đổi tham số giữa các CH | Chứng minh hiệu quả của MRFO |
| **CFL-KDT (Đề xuất)** | Clustered FL + Knowledge Distillation + Ternary Quantization + RL-MRFO | Hệ thống đề xuất |

---

### 5.5. Các Chỉ số Đánh giá (KPI Metric Formulations)

Các KPI đánh giá được công thức hóa rõ ràng, tránh dùng từ ngữ định tính:

**(1) Mức tiêu thụ năng lượng trung bình ($E_{avg}$):**
Đo lường số Joules tiêu tốn trên mỗi node cho đến khi mô hình hội tụ. Năng lượng tính toán được gán là $0.5$ J/epoch; năng lượng truyền tải là $2$ J/KB.

$$E_{avg} = \frac{1}{|\mathcal{S}^{(t)}|} \sum_{i \in \mathcal{S}^{(t)}} \left( E_{comp,i} + E_{tx,i} \right)$$

**(2) Băng thông truyền tải (Communication Overhead):**
Đo lường bằng số Kilobytes trung bình mỗi AUV phải truyền trong một vòng lặp. Kỳ vọng giảm từ $\approx 20$ KB/vòng (FedAvg gốc) xuống còn $\approx 6.5$ KB/vòng nhờ lượng tử hóa Ternary 2-bit:

$$\text{Overhead} = \frac{|\Theta^s| \times b_{quant}}{8 \times 1024} \quad [\text{KB/round}]$$

trong đó $b_{quant} = 2$ bit cho Ternary Quantization (TTQ), so với $b_{quant} = 32$ bit cho truyền float32 nguyên bản.

**(3) Tuổi thọ mạng lưới — First/Half Node Dies (FND/HND):**
Đo lường bằng số vòng lặp giao tiếp (Rounds) cho đến khi nút đầu tiên cạn kiệt pin (FND) và khi 50% số nút chết (HND):

$$\text{FND} = \min \{ t : \exists\, i \in \mathcal{N},\; B_i^{(t)} = 0 \}, \quad \text{HND} = \min \left\{ t : \left|\{ i : B_i^{(t)} = 0 \}\right| \ge \frac{N}{2} \right\}$$

**(4) Độ chính xác mô hình toàn cục (Global Model Accuracy / F1-Score):**
Đánh giá trên tập test tập trung tại trạm nổi sau mỗi vòng lặp giao tiếp $t$.

---

### 5.6. Các Kịch bản Thử nghiệm Khắc nghiệt (Stress-test Scenarios)

Hệ thống sẽ được đánh giá qua **3 kịch bản mô phỏng độ khắc nghiệt của đại dương**:

**Kịch bản 1 — Thử nghiệm độ chệch dữ liệu (Data Heterogeneity Sensitivity):**
Dữ liệu được chia theo phân phối Dirichlet với $\alpha = 0.1$ tạo ra môi trường Non-IID cực mạnh (mỗi AUV chỉ chứa ảnh của 2 đến 5 loài). *Mục đích:* Chứng minh cơ chế phân cụm EMD và chưng cất Teacher-Student không bị sụp đổ độ chính xác khi dữ liệu bị phân cực.

**Kịch bản 2 — Thử nghiệm khả năng mở rộng (Scalability Test):**
Quy mô mạng tăng dần theo các mốc: $N \in \{50, 100, 150, 200\}$ AUVs. *Mục đích:* Chứng minh khi mạng lưới phình to, tỷ lệ nút kết nối trực tiếp với trạm nổi có thể rớt xuống dưới 48%, nhưng cấu trúc phân cụm vẫn duy trì tỷ lệ tham gia mạng tiệm cận tuyệt đối.

**Kịch bản 3 — Thử nghiệm chống chịu nhiễu âm thanh (Robustness to Acoustic Noise):**
Tỷ lệ mất gói tin (PLR) được ép lần lượt ở các mức: $\{0\%, 5\%, 10\%, 15\%\}$. *Mục đích:* Khẳng định cơ chế tổng hợp nội cụm dựa trên trung vị (Median-based aggregation) lọc bỏ được các gói tin lỗi, giúp độ chính xác chỉ suy giảm biên độ nhỏ (dưới 5%) ngay cả khi nhiễu sóng cao.

---

### 5.7. Thiết lập Siêu tham số (Hyperparameter Configurations)

Để đảm bảo tính tái lập (reproducibility), các siêu tham số được thiết lập cố định dựa trên cấu hình chuẩn của các hệ thống IoUT:

| Siêu tham số | Giá trị | Ghi chú |
|---|---|---|
| Trình tối ưu hóa | Adam | Tốc độ học toàn cục $\eta = 0.001$ |
| Batch size | 64 | Huấn luyện cục bộ tại AUV |
| Local epochs | 2 | Số vòng lặp nội bộ mỗi giao tiếp |
| Distillation temperature ($T$) | $2$ | Làm trơn phân phối xác suất soft-label |
| Trọng số Loss chưng cất | $0.7$ | $\mathcal{L} = 0.7\mathcal{L}_{KD} + 0.3\mathcal{L}_{CE}$ |
| Hàm kích hoạt | ReLU | Toàn bộ Student Model |
| Số cá thể MRFO | 50 | Tìm kiếm Cluster Head nội cụm |
| Số vòng lặp MRFO (tối đa) | 100 | Không gian tìm kiếm nội cụm |

---

### 5.8. Phân tích Độ phức tạp và Tính khả thi Phần cứng (Complexity & Hardware Feasibility)

**Độ phức tạp Không gian (Space Complexity):** Student Model với $\approx 5{,}000$ tham số float32 chỉ chiếm $\approx 20$ KB RAM, hoàn toàn nằm gọn trong bộ nhớ của ARM Cortex-M4/M7. Sau khi áp dụng Ternary Quantization, dung lượng nén xuống còn $\approx 1.25$ KB — giảm **16 lần** so với biểu diễn 32-bit nguyên bản.

**Độ phức tạp Thuật toán Phân cụm (Time Complexity):** Thuật toán MRFO và RL bầu chọn Cluster Head được đẩy lên xử lý tại các nút CH có năng lượng dư thừa lớn nhất. Phân tích Big-O cho thấy thời gian chạy phân cụm chỉ tăng tuyến tính $\mathcal{O}(N)$ — đảm bảo độ trễ ở mức mili-giây (ms), không bùng nổ theo hàm mũ khi $N \to 200$.

---

### 5.9. Kết quả Kỳ vọng (Expected Results Summary)

| Chỉ số đánh giá | CFL-KDT (Đề xuất) | FedAvg | FedProx | Centralized (Oracle) |
|---|---|---|---|---|
| **Độ chính xác (Non-IID, $\alpha=0.1$)** | $\mathbf{\approx 82\%}$ | $\approx 74\%$ | $\approx 76\%$ | $\approx 89\%$ |
| **Communication overhead/round** | $\mathbf{\approx 6.5}$ KB | $\approx 20$ KB | $\approx 20$ KB | N/A (không khả thi) |
| **Giảm overhead so với FedAvg** | $\mathbf{-67.5\%}$ | — | — | — |
| **Giảm năng lượng truyền thông** | $\mathbf{-31\%\text{–}33\%}$ | — | — | — |
| **Giảm dung lượng gói tin (Ternary)** | $\mathbf{-71\%\text{–}95\%}$ | — | — | — |
| **Tuổi thọ mạng (HND)** | $\mathbf{+40\%}$ so với FedAvg | Baseline | $\approx +5\%$ | N/A |
| **Participation Rate ($N=200$)** | $\mathbf{\approx 100\%}$ | $< 48\%$ | $< 48\%$ | N/A |
| **Độ giảm chính xác khi PLR = 15%** | $\mathbf{< 5\%}$ | $> 12\%$ | $> 10\%$ | N/A |

> **Tóm tắt:** Hệ thống **CFL-KDT** đề xuất đạt được sự cân bằng Pareto tốt nhất giữa bốn chiều tối ưu hóa đồng thời: *(i)* độ chính xác mô hình tiệm cận Centralized Learning; *(ii)* tiêu thụ năng lượng và băng thông tối thiểu; *(iii)* tuổi thọ mạng kéo dài tối đa; và *(iv)* khả năng tham gia mạng ổn định ở mọi quy mô — những mục tiêu mà không có thuật toán baseline nào đáp ứng đồng thời được trong ràng buộc vật lý khắc nghiệt của môi trường IoUT.