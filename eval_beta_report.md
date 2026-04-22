# Báo cáo Đánh giá Beta Sensitivity

## 1. Tổng quan
Thực nghiệm này đánh giá tác động của tham số ngưỡng tham gia ($\beta$) đối với hiệu suất Học liên kết (FL) trong các quy mô đội tàu AUV khác nhau ($M \in \{9, 16, 25\}$). Tham số $\beta$ điều khiển điều kiện để một AUV được phép tham gia vào quá trình huấn luyện dựa trên trạng thái hiện tại (năng lượng, kênh truyền, v.v.).

## 2. Phương pháp nghiên cứu
- **Số vòng FL (Rounds)**: 1000 vòng cho mỗi giá trị $\beta$.
- **Số Episode RL**: 1000 lần cập nhật PPO và reset môi trường
- **Dải $\beta$**: Từ 0.1 đến 0.9 với bước nhảy 0.1.
- **Cấu hình**:
    - Sử dụng giá trị trung bình cho công suất phát ($P_{mid}$) và tần số CPU ($f_{mid}$).
    - Đánh giá trên 3 quy mô swarm ($M=9, 16, 25$).

## 3. Phân tích kết quả thực nghiệm

### Hình 1: Tổng số lần truyền tin vs $\beta$ (Communication Times)
**Mô tả kịch bản:** Đo lường tổng số lượt tương tác giữa các AUV và Server trung tâm trong suốt 1000 vòng FL.
- **Phân tích:** 
    - Khi $\beta$ tăng, điều kiện để các node tham gia trở nên "dễ dãi" hơn, dẫn đến số lượng node tham gia mỗi vòng tăng lên. Do đó, tổng số lần truyền tin tăng tỷ lệ thuận với $\beta$.
    - Khi quy mô swarm ($M$) tăng, số lượng node tiềm năng lớn hơn dẫn đến tổng số lần truyền tin cũng tăng mạnh (đường $M=25$ nằm cao nhất).
- **Kết luận:** $\beta$ cao làm tăng chi phí truyền tin nhưng giúp tận dụng nhiều dữ liệu hơn từ swarm.

### Hình 2: Độ chính xác tại vòng 1000 vs $\beta$ (Accuracy)
**Mô tả kịch bản:** Ghi lại độ chính xác cuối cùng của mô hình toàn cục sau khi hoàn thành 1000 vòng huấn luyện.
- **Phân tích:** 
    - **Xu hướng theo $\beta$:** Về tổng thể, khi $\beta$ tăng, độ chính xác có xu hướng tăng. Điều này là do nhiều node tham gia hơn mang lại sự đa dạng về dữ liệu, giúp mô hình hội tụ tốt hơn.
    - **Xu hướng theo $M$:** Đáng chú ý, **khi $M$ tăng thì độ chính xác lại giảm** (Đường $M=9$ nằm trên cùng, $M=25$ nằm dưới cùng). Điều này có thể giải thích do sự không đồng nhất (Non-IID) của dữ liệu tăng lên khi số lượng node tăng, hoặc do nhiễu từ các node có cấu hình yếu làm giảm chất lượng mô hình chung trong môi trường không đồng bộ.
- **Kết luận:** Có sự đánh đổi giữa việc tăng quy mô swarm và độ chính xác của mô hình.

### Hình 3: Thời gian tiêu thụ trung bình vs $\beta$ (Average Time Consumption)
**Mô tả kịch bản:** Tính toán thời gian trung bình để hoàn thành một vòng FL (đo trên 5 vòng đầu tiên) bao gồm thời gian tính toán và truyền tin thủy âm.
- **Phân tích:** 
    - Khi $\beta$ tăng, số lượng node tham gia truyền tin cùng lúc tăng lên. Trong môi trường tín hiệu thủy âm, việc có nhiều node tham gia dẫn đến thời gian chờ đợi (waiting time) và tranh chấp kênh truyền lớn hơn.
    - Kết quả cho thấy khi $\beta$ càng cao thì thời gian sẽ càng lâu (đặc biệt rõ rệt ở $M=25$ khi $\beta > 0.6$).
- **Kết luận:** $\beta$ cao gây ra độ trễ lớn do phải chờ đợi các node tham gia, đòi hỏi thuật toán RL phải tối ưu để cân bằng giữa thời gian và hiệu năng.


## 4. Các bước tiếp theo: Huấn luyện & Đánh giá các phương án RL

### 4.1. Hình 4, 5, 6: Hiệu suất theo bước huấn luyện
**Mô tả kịch bản:** So sánh 5 phương án (Schemes) trong quá trình huấn luyện RL để thấy khả năng thích nghi của Agent PPO2.
- **Scheme 1:** Phương án đề xuất (RL tối ưu hóa cả lựa chọn node và tài nguyên).
- **Scheme 2:** AFL với $\beta$ được tối ưu hóa động.
- **Scheme 3:** AFL với $\beta$ cố định.
- **Scheme 4:** AFL kết hợp LAG (Lazily Aggregated Gradient).
- **Scheme 5:** AFL truyền thống.
- **Mục tiêu:** Chứng minh **Scheme 1** đạt độ chính xác cao nhất (Hình 4), tiêu thụ năng lượng thấp nhất (Hình 5) và thời gian vòng nhanh nhất (Hình 6).

### 4.2. Hình 7: So sánh thuật toán tối ưu (Profit Comparison)
**Mô tả kịch bản:** So sánh giá trị Phần thưởng (Reward/Profit) thu được giữa thuật toán PPO2 và các thuật toán cổ điển khác.
- **Đối tượng so sánh:** GA (Genetic Algorithm), PSO (Particle Swarm Optimization), AC (Actor-Critic), DDPG (Deep Deterministic Policy Gradient).
- **Mục tiêu:** Chứng minh **PPO2** đạt được mức lợi nhuận (Profit) cao nhất và ổn định nhất trong môi trường swarm AUV biến động.


