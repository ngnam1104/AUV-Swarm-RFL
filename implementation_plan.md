# Kế hoạch tăng tốc (Speed-up Plan) cho AUV-Swarm-RFL

Quá trình đọc lại toàn bộ repo cho thấy **nút thắt cổ chai lớn nhất (bottleneck)** nằm ở vòng lặp FL bên trong môi trường RL. 

Cụ thể, tại `env/auv_env.py`, mỗi lần RL gọi `step()`, trình mô phỏng `FLSimulator` sẽ thực hiện một vòng lặp Federated Learning hoàn chỉnh: chạy SGD qua tập dữ liệu cục bộ trên $M$ AUVs (tại `worker.py`). Nếu huấn luyện PPO 1 triệu steps, chúng ta đang thực hiện 1 triệu vòng lặp FL thực sự với PyTorch, điều này là cực kỳ chậm.

Dưới đây là các phương án tối ưu toàn diện từ cấp độ phần cứng đến thuật toán mà chúng ta có thể áp dụng:

## 1. Tích hợp tăng tốc GPU (CUDA)
Hiện tại toàn bộ mô phỏng FL đang bị ép chạy trên CPU (`v.detach().cpu()`, `os.cpu_count()`).
**Giải pháp:** 
- Định nghĩa biến cục bộ `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` trên toàn module `fl_core`.
- Đưa các dataset tensors (`self.all_images`, `self.all_labels`) và mô hình (`SimpleNN`) lên VRAM của GPU một lần duy nhất lúc khởi tạo.
- Xóa bỏ việc ép kiểuvề `.cpu()` khi truyền tải trọng số (weights) giữa các Aggregator và Worker, giữ cho luồng tensor hoàn toàn trên GPU.

## 2. Song song hóa Local Training bằng ThreadPool
Mặc dù PyTorch hỗ trợ việc tính toán song song, nhưng vòng lặp trong `simulator.py`:
```python
for worker in self.workers:
    w_local, _ = worker.local_train(...)
```
đang chạy hoàn toàn tuần tự.
**Giải pháp:**
Sử dụng `concurrent.futures.ThreadPoolExecutor` để chạy hàm `local_train` của tất cả $M$ AUVs cùng một lúc. Do khối lượng tính toán trên GPU là độc lập và PyTorch giải phóng GIL đối với các phép toán tensor, cách này sẽ giúp lấp đầy khả năng tính toán của thiết bị và giảm độ trễ vòng lặp xuống mức của AUV mất thời gian lâu nhất.

## 3. Tối ưu tần suất Evaluation (Đánh giá)
Hàm `_evaluate_accuracy` đang duyệt qua 10.000 test images. Mặc dù cấu hình đã cài `eval_interval=10`, thời gian tính toán này vẫn lớn nếu tính tổng lại.
**Giải pháp:**
- Đưa `DataLoader` của test set về tensor dạng khối (pre-loaded tensors) trên GPU tương tự như cách đã làm ở `worker.py` thay vì lặp qua từng batch. Tốc độ kiểm tra (evaluation) sẽ gần như diễn ra tức thì.

## 4. Tối ưu PyTorch Compilation (Dành cho PyTorch 2.x)
Sử dụng `torch.compile(model)` để JIT biên dịch mô hình `SimpleNN` thành biểu đồ tính toán tối ưu, giảm overhead của Python.

## Thay đổi dự kiến
- **[MODIFY]** `fl_core/models.py`: Thêm `device` và thiết lập hàm sao chép weight nằm gọn trên GPU.
- **[MODIFY]** `fl_core/worker.py`: Gửi dữ liệu và model lên thiết bị GPU, bật `torch.compile`.
- **[MODIFY]** `fl_core/simulator.py`: Thêm đa luồng `ThreadPoolExecutor` trong hàm `sync_run_step` và xử lý logic test accuracy thuần tensor.
- **[MODIFY]** `fl_core/aggregator.py`: Sửa việc khởi tạo cache để sử dụng GPU tensors thay vì fallback về `.cpu()`.

---
> [!IMPORTANT]
> **User Review Required:** Xin bạn phản hồi xem có đồng ý để tôi bắt tay vào triển khai các sửa đổi **[MODIFY]** theo kế hoạch tăng tốc bằng GPU và Đa luồng này không?
