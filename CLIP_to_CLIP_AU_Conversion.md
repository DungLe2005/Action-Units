# Quá Trình Chuyển Đổi Mô Hình CLIP Thành CLIP Action Unit (CLIP-AU)

Tài liệu này trình bày chi tiết từng bước được thực hiện để chuyển đổi mô hình CLIP gốc (vốn được thiết kế cho bài toán Contrastive Learning giữa ảnh và văn bản) thành một hệ thống nhận diện Action Unit trên khuôn mặt đa nhãn (Multi-label Classification).

---

## Bước 1: Xác định lại bài toán và kiến trúc đầu ra

- **Mô hình gốc:** CLIP lấy một bức ảnh và một đoạn văn bản làm đầu vào, cho ra độ tương đồng (similarity) giữa chúng.
- **Mô hình mong muốn (CLIP-AU):** Lấy một bức ảnh khuôn mặt làm đầu vào và xuất ra một vector xác suất cho nhiều nhãn Action Unit (AU) cùng lúc (VD: AU1, AU2, AU4...). Lớp hoàn thành phải hỗ trợ **Multi-label Classification**.

## Bước 2: Tải Backbone Vision Encoder của CLIP

- **Cách thực hiện:** Thay vì tải toàn bộ mã nguồn CLIP gốc chứa cả Text Encoder và Image Encoder, hệ thống sử dụng module `CLIPVisionModel` từ thư viện `transformers` của HuggingFace.
- **Code liên quan (trong `models/clip_au.py`):**
  ```python
  self.backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
  ```
- **Mục đích:** Chỉ lấy phần mạng Neural Network xử lý hình ảnh (Vision Transformer - ViT) vì bài toán AU chỉ nhận đầu vào là ảnh khuôn mặt, không dùng đến văn bản tự nhiên.

## Bước 3: Loại bỏ hoàn toàn Text Encoder

- **Cách thực hiện:** Bằng việc chỉ khai báo `CLIPVisionModel` ở Bước 2, toàn bộ nhánh Text Encoder của CLIP gốc đã bị loại bỏ một cách triệt để.
- **Mục đích:** Giảm thiểu một nửa lượng VRAM (bộ nhớ GPU) cần thiết và tăng tốc độ xử lý, vì nhánh Text Encoder là không cần thiết cho bài toán phân loại hình ảnh thuần túy.

## Bước 4: Thêm Khối Phân Loại Đa Nhãn (Classification Head)

- **Cách thực hiện:** Embedding Vector đầu ra từ CLIP Vision (kích thước `768` chiều đối với bản ViT-B/32) được nối với một mạng nơ-ron truyền thẳng (Feed Forward Network) tùy chỉnh.
- **Cấu trúc:**
  1. `nn.Linear(768, 512)`: Giảm chiều dữ liệu từ đặc trưng của CLIP xuống không gian của AU.
  2. `nn.ReLU()`: Hàm kích hoạt phi tuyến.
  3. `nn.Dropout(0.3)`: Chống over-fitting (vật lộn học thuộc lòng tập train).
  4. `nn.Linear(512, num_classes)`: Lớp xuất ra số lượng giá trị (logits) bằng đúng với số lượng AU cần dự đoán.
- **Mục đích:** Chuyển đổi không gian đặc trưng chung chung của CLIP thành không gian đặc trưng chuyên biệt cho từng AU độc lập.

## Bước 5: Cấu hình Hàm Mất Mát (Loss Function) cho Đa Nhãn

- **Cách thực hiện:** Sử dụng hàm `BCEWithLogitsLoss` (Binary Cross Entropy with Logits) thay thế cho hàm Contrastive Loss ban đầu của CLIP.
- **Mục đích:** `BCEWithLogitsLoss` tự động áp dụng hàm Sigmoid lên từng điểm ảnh trong vector đầu ra, biến đổi chúng thành xác suất độc lập (từ 0 đến 1) cho từng AU. Điều này cho phép một bức ảnh có thể không có AU nào, có 1 AU, hoặc hiển thị nhiều AU cùng một lúc.

## Bước 6: Điều chỉnh Data Pipeline cho phù hợp với CLIP

- **Cách thực hiện:** CLIP đã được huấn luyện với một chuẩn đầu vào ảnh nghiêm ngặt. Hệ thống `dataset.py` được xây dựng để bắt buộc ảnh tuân theo chuẩn này:
  - Resize/Crop về kích thước `224x224`.
  - Normalize với `mean=[0.48145466, 0.4578275, 0.40821073]` và `std=[0.26862954, 0.26130258, 0.27577711]`.
- **Mục đích:** Giữ nguyên được chất lượng các trọng số pre-trained của CLIP để trích xuất đặc trưng chính xác nhất mà không bị méo lệch màu sắc hay kích thước.

## Bước 7: Thêm cơ chế Đóng Băng (Freeze) và Fine-tuning

- **Cách thực hiện:** Tạo tùy chọn (tham số `freeze_backbone` trong cấu hình) cho phép lập trình viên khóa (freeze) toàn bộ trọng số của CLIP Vision Encoder (`requires_grad = False`).
- **Mục đích:** Tùy thuộc vào lượng dữ liệu AU có sẵn:
  - Nếu dữ liệu ít, chỉ huấn luyện (train) phần Classification Head ở Bước 4 (chống vỡ trọng số tốt của CLIP).
  - Nếu dữ liệu lớn, bỏ đóng băng để Fine-tune toàn bộ hệ thống cho độ chính xác cao nhất.

## Bước 8: Viết kịch bản Đánh Giá (Evaluation) và Inference

- **Cách thực hiện:** Chuyển đổi xác suất thu được ở Bước 5 thành bộ nhãn nhị phân `0`/`1` thông qua một ngưỡng đánh giá (`threshold = 0.5`).
- **Mục đích:** Tính toán được đúng ý nghĩa của các chỉ số F1-Score, Precision (Độ chính xác) và Recall (Độ bao phủ) dành cho kiểm định khoa học của Action Unit.
