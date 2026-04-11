# Tổng Kết Triển Khai: AU Detection System

Dự án đã hoàn thành việc chuyển đổi architecture CLIP-ReID sang hệ thống nhận diện Facial Action Unit (AU) và giải thích bằng ngôn ngữ tự nhiên.

## Các Thành Phần Đã Triển Khai

| Component | Files | Description |
|---|---|---|
| **Data Processing** | [prepare_data.py](file:///d:/CLIP-ReID/prepare_data.py) | Xử lý DISFA labels thô (intensity) sang CSV binary (activation >= 2). |
| **Dataset Loader** | [datasets/disfa.py](file:///d:/CLIP-ReID/datasets/disfa.py) | Lớp Dataset chuyên biệt cho DISFA, hỗ trợ multi-label. |
| **Transforms** | [datasets/preprocessing.py](file:///d:/CLIP-ReID/datasets/preprocessing.py) | Thêm AU-specific transforms (face-safe, bỏ RandomErasing). |
| **Model** | [model/au_head.py](file:///d:/CLIP-ReID/model/au_head.py)<br>[model/make_model.py](file:///d:/CLIP-ReID/model/make_model.py) | Thay classifier ReID bằng [AUHead](file:///d:/CLIP-ReID/model/au_head.py#11-19) (12 classes). Output sigmoid khi inference. |
| **Loss Function** | [loss/au_loss.py](file:///d:/CLIP-ReID/loss/au_loss.py)<br>[loss/make_loss.py](file:///d:/CLIP-ReID/loss/make_loss.py) | Triển khai [WeightedBCELoss](file:///d:/CLIP-ReID/loss/au_loss.py#4-24) để xử lý class imbalance trong DISFA. |
| **Training Pipeline** | [processor/processor_au.py](file:///d:/CLIP-ReID/processor/processor_au.py)<br>[train_au.py](file:///d:/CLIP-ReID/train_au.py) | Loop huấn luyện đặc thù cho AU và [AUEvaluator](file:///d:/CLIP-ReID/processor/processor_au.py#11-56) (F1, AUC metrics). |
| **Explanation** | [au_explainer.py](file:///d:/CLIP-ReID/au_explainer.py) | Module Rule-based chuyển AU vector sang câu mô tả tiếng Anh và cảm xúc. |
| **Inference** | [inference_au.py](file:///d:/CLIP-ReID/inference_au.py) | Full pipeline: Image → CLIP → AU Head → Explainer → Text. |

## Quy Trình Thực Hiện (Workflow)

### 1. Chuẩn bị dữ liệu
Chạy script để tạo file label CSV:
```bash
python prepare_data.py
```

### 2. Huấn luyện (Training)
Chạy script train mới với config AU:
```bash
python train_au.py --config_file configs/au/vit_base_au.yaml
```

### 3. Suy luận (Inference)
Chạy pipeline nhận diện và giải thích cho một ảnh bất kỳ:
```bash
python inference_au.py --image_path path/to/face.jpg --weight_path logs/au_vit_base/ViT-B-16_au_30.pth
```

## Những Thay Đổi Quan Trọng So Với ReID Gốc
- **Loại bỏ RandomErasing**: Đảm bảo không mất thông tin vùng mắt/miệng quan trọng cho AU.
- **Dùng Weighted BCE**: Thay thế cho CrossEntropy + Triplet Loss để phù hợp với multi-label và dữ liệu mất cân bằng.
- **Sigmoid Output**: Chuyển đổi logits sang xác suất độc lập cho từng AU thay vì softmax (vốn chỉ chọn 1 class).

---
*Dự án đã sẵn sàng để bạn chạy huấn luyện trên máy cá nhân.*


## Cập Nhật Mới Nhất: Transformer-based Landmark-guided Architecture

**Timestamp:** 2026-04-11 16:17:51

### 1. preprocessing_pipeline_changes
* **Module:** datasets/preprocessing.py, datasets/disfa.py
* **Thay đổi (Change Description):** Tích hợp extraction features mới và xử lý landmarks output từ MediaPipe, truyền cùng với Image qua pipeline Dataloader.
* **Tác động hiệu năng dự kiến (Expected Performance Impact):** Tăng nhẹ thời gian I/O lúc load dữ liệu nhưng đảm bảo pipeline ổn định, model có dữ liệu không gian trực tiếp thay vì tự học heuristic.

### 2. mediapipe_integration
* **Module:** datasets/preprocessing.py
* **Thay đổi (Change Description):** Tạo module MediaPipeFaceMeshExtractor trích xuất nhóm tọa độ ảnh thành 9 vùng chuẩn (left_eye, right_eye, lips, jaw...).
* **Tác động hiệu năng dự kiến (Expected Performance Impact):** Overhead nhỏ khi train do suy luận MediaPipe, tuy nhiên MediaPipe rất nhanh và giảm nhẹ được gánh nặng học hình học cho Transformer.

### 3. FRLP_module_added
* **Module:** model/au_modules.py
* **Thay đổi (Change Description):** Thêm class FaceRegionLandmarkProjector tạo ra một embedding vectors từ toạ độ MediaPipe landmarks.
* **Tác động hiệu năng dự kiến (Expected Performance Impact):** Giúp chuyển tiếp tokens landmark về cùng không gian vector của Transformer, cải thiện độ chính xác phân tích vùng mặt.

### 4. region_tokenization_added
* **Module:** model/make_model.py
* **Thay đổi (Change Description):** Áp dụng pipeline Tokenizer theo vùng từ FRLP làm dữ liệu Query/Key chéo cho các hàm Attention.
* **Tác động hiệu năng dự kiến (Expected Performance Impact):** Giảm thiểu độ hao hụt vùng nhận sự chú ý, làm sắc nét quá trình khoanh vùng các bó cơ AU.

### 5. FRGCA_attention_added
* **Module:** model/au_modules.py, model/make_model.py
* **Thay đổi (Change Description):** Thiết lập \FaceRegionGuidedCrossAttention\ áp đặt attention của hình ảnh gốc dựa trên token từ landmark.
* **Tác động hiệu năng dự kiến (Expected Performance Impact):** Cải thiện nhận diện Micro-expressions tăng mạnh vì vùng hình ảnh ngoại vi bị khử nhiễu nhanh chóng.

### 6. PTA_module_added
* **Module:** model/au_modules.py, model/make_model.py
* **Thay đổi (Change Description):** Thêm \PatchTokenAttention\ (weighted patch pooling) để gom gọn các token cục bộ về 12 dimensional token list đại diện 12 AU.
* **Tác động hiệu năng dự kiến (Expected Performance Impact):** Thay vì pooling trung bình (average pooling) mất thông tin, PTA bảo lưu được những activation siêu nhỏ của từng Action Unit cụ thể.

### 7. multi_level_features_enabled
* **Module:** model/make_model.py
* **Thay đổi (Change Description):** Tái xuất token layers ở nhiều tần số, bỏ giới hạn output CLS token và monkey-patch output ResBlocks \ & \.
* **Tác động hiệu năng dự kiến (Expected Performance Impact):** Thể hiện chi tiết cả cấu trúc semantic lớn và texture nhỏ, tăng F1 trên các class khó.

### 8. multi_scale_features_enabled
* **Module:** model/make_model.py
* **Thay đổi (Change Description):** Sử dụng các sequence output từ patch đa cấp của ViT encoder.
* **Tác động hiệu năng dự kiến (Expected Performance Impact):** Củng cố tính năng Multi-level Features cho độ sắc nét cao hơn trên nhiều scale khuôn mặt khác nhau.

### 9. relational_AU_modeling_added
* **Module:** model/au_modules.py
* **Thay đổi (Change Description):** Thiết lập AURelationalModeling áp dụng 2 tầng Transformer encoder thuần nội bộ (AU Token interaction).
* **Tác động hiệu năng dự kiến (Expected Performance Impact):** AU6 và AU12 (cười) luôn xuất hiện cùng nhau, modeling này tận dụng triệt để những co-occurrence dependencies đó.

### 10. RandReAU_training_enabled
* **Module:** model/make_model.py
* **Thay đổi (Change Description):** Trong mode \	raining\, thứ tự của AU tokens được shuffle qua \	orch.randperm(12)\ và un-shuffle trước logits xuất ra.
* **Tác động hiệu năng dự kiến (Expected Performance Impact):** Hạn chế positional bias của Transformer trong Relational Modeling module, giảm Overfitting.

### 11. LoRA_finetuning_enabled
* **Module:** model/make_model.py
* **Thay đổi (Change Description):** Bọc Image encoder (CLIP ViT) qua thư viện \peft\ với LoRA config cho \c_fc\ và \c_proj\ (rank=16).
* **Tác động hiệu năng dự kiến (Expected Performance Impact):** Trainable parameters nay dưới 5%, train siêu nhanh tiết kiệm VRAM mà giữ nguyên sức mạnh foundation model của CLIP.

### 12. RetinaFace_removed
* **Module:** Dự án tổng thể
* **Thay đổi (Change Description):** Thay thế pipeline nhận diện của RetinaFace (nếu có nằm ngầm định) hoàn toàn sang MediaPipe Face Mesh.
* **Tác động hiệu năng dự kiến (Expected Performance Impact):** Pipeline nhẹ hơn, chuyên nghiệp hơn cho local edge-devices và độ chi tiết landmarks từ MediaPipe 468 điểm tốt hơn đáng kể.
