# Kế Hoạch Triển Khai: CLIP-ReID → AU Detection + Natural Language Generation

## Tóm Tắt Mục Tiêu

Mở rộng CLIP-ReID để nhận diện 12 Facial Action Units (AUs) từ dữ liệu DISFA và tạo mô tả ngôn ngữ tự nhiên tương ứng. Pipeline cuối: `Face Image → CLIP Encoder → AU Vector → Natural Language`.

---

## ⚠️ Phân Tích Tính Khả Thi

### ✅ Những gì HOÀN TOÀN KHẢ THI

| Hạng mục | Lý do |
|---|---|
| Tái sử dụng CLIP Image Encoder | Code [make_model.py](file:///d:/CLIP-ReID/model/make_model.py) và [make_model_clipreid.py](file:///d:/CLIP-ReID/model/make_model_clipreid.py) đã có sẵn, chỉ cần thay classifier |
| BNNeck (`bottleneck`) | Đã tồn tại trong [build_transformer](file:///d:/CLIP-ReID/model/make_model_clipreid.py#52-167), giữ nguyên |
| Hai giai đoạn training | Cấu trúc [processor_clipreid_stage1.py](file:///d:/CLIP-ReID/processor/processor_clipreid_stage1.py) + [processor_clipreid_stage2.py](file:///d:/CLIP-ReID/processor/processor_clipreid_stage2.py) đã có sẵn |
| `au_explainer.py` | Module đơn giản, rule-based, không cần ML |
| Weighted BCE loss | Dễ implement bằng `torch.nn.BCEWithLogitsLoss` |

### ⚠️ Những gì CẦN CHÚ Ý ĐẶC BIỆT

#### 1. Cấu trúc Labels DISFA thực tế KHÁC với spec

> **Spec mô tả:**
> ```
> Labels/SN001/Trial_1/AU1.txt
> ```
> **Thực tế trong project:**
> ```
> Labels/SN001/A1_AU1_TrailNo_1/AU1.txt
> Labels/SN001/A2_AU2_TrailNo_2/AU12.txt
> ...
> ```
> Mỗi subject (`SN001`, `SN003`...) có **~65 trial folders** với tên phức tạp như `A1_AU1_TrailNo_1`, `Y_HappyDescribed_TrailNo_1`, v.v. `prepare_data.py` phải xử lý đúng cấu trúc này.

#### 2. Format file Label đã dùng `frame.jpg` thay vì frame_id số

> **Spec mô tả:** `000 0` (frame_id integer)
> **Thực tế:** `000.jpg     0` (filename + intensity)
> `prepare_data.py` cần parse format này.

#### 3. Two-stage training cho AU detection phức tạp hơn ReID

> Trong CLIP-ReID gốc, [PromptLearner](file:///d:/CLIP-ReID/model/make_model_clipreid.py#191-239) tạo prompt per-identity (`num_class = số người`). Với AU detection, prompt phải per-AU hoặc per-frame. Cần thiết kế lại [PromptLearner](file:///d:/CLIP-ReID/model/make_model_clipreid.py#191-239) để phù hợp với multi-label classification.

#### 4. [PromptLearner](file:///d:/CLIP-ReID/model/make_model_clipreid.py#191-239) trong [make_model_clipreid.py](file:///d:/CLIP-ReID/model/make_model_clipreid.py) gắn chặt với ReID

> [PromptLearner](file:///d:/CLIP-ReID/model/make_model_clipreid.py#191-239) dùng `label` là person_id (single label). Với AU, label là vector 12 chiều binary. Cần tạo `AUPromptLearner` mới hoặc thay đổi cách tạo prompt.

### ❌ Những gì KHÔNG KHẢ THI / CẦN THAY ĐỔI THIẾT KẾ

| Vấn đề | Giải pháp đề xuất |
|---|---|
| Two-stage training dùng `Identity Loss (L_ID)` — không phù hợp với multi-label AU | Ở Stage 1: dùng Weighted BCE thay L_ID; giữ L_ITC (image-text contrastive) |
| [PromptLearner](file:///d:/CLIP-ReID/model/make_model_clipreid.py#191-239) tạo prompt per-class (1 prompt per person) | Với AU: tạo 12 prompts cố định, 1 per AU (ví dụ: "A face showing AU1...") |
| Metric `R1_mAP_eval` trong [processor.py](file:///d:/CLIP-ReID/processor/processor.py) là cho ReID | Cần viết `AUEvaluator` với F1, AUC, Accuracy per AU |

---

## Proposed Changes

### 1 — Chuẩn Bị Dữ Liệu

#### [NEW] [prepare_data.py](file:///d:/CLIP-ReID/prepare_data.py)

Script tạo `AUs_DATA/labels.csv` từ dữ liệu DISFA thô.

**Logic xử lý:**
```python
AU_LIST = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
SUBJECTS = ['SN001', 'SN003', 'SN004', 'SN007', ...]

for subject in SUBJECTS:
    for trial_folder in os.listdir(f"AUs_DATA/Labels/{subject}"):
        # Đọc tất cả AU*.txt trong trial_folder
        # Parse format: "000.jpg     0"
        # Convert intensity >= 2 → binary label 1
        # Merge 12 AU vectors per frame
        # Append to CSV với image_path = subject/trial_folder/frame.jpg
```

**Output CSV format:**
```
image_path,AU1,AU2,AU4,AU5,AU6,AU9,AU12,AU15,AU17,AU20,AU25,AU26
SN001/A1_AU1_TrailNo_1/000.jpg,0,0,0,0,0,0,0,0,0,0,0,0
```

> [!IMPORTANT]
> Cần xử lý trường hợp một số trial không có đủ 12 AU files (thiếu AU → default = 0).

---

#### [NEW] [datasets/disfa.py](file:///d:/CLIP-ReID/datasets/disfa.py)

DISFA Dataset class kế thừa `torch.utils.data.Dataset`.

```python
class DISFADataset(Dataset):
    def __init__(self, csv_path, img_root, transform=None, split='train'):
        self.df = pd.read_csv(csv_path)
        # train/val split: 80/20 theo subject
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = PIL.Image.open(row['image_path'])
        au_label = torch.tensor(row[AU_COLS].values.astype(float), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, au_label
```

---

#### [MODIFY] [datasets/make_dataloader.py](file:///d:/CLIP-ReID/datasets/make_dataloader.py)

Thêm hàm `make_au_dataloader(cfg)` độc lập cho AU task. Xem chi tiết transforms ở Section 1.5 bên dưới.

---

### 1.5 — Tiền Xử Lý Ảnh (Image Preprocessing)

> [!IMPORTANT]
> **Phân tích ảnh DISFA thực tế:** Mỗi file ảnh trong `AUs_DATA/Images/` có kích thước **~200KB** — đây là ảnh màu độ phân giải cao, KHÔNG phải thumbnail. Quan sát cấu trúc tên thư mục (e.g., `A1_AU1_TrailNo_1`) — đây là video recording của từng subject, ảnh **đã là full face frame**, chưa được crop/align.

#### Quyết Định Thiết Kế: Có Cần Face Detection Không?

| Phương án | Ưu điểm | Nhược điểm | Quyết định |
|---|---|---|---|
| **Dùng RetinaFace/MTCNN** | Chuẩn hóa vùng mặt, loại background | Phụ thuộc thư viện nặng, chậm, có thể fail | ⚠️ Tùy chọn |
| **Chỉ Resize + Crop** | Đơn giản, nhanh, không phụ thuộc | Background còn lại nếu ảnh chưa crop | ✅ Mặc định ban đầu |

**Khuyến nghị:** Bắt đầu với resize đơn giản. DISFA là dataset chuẩn, ảnh thường đã framing mặt khá tốt. Thêm face detection nếu kết quả training kém.

---

#### [MODIFY] [datasets/preprocessing.py](file:///d:/CLIP-ReID/datasets/preprocessing.py)

`preprocessing.py` hiện tại **chỉ có `RandomErasing`** — không phù hợp cho AU detection vì:
- `RandomErasing` che đi vùng mặt ngẫu nhiên → xóa mất thông tin AU quan trọng (mắt, miệng, mũi)
- Không có CLIP-specific normalization
- Không có face-safe augmentation

**Cần thêm vào `preprocessing.py`:**

```python
import torchvision.transforms as T
from PIL import Image

# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

def build_au_train_transforms(cfg):
    """
    Face-safe augmentation pipeline cho AU detection.
    Tránh các transforms có thể thay đổi cấu trúc cơ bắp mặt.
    """
    return T.Compose([
        T.Resize((cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])),
        T.RandomHorizontalFlip(p=0.5),
        # Chỉ xoay nhẹ, tránh làm sai vị trí AU
        T.RandomRotation(degrees=5),
        # Thay đổi màu sắc nhẹ
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.0        # KHÔNG thay đổi hue — AU phụ thuộc màu da
        ),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        # KHÔNG dùng RandomErasing — sẽ xóa vùng mắt/miệng mang thông tin AU
    ])

def build_au_val_transforms(cfg):
    """Chỉ resize + normalize, không augmentation."""
    return T.Compose([
        T.Resize((cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
```

**Lý do loại bỏ từng transform so với ReID gốc:**

| Transform | ReID gốc | AU Detection | Lý do |
|---|---|---|---|
| `RandomErasing` | ✅ Dùng | ❌ Bỏ | Xóa vùng mặt mang thông tin AU |
| `RandomCrop` + `Padding` | ✅ Dùng | ❌ Bỏ | Cắt sát mép có thể mất vùng brow/jaw |
| `RandomHorizontalFlip` | ✅ | ✅ | An toàn (AU đối xứng) |
| `RandomRotation(5°)` | ❌ | ✅ Thêm | Simulate góc chụp khác nhau nhẹ |
| `ColorJitter` | ❌ | ✅ Thêm | Tăng robustness với lighting |
| CLIP normalization | ✅ | ✅ | Bắt buộc với CLIP encoder |

---

#### (Tùy Chọn) Face Detection & Alignment

Nếu cần thêm face detection, thêm bước **pre-processing offline** (chạy 1 lần, lưu ảnh đã crop):

```python
# offline_face_crop.py (chạy 1 lần trước training)
from retinaface import RetinaFace  # pip install retina-face

def crop_and_save(input_dir, output_dir):
    for img_path in all_images:
        faces = RetinaFace.detect_faces(img_path)
        if faces:
            face = faces['face_1']
            x1, y1, x2, y2 = face['facial_area']
            cropped = Image.open(img_path).crop((x1, y1, x2, y2))
            cropped.save(output_path)
```

> [!NOTE]  
> Không nhúng face detection vào runtime DataLoader — quá chậm khi training. Thực hiện offline và lưu ảnh đã crop vào `AUs_DATA/Images_Cropped/`.

---

### 2 — Model Architecture

#### [NEW] [model/au_head.py](file:///d:/CLIP-ReID/model/au_head.py)

```python
class AUClassificationHead(nn.Module):
    def __init__(self, feature_dim, num_aus=12):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_aus)
        # Không dùng Sigmoid ở đây, dùng trong loss (BCEWithLogitsLoss)
        
    def forward(self, feat):
        return self.classifier(feat)
```

---

#### [MODIFY] [model/make_model.py](file:///d:/CLIP-ReID/model/make_model.py)

Thay `nn.Linear(in_planes, num_classes)` bằng `AUClassificationHead(in_planes, 12)`.

**Thay đổi `forward()`:**
- Training: trả về `au_logits` (shape: `[B, 12]`) và `features`
- Inference: trả về `au_probabilities` = `sigmoid(au_logits)`

**Giữ nguyên:** `BNNeck`, `CLIP Image Encoder`, `load_param()`, `load_param_finetune()`

---

### 3 — Loss Function

#### [NEW] [loss/au_loss.py](file:///d:/CLIP-ReID/loss/au_loss.py)

```python
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weights):
        # pos_weights: tensor shape [12] = N_total / (2 * N_positive_i)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        
    def forward(self, logits, targets):
        return self.bce(logits, targets)
```

**Tính `pos_weights` từ `labels.csv`:**
```python
pos_counts = df[AU_COLS].sum()
pos_weights = len(df) / (2 * pos_counts)
```

---

#### [MODIFY] [loss/make_loss.py](file:///d:/CLIP-ReID/loss/make_loss.py)

Thêm nhánh cho AU detection:
```python
if cfg.DATASETS.NAMES == 'disfa':
    return make_au_loss(cfg, labels_csv_path)
```

---

### 4 — Processor

#### [NEW] [processor/processor_au.py](file:///d:/CLIP-ReID/processor/processor_au.py)

Training loop mới thay `R1_mAP_eval` bằng AU evaluator:

```python
class AUEvaluator:
    def update(self, preds, targets):
        # preds: sigmoid output, threshold=0.5
        # targets: binary labels
        
    def compute(self):
        # Per-AU F1 score
        # Average F1
        # Per-AU AUC
        # Overall Accuracy
        return metrics_dict
```

**Training loop thay đổi:**
- `(img, au_label)` thay cho `(img, vid, target_cam, target_view)`
- Loss: `WeightedBCE(au_logits, au_label)`
- Không cần `center_criterion`, `optimizer_center`

---

### 5 — Config

#### [NEW] [configs/au/vit_base_au.yaml](file:///d:/CLIP-ReID/configs/au/vit_base_au.yaml)

```yaml
MODEL:
  NAME: 'ViT-B-16'
  NECK: 'bnneck'
DATASETS:
  NAMES: 'disfa'
  ROOT_DIR: 'AUs_DATA'
INPUT:
  SIZE_TRAIN: [224, 224]
  PIXEL_MEAN: [0.48145466, 0.4578275, 0.40821073]
  PIXEL_STD: [0.26862954, 0.26130258, 0.27577711]
SOLVER:
  MAX_EPOCHS: 30
  BASE_LR: 0.00035
```

---

### 6 — Natural Language Generation

#### [NEW] [au_explainer.py](file:///d:/CLIP-ReID/au_explainer.py)

```python
AU_PHRASES = {
    1: "raises the inner brows",
    2: "raises the outer brows",
    4: "lowers the brows",
    6: "raises the cheeks",
    12: "pulls the lip corners upward",
    25: "parts the lips",
    26: "drops the jaw",
    ...
}

EMOTION_RULES = {
    frozenset([6, 12]): "happiness",
    frozenset([4, 15]): "sadness",
    frozenset([1, 2, 5]): "surprise",
    frozenset([4, 7, 23]): "anger",
    ...
}

def generate_description(au_vector: list[int]) -> str:
    active_aus = [AU_LIST[i] for i, v in enumerate(au_vector) if v == 1]
    phrases = [AU_PHRASES[au] for au in active_aus if au in AU_PHRASES]
    emotion = infer_emotion(active_aus)
    return build_sentence(phrases, emotion)
```

> [!NOTE]
> `au_explainer.py` là module **độc lập, không cần training**. Tính khả thi cao nhất trong toàn bộ spec — chỉ cần logic if-else đơn giản.

---

### 7 — Train Entry Point

#### [NEW] [train_au.py](file:///d:/CLIP-ReID/train_au.py)

Entry point mới tương tự `train.py` nhưng gọi `make_au_dataloader`, `make_au_loss`, `processor_au.py`.

---

## Kế Hoạch Đánh Giá Tính Khả Thi Tổng Thể

| Công việc | Độ khó | Khả thi? | Ghi chú |
|---|---|---|---|
| `prepare_data.py` | Trung bình | ✅ Cao | Cần xử lý đúng cấu trúc thư mục thực tế |
| `datasets/disfa.py` | Dễ | ✅ Cao | Dataset class chuẩn |
| `model/au_head.py` | Dễ | ✅ Cao | Chỉ thay Linear layer |
| `loss/au_loss.py` | Dễ | ✅ Cao | BCEWithLogitsLoss có sẵn trong PyTorch |
| `processor_au.py` | Trung bình | ✅ Cao | Thay evaluator, simplify training loop |
| `au_explainer.py` | Dễ | ✅ Rất Cao | Rule-based, không cần ML |
| Two-stage training | Khó | ⚠️ Trung bình | `PromptLearner` cần thiết kế lại cho multi-label |
| Image preprocessing (resize + augmentation) | Dễ | ✅ Cao | Dùng face-safe transforms, bỏ RandomErasing |
| Face detection offline (RetinaFace) | Trung bình | ⚠️ Tùy chọn | Chỉ cần nếu kết quả training kém |
| `datasets/preprocessing.py` | Dễ | ✅ Cao | Thêm `build_au_train_transforms()` và `build_au_val_transforms()` |

> [!WARNING]
> **Two-stage training** là phần phức tạp nhất. `PromptLearner` trong `make_model_clipreid.py` được thiết kế cho single-label ReID. Với multi-label AU, strategy mapping "label → prompt" cần thiết kế lại.
> **Đề xuất:** Giai đoạn đầu, bỏ qua two-stage, dùng fine-tuning đơn giản (Stage 2 only). Sau đó add Stage 1 nếu cần.

> [!CAUTION]
> **Không dùng `RandomErasing`** (từ `preprocessing.py` gốc) trong AU pipeline. `RandomErasing` che khuất vùng mặt ngẫu nhiên — có thể xóa chính xác vùng cơ mặt mang thông tin AU (mắt, mày, miệng) làm nhiễu training.

---

## Verification Plan

### 1. Kiểm tra `prepare_data.py`
```bash
cd d:\CLIP-ReID
python prepare_data.py
```
Kiểm tra: File `AUs_DATA/labels.csv` được tạo ra. Mở và verify format đúng với header `image_path,AU1,...,AU26` và các giá trị 0/1.

### 2. Kiểm tra `au_explainer.py`
```bash
python -c "
from au_explainer import generate_description
vec = [1,1,0,0,1,0,1,0,0,0,1,0]
print(generate_description(vec))
"
```
Expected output: Câu mô tả tiếng Anh có chứa "inner brow", "cheeks", "lip corners", "lips".

### 3. Kiểm tra dataset loader
```bash
python -c "
from datasets.disfa import DISFADataset
import torchvision.transforms as T
ds = DISFADataset('AUs_DATA/labels.csv', 'AUs_DATA/Images')
img, label = ds[0]
print('Image shape:', img.size)
print('AU label shape:', label.shape)  # Expected: torch.Size([12])
"
```

### 4. Chạy training smoke test (1 epoch)
```bash
python train_au.py --config_file configs/au/vit_base_au.yaml SOLVER.MAX_EPOCHS 1
```
Kiểm tra: Training loop chạy không lỗi, loss in ra màn hình và có giá trị hợp lý (< 10.0).

### 5. Kiểm tra model forward pass
```bash
python -c "
import torch
from model.make_model import make_model
# Mock cfg và test forward pass với dummy input
img = torch.randn(2, 3, 224, 224).cuda()
# Expected output shape: [2, 12]
"
```
