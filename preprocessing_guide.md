# Cẩm nang Cải thiện Pipeline Tiền xử lý Dữ liệu cho Nhận diện Action Unit (AU)

Dự án hiện tại đang gặp vấn đề **mô hình dự đoán ra xác suất rất thấp (gần 0)** cho tất cả các AU. Nguyên nhân lớn nhất đến từ **pipeline tiền xử lý hình ảnh (data preprocessing)** chưa được tối ưu cho bài toán nhận diện cử động khuôn mặt (Facial Action Units).

Dưới đây là phân tích chi tiết các vấn đề và giải pháp kèm code thực hành để bạn xây dựng lại `dataset.py` chuẩn chỉ.

---

## 1. Phân tích Các Vấn đề Hiện tại

1. **Chưa có phát hiện và cắt khuôn mặt (No Face Detection/Cropping)**:
   - _Hiện tại:_ Đưa cả bức ảnh gốc (chứa tóc, cổ, hậu cảnh rối rắm) vào CLIP.
   - _Vấn đề:_ Mô hình sẽ bị phân tâm bởi background. Với AU, chúng ta chỉ cần tập trung cực kỳ chi tiết vào các cơ trên khuôn mặt.
2. **Resize sai cách gây méo mặt (Incorrect Resizing)**:
   - _Hiện tại:_ `Resize((224, 224))` ép thẳng ảnh thành hình vuông.
   - _Vấn đề:_ Thay đổi tỷ lệ khung hình (aspect ratio) làm biến dạng hình học của khuôn mặt và các cơ, khiến mô hình học sai đặc trưng.
3. **Data Augmentation quá đà (Aggressive Augmentation)**:
   - _Hiện tại:_ Dùng `RandomRotation(10)` hoặc các phép biến đổi hình học mạnh.
   - _Vấn đề:_ Việc xoay/kéo giãn quá mức làm phá vỡ vị trí tương đối của các cơ trên mặt (ví dụ: AU15 - nhếch mép xuống sẽ bị nhầm lẫn nếu ảnh bị xoay méo).
4. **Xử lý lỗi đọc ảnh bằng ảnh đen (Bad Error Handling)**:
   - _Hiện tại:_ `try-except` sinh ra ảnh `Image.new('RGB', (224, 224))` khi lỗi.
   - _Vấn đề:_ Đưa ảnh đen tinh với nhãn thực tế ghép vào sẽ làm nhiễu trầm trọng quá trình huấn luyện của mô hình.

---

## 2. Giải pháp Cải thiện & Khuyến nghị

### A. Tiền xử lý cắt khuôn mặt (Nên làm Offline)

Thay vì chạy Face Detection (như MTCNN, RetinaFace, MediaPipe) trong lúc load DataLoader (rất chậm), **hãy viết một script chạy 1 lần duy nhất** để quét toàn bộ dataset ban đầu, cắt ra vùng khuôn mặt (có cộng thêm một khoảng lề - padding) và lưu thành thư mục data mới (ví dụ: `AUs_DATA_Cropped`). Sau đó trỏ DataLoader vào thư mục mới này.

### B. Resize giữ nguyên tỷ lệ

Chuyển từ `Resize((224, 224))` sang:

```python
transforms.Resize(256),
transforms.CenterCrop(224)
```

### C. Augmentation an toàn (Safe Augmentations)

Chỉ dùng các phép biến đổi nhẹ nhàng:

- Lật ngang: `RandomHorizontalFlip(p=0.5)` (Cần cẩn thận nếu có các AU không đối xứng, nhưng thường là an toàn).
- Thay đổi màu/độ sáng nhẹ: `ColorJitter(brightness=0.2, contrast=0.2)`.
- **Tuyệt đối tránh**: Xoay quá 5 độ, RandomResizedCrop với scale quá nhỏ làm mất mặt.

### D. Xử lý ảnh lỗi chuẩn (Safe Loading & Collation)

Trong `__getitem__`, nếu lỗi, trả về `None`. Sau đó dùng một hàm `collate_fn` tùy chỉnh trong DataLoader để lọc các mẫu `None` này đi ra khỏi batch.

---

## 3. Bản viết lại hoàn chỉnh cho `dataset.py`

Dưới đây là đoạn code đã được tối ưu lại theo chuẩn Senior CV Engineer:

```python
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

class AUDataset(Dataset):
    """Action Unit (AU) Dataset."""

    def __init__(self, df, image_dir, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame chứa đường dẫn ảnh và nhãn AU.
            image_dir (string): Thư mục chứa toàn bộ ảnh (Nên là data đã được Crop khuôn mặt).
            transform (callable, optional): Các phép biến đổi áp dụng lên ảnh.
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

        # Cột đầu tiên là đường dẫn ảnh tương đối, phần còn lại là nhãn AUs
        self.image_names = self.df.iloc[:, 0].values
        self.labels = self.df.iloc[:, 1:].values.astype(float)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.image_dir, str(self.image_names[idx]))

        try:
            # Đọc ảnh và chuyển đổi sang RGB an toàn
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Nếu lỗi (ảnh hỏng/không tồn tại), CẦN TRẢ VỀ None, không sinh mẫu giả.
            print(f"Warning: Lỗi đọc ảnh {img_path}: {e}")
            return None

        # Đảm bảo label là tensor float32 cho binary/multi-label classification
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

def safe_collate_fn(batch):
    """
    Collate function lọc bỏ các mẫu bị lỗi (None) khỏi batch trước khi đưa vào model.
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor() # Batch rỗng nếu toàn bộ ảnh hỏng
    return default_collate(batch)

def get_transforms(image_size=224, is_train=True):
    """
    Lấy Pipeline Transforms an toàn cho Face & CLIP.
    CLIP Mean: [0.48145466, 0.4578275, 0.40821073], Std: [0.26862954, 0.26130258, 0.27577711]
    """
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    # Dự trù kích thước để CenterCrop
    resize_size = int(image_size * Response_Ratio) if hasattr(transforms, 'Resize') else int(image_size / 0.875)
    resize_size = 256 if image_size == 224 else int(image_size * 1.14)

    if is_train:
        return transforms.Compose([
            # Giữ tỷ lệ khuôn mặt, tránh ép vuông (Squash)
            transforms.Resize(resize_size),
            transforms.RandomCrop(image_size),

            # Augmentation An Toàn (Mức độ nhẹ)
            transforms.RandomHorizontalFlip(p=0.5),

            # Giới hạn xoay rất nhỏ để không làm mất cấu trúc hàm / nhếch mép
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),

            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std)
        ])
    else:
        return transforms.Compose([
            # Validation/Test: Luôn dùng CenterCrop để giữ nguyên góc mặt đẹp nhất
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std)
        ])

def create_dataloaders(params):
    """
    Creates train, val, and test dataloaders an toàn.
    """
    csv_path = params['data']['annotation_file']
    image_dir = params['data']['image_dir']
    batch_size = params['train']['batch_size']
    image_size = params['data']['image_size']
    split_ratios = params['data']['split_ratios']
    seed = params['train'].get('seed', 42)
    num_workers = params['train'].get('num_workers', 4)

    if not os.path.exists(csv_path):
        print(f"Error: Không tìm thấy file nhãn tại {csv_path}")
        return None, None, None, None

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Lỗi: CSV rỗng")
        return None, None, None, None

    # Chia tập dữ liệu ngẫu nhiên (nếu imbalanced quá, cần cân nhắc Iterative Stratification)
    val_test_ratio = split_ratios[1] + split_ratios[2]

    train_df, val_test_df = train_test_split(df, test_size=val_test_ratio, random_state=seed)

    test_ratio = split_ratios[2] / val_test_ratio
    val_df, test_df = train_test_split(val_test_df, test_size=test_ratio, random_state=seed)

    num_classes = train_df.shape[1] - 1
    print(f"Tổng số ảnh: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Số lượng Action Units (Classes): {num_classes}")

    # Khởi tạo Transforms
    train_transform = get_transforms(image_size, is_train=True)
    val_transform = get_transforms(image_size, is_train=False)

    # Khởi tạo Datasets
    train_dataset = AUDataset(train_df, image_dir, transform=train_transform)
    val_dataset = AUDataset(val_df, image_dir, transform=val_transform)
    test_dataset = AUDataset(test_df, image_dir, transform=val_transform)

    # Khởi tạo DataLoaders (thêm collate_fn=safe_collate_fn)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=safe_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, collate_fn=safe_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, collate_fn=safe_collate_fn)

    return train_loader, val_loader, test_loader, num_classes
```

---

## 4. Các cấu hình giúp Train ổn định hơn (Stability Tips)

1. **Class Imbalance (Mất cân bằng dữ liệu):**
   - Bài toán AUs luôn gặp tình trạng mất cân bằng nghiệm trọng (Positive samples rất ít so với Negative). Bạn nên tích hợp thêm tỷ trọng vào hàm Loss (**BCEWithLogitsLoss** có thuộc tính `pos_weight`). Tính tổng số nhãn dương và âm trên tập train để tính toán `pos_weight`.
2. **Offline Face Cropping:**
   - Việc sinh một thư mục Dataset chỉ chứa khuôn mặt được cắt bằng **MTCNN** hay **MediaPipe Face Mesh** sẽ giúp độ chính xác của mô hình nhảy vọt. Bạn không nên đưa ảnh cả background vào train.
3. **Kiểm tra Learning Rate:**
   - Hãy chắc chắn bạn đang dùng `Learning Rate` phân tầng. Lớp CLIP base chỉ nên train với LR rất bé (ví dụ $10^{-6}$), trong khi các lớp Classifier / Linear mới thì train với ($10^{-4}$).
