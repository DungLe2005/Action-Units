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
            df (pd.DataFrame): DataFrame chứa tên tệp hình ảnh và nhãn AU.
            image_dir (string): Thư mục chứa tất cả các hình ảnh.
            transform (callable, optional): Phép biến đổi tùy chọn được áp dụng
            trên một mẫu.
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        
        # Assume first column is filename, rest are AU labels
        self.image_names = self.df.iloc[:, 0].values
        self.labels = self.df.iloc[:, 1:].values.astype(float)
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.image_dir, str(self.image_names[idx]))
        
        # Bắt lỗi file ảnh không tồn tại để xử lý an toàn thay vì crash
        if not os.path.exists(img_path):
            return None
            
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Return None to be handled by safe_collate_fn instead of returning a black image
            print(f"Warning: Error loading image {img_path}: {e}")
            return None
            
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

def safe_collate_fn(batch):
    """
    Chức năng Collate dùng để lọc bỏ các mục None (hình ảnh tải không thành công).
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor()
    return default_collate(batch)

def get_transforms(image_size=224, is_train=True):
    """
    Lấy Transform torchvision cho CLIP.
    CLIP uses: Mean: [0.48145466, 0.4578275, 0.40821073], Std: [0.26862954, 0.26130258, 0.27577711]
    """
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    
    if is_train:
        return transforms.Compose([
            # Ảnh đã là 224x224 từ preprocess.py nên ta chỉ cần đảm bảo an toàn kích thước
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5), # An toàn cho hầu hết các khuôn mặt
            transforms.RandomRotation(degrees=5),   # Giữ cho độ xoay rất nhỏ để không làm biến dạng AU
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # Thêm phép xoay 180 độ (xoay ngược lại) do ảnh tiền xử lý bị ngược
            transforms.RandomRotation(degrees=(180, 180)),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std)
        ])

def create_dataloaders(params):
    """
    Tạo train, val và test dataloaders.
    Xử lý phân chia tầng (nếu có thể, nếu không thì phân chia ngẫu nhiên).
    """
    try:
        csv_path = params['data']['annotation_file']
        image_dir = params['data']['image_dir']
        batch_size = params['train']['batch_size']
        image_size = params['data']['image_size']
        split_ratios = params['data']['split_ratios']  # e.g. [0.7, 0.15, 0.15]
    except Exception as e:
        print(f"Error loading data config: {e}")
        return None, None, None, None
        
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print("CSV file is empty")
            return None, None, None, None
            
        # LỌC BỎ CÁC ẢNH KHÔNG CÒN TỒN TẠI (do bị loại ở bước tiền xử lý)
        print("Đang kiểm tra và lọc bỏ các file ảnh không tồn tại (do không nhận diện được viền mặt)...")
        # Giả định cột đầu tiên là filename (e.g. SN009/A7_AU5z_TrailNo_2/046.jpg)
        image_col = df.columns[0]
        valid_rows = []
        for idx, row in df.iterrows():
            img_path = os.path.join(image_dir, str(row[image_col]))
            if os.path.exists(img_path):
                valid_rows.append(True)
            else:
                valid_rows.append(False)
                
        df = df[valid_rows].reset_index(drop=True)
        print(f"Giữ lại {len(df)} ảnh hợp lệ có thật trên ổ cứng.")
        
        if df.empty:
            print("Toàn bộ ảnh trong CSV không tồn tại ở thư mục: ", image_dir)
            return None, None, None, None
            
    except Exception as e:
        print(f"Error loading CSV {csv_path}: {e}")
        return None, None, None, None
    
    # Phân chia đơn giản (Stratified phức tạp đối với multi-label, hiện tại sử dụng phân chia ngẫu nhiên
    # nhưng người dùng có thể thay thế bằng phân chia tầng lặp nếu cần)
    val_test_ratio = split_ratios[1] + split_ratios[2]
    
    train_df, val_test_df = train_test_split(
        df, test_size=val_test_ratio, random_state=params['train']['seed']
    )
    
    test_ratio = split_ratios[2] / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df, test_size=test_ratio, random_state=params['train']['seed']
    )
    
    num_classes = train_df.shape[1] - 1
    print(f"Total samples: {len(df)}, Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Number of Action Units (classes) detected: {num_classes}")
    
    train_transform = get_transforms(image_size, is_train=True)
    val_transform = get_transforms(image_size, is_train=False)
    
    train_dataset = AUDataset(train_df, image_dir, transform=train_transform)
    val_dataset = AUDataset(val_df, image_dir, transform=val_transform)
    test_dataset = AUDataset(test_df, image_dir, transform=val_transform)
    
    # Lưu ý: đã thêm collate_fn=safe_collate_fn để xử lý các giá trị None có thể trả về từ hình ảnh bị lỗi
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, collate_fn=safe_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, collate_fn=safe_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, collate_fn=safe_collate_fn
    )
    
    return train_loader, val_loader, test_loader, num_classes
