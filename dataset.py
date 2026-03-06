import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

class AUDataset(Dataset):
    """Action Unit (AU) Dataset."""

    def __init__(self, df, image_dir, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame with image file names and AU labels.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
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

        img_name = os.path.join(self.image_dir, str(self.image_names[idx]))
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a dummy image and zeroes for labels safely if broken, or could raise
            image = Image.new('RGB', (224, 224))
            
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

def get_transforms(image_size=224, is_train=True):
    """
    Get torchvision transforms for CLIP.
    CLIP uses: Mean: [0.48145466, 0.4578275, 0.40821073], Std: [0.26862954, 0.26130258, 0.27577711]
    """
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std)
        ])

def create_dataloaders(params):
    """
    Creates train, val, and test dataloaders.
    Handles stratified split (if possible, else random split).
    """
    csv_path = params['data']['annotation_file']
    image_dir = params['data']['image_dir']
    batch_size = params['train']['batch_size']
    image_size = params['data']['image_size']
    split_ratios = params['data']['split_ratios']  # e.g. [0.7, 0.15, 0.15]
    
    df = pd.read_csv(csv_path)
    
    # Simple split (Stratified is complex for multi-label, using random split for now
    # but user can substitute iterative stratification if needed)
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, num_classes
