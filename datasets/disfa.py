import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .bases import read_image


def subject_from_image_path(image_path):
    return str(image_path).replace("\\", "/").split("/")[0]


class DISFA(Dataset):
    """
    DISFA Dataset for Facial Action Unit Detection.
    Expects labels.csv created by prepare_data.py
    """
    def __init__(self, root="AUs_DATA", transform=None, subjects=None):
        self.root = root
        self.transform = transform
        self.csv_path = os.path.join(self.root, "labels.csv")
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"labels.csv not found at {self.csv_path}. Please run prepare_data.py first.")
            
        self.df = pd.read_csv(self.csv_path)
        self.au_cols = [c for c in self.df.columns if c.startswith('AU')]
        self.df["subject_id"] = self.df["image_path"].map(subject_from_image_path)
        self.all_subjects = sorted(self.df["subject_id"].unique().tolist())

        if subjects is not None:
            subject_set = {str(subject) for subject in subjects}
            self.df = self.df[self.df["subject_id"].isin(subject_set)].reset_index(drop=True)
            if self.df.empty:
                raise ValueError(f"No DISFA samples found for subjects: {sorted(subject_set)}")

        self.subjects = sorted(self.df["subject_id"].unique().tolist())
        
        # Prepare data list for consistency with BaseImageDataset if needed
        # Format: (img_path, au_vector, dummy_cam, dummy_view)
        self.train = []
        for _, row in self.df.iterrows():
            img_path = os.path.join(self.root, "Images", row['image_path'])
            au_label = row[self.au_cols].values.astype(float)
            # We use 0 for cam and view as dummy values to keep compatibility with CLIP-ReID pipeline
            self.train.append((img_path, au_label, 0, 0))
            
        self.num_train_pids = 12 # With AU, "classes" are AU dimensions
        self.num_train_cams = 1
        self.num_train_vids = 1
        
    def __len__(self):
        return len(self.train)

    def __getitem__(self, index):
        img_path, au_label, camid, viewid = self.train[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        # Convert label to float tensor for BCE loss
        au_label = torch.tensor(au_label, dtype=torch.float32)
        
        return img, au_label, camid, viewid, os.path.basename(img_path)
