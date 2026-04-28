import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from .bases import read_image
import warnings

class DISFA(Dataset):
    """
    DISFA Dataset for Facial Action Unit Detection.
    Expects labels.csv created by prepare_data.py
    """
    def __init__(self, root="AUs_DATA", transform=None, precomputed_landmarks=True):
        self.root = root
        self.transform = transform
        self.csv_path = os.path.join(self.root, "labels.csv")
        self.landmark_dir = os.path.join(self.root, "landmarks")
        self.precomputed_landmarks = precomputed_landmarks and os.path.isdir(self.landmark_dir)

        if not self.precomputed_landmarks:
            warnings.warn(
                f"Landmark cache not found at '{self.landmark_dir}'. "
                "MediaPipe will run per-sample (SLOW). "
                "Run 'python precompute_landmarks.py' first to speed up training."
            )
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"labels.csv not found at {self.csv_path}. Please run prepare_data.py first.")
            
        self.df = pd.read_csv(self.csv_path)
        self.au_cols = [c for c in self.df.columns if c.startswith('AU')]
        
        # Prepare data list for consistency with BaseImageDataset if needed
        # Format: (img_path, au_vector, dummy_cam, dummy_view, landmark_key)
        self.train = []
        for _, row in self.df.iterrows():
            img_path = os.path.join(self.root, "Images", row['image_path'])
            au_label = row[self.au_cols].values.astype(float)
            # Landmark cache key matches precompute_landmarks.py naming convention
            lm_key = row['image_path'].replace("/", "_").replace("\\", "_")
            lm_key = os.path.splitext(lm_key)[0] + ".pt"
            self.train.append((img_path, au_label, 0, 0, lm_key))
            
        self.num_train_pids = 12  # With AU, "classes" are AU dimensions
        self.num_train_cams = 1
        self.num_train_vids = 1

    def _load_landmark(self, lm_key):
        """Load pre-computed landmark tensor from disk."""
        lm_path = os.path.join(self.landmark_dir, lm_key)
        if os.path.exists(lm_path):
            return torch.load(lm_path, weights_only=True)
        # Fallback: return zeros if cache file missing
        return torch.zeros((9, 21, 3))
        
    def __len__(self):
        return len(self.train)

    def __getitem__(self, index):
        img_path, au_label, camid, viewid, lm_key = self.train[index]
        img = read_image(img_path)

        if self.precomputed_landmarks:
            # Fast path: load from disk cache (~0.1ms)
            landmarks = self._load_landmark(lm_key)
        else:
            # Slow path: run MediaPipe on-the-fly (~30-100ms per sample)
            if not hasattr(self, 'mp_extractor'):
                from .preprocessing import MediaPipeFaceMeshExtractor
                self.mp_extractor = MediaPipeFaceMeshExtractor()
            landmarks = self.mp_extractor(img)
            
        # Sanitize landmark tensor to remove NaN and Inf values (replace with 0.0)
        if torch.isnan(landmarks).any() or torch.isinf(landmarks).any():
            landmarks = torch.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0)

        if self.transform is not None:
            img = self.transform(img)

        # Convert label to float tensor for BCE loss
        au_label = torch.tensor(au_label, dtype=torch.float32)
        
        return img, au_label, camid, viewid, os.path.basename(img_path), landmarks
