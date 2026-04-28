"""
Pre-compute MediaPipe landmarks cho toàn bộ dataset một lần.
Chạy script này trước khi training để tăng tốc đáng kể.

Usage: python precompute_landmarks.py
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from datasets.preprocessing import MediaPipeFaceMeshExtractor

ROOT = "Face_DATA"
CSV_PATH = os.path.join(ROOT, "labels.csv")
LANDMARK_DIR = os.path.join(ROOT, "landmarks")

os.makedirs(LANDMARK_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
extractor = MediaPipeFaceMeshExtractor()

print(f"Pre-computing landmarks for {len(df)} images...")
failed = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(ROOT, "Images", row['image_path'])
    # Dùng tên file làm key, thay '/' và '\' thành '_' để lưu file phẳng
    save_name = row['image_path'].replace("/", "_").replace("\\", "_").replace(".jpg", "").replace(".png", "")
    save_path = os.path.join(LANDMARK_DIR, save_name + ".pt")

    if os.path.exists(save_path):
        continue  # Skip nếu đã tính rồi

    try:
        img = np.array(Image.open(img_path).convert("RGB"))
        landmark_tensor = extractor(img)  # shape: [9, 21, 3]
        torch.save(landmark_tensor, save_path)
    except Exception as e:
        print(f"Failed: {img_path} — {e}")
        failed += 1

print(f"Done! {len(df) - failed}/{len(df)} landmarks saved to {LANDMARK_DIR}")
print(f"Failed: {failed}")
