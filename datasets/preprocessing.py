import random
import math



import torchvision.transforms as T
import torch
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    pass


# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

def build_au_train_transforms(cfg):
    """
    Face-safe augmentation pipeline for AU detection.
    Avoids aggressive transforms that distort facial muscle positions.
    """
    return T.Compose([
        T.Resize((cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=5),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.0
        ),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

def build_au_val_transforms(cfg):
    """Simple resize and normalize for validation."""
    return T.Compose([
        T.Resize((cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

class RandomErasing(object):
    # Existing RandomErasing code (kept for compatibility)
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class MediaPipeFaceMeshExtractor:
    def __init__(self):
        import mediapipe as mp
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        self.regions = {
            "left_eye": [33, 133, 157, 158, 159, 160, 161, 246, 163, 144, 145, 153, 154, 155],
            "right_eye": [362, 263, 384, 385, 386, 387, 388, 466, 390, 373, 374, 380, 381, 382],
            "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            "right_eyebrow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
            "nose": [1, 2, 98, 327, 168, 6, 8, 4, 45, 275],
            "upper_lip": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
            "lower_lip": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
            "cheeks": [143, 111, 117, 118, 119, 100, 121, 147, 137, 227, 372, 340, 346, 347, 348, 329, 350, 376],
            "jaw": [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
        }
        
    def __call__(self, img):
        import torch
        import numpy as np
        
        if not isinstance(img, np.ndarray):
            img = np.array(img)
            
        results = self.face_mesh.process(img)
        
        region_tensors = []
        max_len = max([len(idx) for idx in self.regions.values()])
        
        if not results.multi_face_landmarks:
            return torch.zeros((9, max_len, 3))
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        for region_name, indices in self.regions.items():
            region_points = []
            for idx in indices:
                lm = landmarks[idx]
                region_points.append([lm.x, lm.y, lm.z])
                
            while len(region_points) < max_len:
                region_points.append([0.0, 0.0, 0.0])
                
            region_tensors.append(region_points)
            
        return torch.tensor(region_tensors, dtype=torch.float32)
