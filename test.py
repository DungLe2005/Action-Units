import torch
from transformers import CLIPProcessor, CLIPVisionModel
from PIL import Image
import requests

# Load mô hình và processor
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPVisionModel.from_pretrained(model_name)

# Tải ảnh
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Chuẩn bị input
inputs = processor(images=image, return_tensors="pt")

# Lấy output
with torch.no_grad():
    outputs = model(**inputs)
    image_features = outputs.pooler_output  # (1, 768)

print("Image features shape:", image_features.shape)