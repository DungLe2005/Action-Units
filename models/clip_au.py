import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig

class CLIPActionUnitDetector(nn.Module):
    """
    Bộ phân loại Action Unit sử dụng CLIP làm backbone.
    """
    def __init__(self, 
                 model_name="openai/clip-vit-base-patch32", 
                 num_classes=12,
                 freeze_backbone=False,
                 hidden_dim=512,
                 dropout_rate=0.3):
        super(CLIPActionUnitDetector, self).__init__()
        
        # Tải mô hình CLIP (chúng ta bỏ qua bộ mã hóa văn bản)
        self.backbone = CLIPVisionModel.from_pretrained(model_name)
        
        # Cấu hình tham số requires_grad
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Lấy chiều embedding
        embed_dim = self.backbone.config.hidden_size # Usually 768 for ViT-B/32
        
        # Bộ phân loại
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, pixel_values):
        """
        Input: pixel_values (B, C, H, W)
        Output: logits (B, num_classes)
        """
        # Lấy embedding ảnh (pooled_output)
        outputs = self.backbone(pixel_values=pixel_values)
        
        # pooled_output là các biểu diễn sau khi áp dụng LayerNorm
        # applied on the `[CLS]` token representation.
        image_embeds = outputs.pooler_output 
        
        # Đi qua bộ phân loại
        logits = self.classifier(image_embeds)
        
        return logits
