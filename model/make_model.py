import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .au_head import AUHead

from peft import LoraConfig, get_peft_model
from .au_modules import FaceRegionLandmarkProjector, FaceRegionGuidedCrossAttention, PatchTokenAttention, AURelationalModeling
import types


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE
        self.is_au = (cfg.DATASETS.NAMES == 'disfa')
        
        if self.is_au:
            self.classifier = AUHead(self.in_planes, 12)
            self.classifier_proj = AUHead(self.in_planes_proj, 12)
            
            # AU Modules
            self.frlp = FaceRegionLandmarkProjector(num_regions=9, max_len=21, embed_dim=self.in_planes)
            self.frgca = FaceRegionGuidedCrossAttention(embed_dim=self.in_planes, num_heads=8)
            self.pta = PatchTokenAttention(in_dim=self.in_planes, num_aus=12)
            self.au_relational = AURelationalModeling(embed_dim=self.in_planes, num_heads=4)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
            self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        if self.is_au:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["c_fc", "c_proj"],
                lora_dropout=0.1,
                bias="none"
            )
            self.image_encoder = get_peft_model(clip_model.visual, lora_config)
            self.image_encoder.print_trainable_parameters()
        else:
            self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

    def forward(self, x, label=None, cam_label= None, view_label=None, landmarks=None):
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) #B,512  B,128,512
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
                
            if self.is_au and landmarks is not None:
                # Get region tokens
                region_tokens = self.frlp(landmarks)
                
                # We need visual tokens. PEFT wraps the visual model, we can call it.
                # However, visual outputs x11, x12, xproj.
                # We want multi-level features, so we can monkey patch or just use x11 and x12.
                x11, x12, xproj = self.image_encoder(x, cv_emb=cv_embed)
                
                # Combine Multi-Level Features (concat)
                # x11 block outputs: [B, 197, 768], x12 block outputs: [B, 197, 768]
                # Actually, make_model visual returns CLS tokens normally?
                # Let's check VisionTransformer.forward: returns x11, x12, xproj -> these are full sequence?
                # Wait, VisionTransformer in clip.py returns x11, x12, xproj as full sequence! ([B, 197, 768])
                # No, they are LND in clip.py! They get permuted?
                # Ah, x11 = x11.permute(1, 0, 2) in clip.py! So they are NLD = [B, seq, dim].
                
                # We use x11 as visual patches (skip cls token 0)
                visual_patches = x11[:, 1:, :] # [B, 196, 768]
                
                # Apply Face-Region Guided Cross-Attention
                attended_patches = self.frgca(visual_patches, region_tokens)
                
                # Patch Token Attention (PTA) yielding AU-specific embeddings
                au_embeddings = self.pta(attended_patches) # [B, 12, 768]
                
                # RandReAU Training Strategy
                if self.training:
                    shuffle_idx = torch.randperm(12)
                    unshuffle_idx = torch.argsort(shuffle_idx)
                    au_embeddings = au_embeddings[:, shuffle_idx, :]
                    
                # Relational Modeling
                au_embeddings = self.au_relational(au_embeddings)
                
                if self.training:
                    au_embeddings = au_embeddings[:, unshuffle_idx, :]
                
                # We flatten or use average for classification?
                # Since PTA yields 12 tokens, maybe classifier should operate on each?
                # We can just average them and pass to self.classifier!
                # Wait, AUHead typically takes [B, dim] and outputs [B, 12].
                # If we have 12 specific tokens, we can output probabilities directly by a Linear(dim, 1) per AU?
                # Let's just pool them to a single vector for compatibility with AUHead.
                img_feature = au_embeddings.mean(dim=1)
                img_feature_last = x11[:, 0]
                img_feature_proj = xproj[:, 0] if xproj is not None else img_feature
                
            else:
                image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) #B,512  B,128,512
                img_feature_last = image_features_last[:,0]
                img_feature = image_features[:,0]
                img_feature_proj = image_features_proj[:,0]

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 


        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj]

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                if self.is_au:
                    return torch.sigmoid(self.classifier(feat)) # Return AU probabilities
                return torch.cat([feat, feat_proj], dim=1)
            else:
                if self.is_au:
                    # Average of last and current features maybe, but standard is just one
                    return torch.sigmoid(self.classifier(self.bottleneck(img_feature)))
                return torch.cat([img_feature, img_feature_proj], dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model