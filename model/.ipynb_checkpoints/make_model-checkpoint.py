import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .au_head import AUHead

from peft import LoraConfig, get_peft_model
from .au_modules import FaceRegionLandmarkProjector, FaceRegionGuidedCrossAttention, PatchTokenAttention, AURelationalModeling
import types

# Anatomical Mapping: AU_idx -> List of Region_idx
# Regions: 0:L_Eye, 1:R_Eye, 2:L_Eyebrow, 3:R_Eyebrow, 4:Nose, 5:U_Lip, 6:L_Lip, 7:Cheeks, 8:Jaw
AU_REGION_MAP = {
    0: [2, 3],       # AU1
    1: [2, 3],       # AU2
    2: [2, 3, 4],    # AU4
    3: [0, 1],       # AU5
    4: [0, 1, 7],    # AU6
    5: [4],          # AU9
    6: [5, 6, 7],    # AU12
    7: [5, 6],       # AU15
    8: [6, 8],       # AU17
    9: [5, 6],       # AU20
    10: [5, 6],      # AU25
    11: [5, 6, 8]    # AU26
}

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
            self.visual_ln = nn.LayerNorm(self.in_planes) # Add LayerNorm for raw visual patches
            self.frgca = FaceRegionGuidedCrossAttention(embed_dim=self.in_planes, num_heads=8)
            self.pta = PatchTokenAttention(in_dim=self.in_planes, num_aus=12)
            self.au_relational = AURelationalModeling(embed_dim=self.in_planes, num_heads=4)
            
            # Anatomical Prior Mask [12, 9]
            self.register_buffer('au_region_prior', torch.zeros(12, 9))
            for au_idx, regions in AU_REGION_MAP.items():
                for r_idx in regions:
                    self.au_region_prior[au_idx, r_idx] = 1.0
            
            self.current_stage = 1
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

    def _get_anatomical_assignment(self, landmarks, mode='hard', temp=0.1):
        """
        landmarks: [B, 9, 21, 3] trong [0, 1]
        Returns: [B, 196, 9] (Patch-to-Region assignment weights)
        """
        B = landmarks.shape[0]
        device = landmarks.device
        lm_xy = landmarks[:, :, :, :2]
        
        # Patch centers (14x14 grid)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.5/14, 13.5/14, 14, device=device),
            torch.linspace(0.5/14, 13.5/14, 14, device=device),
            indexing='ij'
        )
        patch_xy = torch.stack([grid_x, grid_y], dim=-1).view(1, 196, 2)
        
        # Distance between patch centers and region landmarks [B, 196, 9, 21]
        dist = torch.cdist(patch_xy, lm_xy.view(B, 9*21, 2)).view(B, 196, 9, 21)
        min_dist_to_region, _ = dist.min(dim=-1) # [B, 196, 9]
        
        if mode == 'hard':
            # Voronoi: mỗi patch thuộc về vùng gần nhất
            nearest_region_idx = min_dist_to_region.argmin(dim=-1)
            mapping = F.one_hot(nearest_region_idx, num_classes=9).float()
        else:
            # Soft: phân bổ trọng số dựa trên âm khoảng cách
            mapping = F.softmax(-min_dist_to_region / temp, dim=-1)
            
        # Lọc bỏ các patch quá xa bất kỳ vùng nào (background)
        background_mask = (min_dist_to_region.min(dim=-1).values < 0.25).float().unsqueeze(-1)
        return mapping * background_mask

    def forward(self, x, label=None, cam_label= None, view_label=None, landmarks=None, return_attn=False, use_relation=True):
        # Reset attention penalty at each pass to avoid stale values
        if hasattr(self, 'attn_penalty'):
            del self.attn_penalty
            
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
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
                # Stage 1: Hard anatomical masking (Voronoi)
                # assignment_hard is always needed now to anchor region tokens
                assignment_hard = self._get_anatomical_assignment(landmarks, mode='hard')
                frgca_mask = assignment_hard
                
                # pta_mask: Map from regions to patches based on prior [B, 12, 196]
                # Enforce in BOTH stages to guarantee anatomical focus
                pta_mask = torch.matmul(assignment_hard, self.au_region_prior.T).transpose(1, 2)
                pta_mask = (pta_mask > 0).float()
                
                region_tokens = self.frlp(landmarks)
                x11, x12, xproj = self.image_encoder(x, cv_emb=cv_embed)
                visual_patches = self.visual_ln(x11[:, 1:, :])

                # FRGCA - Cross-attention between patches and region tokens
                # Anchored with frgca_mask in both stages to keep regions anatomically pure
                if return_attn or self.current_stage == 2:
                     attended_patches, frgca_weights = self.frgca(visual_patches, region_tokens, distance_mask=frgca_mask, return_attn=True)
                else:
                     attended_patches = self.frgca(visual_patches, region_tokens, distance_mask=frgca_mask)
                     
                if torch.isnan(attended_patches).any() or torch.isinf(attended_patches).any(): raise ValueError("Inf/NaN detected after FaceRegionGuidedCrossAttention.")
                
                # PTA - AU queries attending to visual patches
                # Use a low temperature (0.1) in Stage 2 and hard mask to prevent uniformity/collapse
                pta_temp = 0.1 if self.current_stage == 2 else 0.5
                
                if return_attn or self.current_stage == 2:
                    au_embeddings, pta_weights = self.pta(attended_patches, return_attn=True, attn_mask=pta_mask, temperature=pta_temp)
                    
                    if self.current_stage == 2:
                        # Stage 2: Stricter Soft Penalty Calculation (Lower temp = more focused violation detect)
                        assignment_soft = self._get_anatomical_assignment(landmarks, mode='soft', temp=0.05)
                        # au_region_focus_smooth: [B, 12, 196] @ [B, 196, 9] -> [B, 12, 9]
                        au_region_focus_smooth = torch.bmm(pta_weights, assignment_soft)
                        # Illegal = 1 - Prior
                        illegal_mask = 1.0 - self.au_region_prior # [12, 9]
                        focus_violation = (au_region_focus_smooth * illegal_mask.unsqueeze(0)).sum() / x.shape[0]
                        self.attn_penalty = focus_violation
                        # For visualization, compute standard focus as well
                        au_region_focus = torch.bmm(pta_weights, frgca_weights)
                    else:
                        au_region_focus = torch.bmm(pta_weights, frgca_weights)
                else:
                    au_embeddings, pta_weights = self.pta(attended_patches, return_attn=True, attn_mask=pta_mask, temperature=pta_temp)
                    # Debug PTA Mask
                    active_patches = pta_mask.sum(dim=-1).mean().item()
                    print(f"[DEBUG] PTA Mask Active Patches (avg per AU): {active_patches:.1f} / 196")
                    if active_patches < 1.0:
                        print("[DEBUG] WARNING: PTA Mask is nearly empty! Check landmarks.")
                
                # Rest of the flow
                if self.training:
                    shuffle_idx = torch.randperm(12)
                    unshuffle_idx = torch.argsort(shuffle_idx)
                    au_embeddings = au_embeddings[:, shuffle_idx, :]
                
                if use_relation:
                    au_embeddings = self.au_relational(au_embeddings)
                
                if self.training:
                    au_embeddings = au_embeddings[:, unshuffle_idx, :]
                
                img_feature = au_embeddings.mean(dim=1)
                print(f"[DEBUG] img_feature (pooled) stats: mean={img_feature.mean().item():.4f}, std={img_feature.std().item():.4f}, max={img_feature.max().item():.4f}")
                img_feature_last = x11[:, 0]
                img_feature_proj = xproj[:, 0] if xproj is not None else img_feature
            else:
                image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
                img_feature_last = image_features_last[:,0]
                img_feature = image_features[:,0]
                img_feature_proj = image_features_proj[:,0]

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 

        if self.training:
            cls_score = self.classifier(feat)
            if self.is_au:
                out_scores = [cls_score.float().contiguous()]
                out_feats = [
                    img_feature_last.float().contiguous(),
                    img_feature.float().contiguous(),
                    img_feature_proj.float().contiguous()
                ]
                # Pass attention penalty as a separate feature if Stage 2
                if hasattr(self, 'attn_penalty'):
                    out_feats.append(self.attn_penalty)
                
                if return_attn:
                    return out_scores, out_feats, au_region_focus if 'au_region_focus' in locals() else None
                return out_scores, out_feats
            else:
                cls_score_proj = self.classifier_proj(feat_proj)
                out_scores = [cls_score.float().contiguous(), cls_score_proj.float().contiguous()]
                out_feats = [img_feature_last.float().contiguous(), img_feature.float().contiguous(), img_feature_proj.float().contiguous()]
                return out_scores, out_feats
        else:
            if self.neck_feat == 'after':
                if self.is_au:
                    probs = torch.sigmoid(self.classifier(feat))
                    if return_attn:
                        return probs, au_region_focus if 'au_region_focus' in locals() else None
                    return probs
                return torch.cat([feat, feat_proj], dim=1)
            else:
                probs = torch.sigmoid(self.classifier(self.bottleneck(img_feature)))
                if return_attn:
                     return probs, au_region_focus if 'au_region_focus' in locals() else None
                return probs

    def set_train_stage(self, stage):
        self.current_stage = stage
        if self.is_au:
            if stage == 1:
                print(">>> [Stage 1] Đóng băng CLIP Backbone. Chỉ huấn luyện AU Modules.")
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
                modules_to_train = [self.frlp, self.visual_ln, self.frgca, self.pta, self.au_relational, self.classifier, self.bottleneck]
                for module in modules_to_train:
                    for param in module.parameters():
                        param.requires_grad = True
            elif stage == 2:
                print(">>> [Stage 2] Mở khóa LoRA adapters. Bắt đầu tinh chỉnh toàn mạng.")
                for name, param in self.image_encoder.named_parameters():
                    param.requires_grad = True if "lora_" in name else False
                modules_to_train = [self.frlp, self.visual_ln, self.frgca, self.pta, self.au_relational, self.classifier, self.bottleneck]
                for module in modules_to_train:
                    for param in module.parameters():
                        param.requires_grad = True

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
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model
