import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceRegionLandmarkProjector(nn.Module):
    """
    FRLP: Convert grouped landmark regions into embedding tokens.
    Input: [B, 9, max_len, 3] from MediaPipeFaceMeshExtractor
    Output: [B, 9, embed_dim] (region tokens)
    """
    def __init__(self, num_regions=9, max_len=21, embed_dim=512):
        super().__init__()
        # Flatten each region (max_len * 3) and project
        self.proj = nn.Linear(max_len * 3, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, R, N, C = x.shape
        x_flat = x.view(B, R, N * C) # [B, 9, max_len*3]
        tokens = self.proj(x_flat) # [B, 9, embed_dim]
        return self.layer_norm(tokens)
        

class FaceRegionGuidedCrossAttention(nn.Module):
    """
    FRGCA: Region-guided cross-attention layer.
    Focus visual patch tokens based on physical distances to region landmarks.
    """
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # We need a fallback if distance masking is tricky, but let's implement standard attn first,
        # then inject distance in attn_weights later or use it as a prior.
        # Since MHA doesn't allow direct arbitrary masking of attn_weights except via attn_mask (add to logits),
        # we can compute a distance mask and pass it as attn_mask.
        
    def forward(self, visual_tokens, region_tokens, distance_mask=None, return_attn=False):
        """
        visual_tokens: [B, num_patches, embed_dim] (Query)
        region_tokens: [B, num_regions, embed_dim] (Key/Value)
        distance_mask: [B, num_patches, num_regions] (Additive logit mask or multiplicative mask)
        """
        if distance_mask is not None:
            # MultiHeadAttention takes mask of shape [B * num_heads, L, S]
            # distance_mask is [B, num_patches, num_regions] (1 for allow, 0 for block)
            # convert to additive bias:
            B = visual_tokens.shape[0]
            num_heads = self.cross_attn.num_heads
            
            mask_bias = (1.0 - distance_mask.to(torch.float32)) * -1e9
            mask = mask_bias.unsqueeze(1).repeat(1, num_heads, 1, 1) # [B, H, patches, regions]
            mask = mask.view(B * num_heads, visual_tokens.shape[1], region_tokens.shape[1])
        else:
            mask = None
            
        with torch.amp.autocast('cuda', enabled=False):
            # Upcast to FP32 to prevent attention logits from overflowing FP16 
            v_f32 = visual_tokens.to(torch.float32)
            r_f32 = region_tokens.to(torch.float32)
            mask_f32 = mask.to(torch.float32) if mask is not None else None
            
            attended_tokens, attn_weights = self.cross_attn(v_f32, r_f32, r_f32, attn_mask=mask_f32)
            
        # Downcast back to original dtype
        attended_tokens = attended_tokens.to(visual_tokens.dtype)
        out = attended_tokens + visual_tokens
        
        if return_attn:
            return out, attn_weights
        return out


class PatchTokenAttention(nn.Module):
    """
    PTA: Weighted patch pooling for subtle muscle activations.
    """
    def __init__(self, in_dim=512, num_aus=12):
        super().__init__()
        # Produce a weight map over the visual patches for each AU
        self.au_queries = nn.Parameter(torch.randn(1, num_aus, in_dim))
        self.attention_proj = nn.Linear(in_dim, in_dim)
        
    def forward(self, patch_tokens, return_attn=False, attn_mask=None, temperature=1.0):
        """
        patch_tokens: [B, num_patches, embed_dim]
        attn_mask: [B, num_aus, num_patches] (1 for allow, 0 for mask)
        returns: [B, num_aus, embed_dim]
        """
        B = patch_tokens.shape[0]
        q = self.au_queries.expand(B, -1, -1) # [B, 12, 512]
        
        with torch.amp.autocast('cuda', enabled=False):
            # Upcast arrays to fp32 before bmm to prevent FP16 overflow (inf)
            pt_f32 = patch_tokens.to(torch.float32)
            q_f32 = q.to(torch.float32)
            attn_proj_weight = self.attention_proj.weight.to(torch.float32)
            attn_proj_bias = self.attention_proj.bias.to(torch.float32) if self.attention_proj.bias is not None else None
            
            proj_pt_f32 = F.linear(pt_f32, attn_proj_weight, attn_proj_bias)
            
            # [B, 12, 512] @ [B, 512, patches] -> [B, 12, patches]
            attn_logits = torch.bmm(q_f32, proj_pt_f32.transpose(1, 2))
            
            if attn_mask is not None:
                # attn_mask [B, 12, 196]. Use large negative value for 0 positions
                m_f32 = attn_mask.to(torch.float32)
                attn_logits = attn_logits + (1.0 - m_f32) * -1e9
                
            # Apply temperature scaling before softmax
            # Divide by sqrt(dim) and temperature
            scale = (pt_f32.shape[-1] ** 0.5) * temperature
            attn_weights = F.softmax(attn_logits / scale, dim=-1)
            
            # [B, 12, patches] @ [B, patches, 512] -> [B, 12, 512]
            au_embeddings = torch.bmm(attn_weights, pt_f32)
            
        res = au_embeddings.to(patch_tokens.dtype)
        if return_attn:
            return res, attn_weights
        return res


class AURelationalModeling(nn.Module):
    """
    Capture AU co-occurrence relationships using a Transformer encoder layer.
    """
    def __init__(self, embed_dim=512, num_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
    def forward(self, au_tokens):
        """
        au_tokens: [B, num_aus, embed_dim]
        returns: Contextualized [B, num_aus, embed_dim]
        """
        with torch.amp.autocast('cuda', enabled=False):
            au_f32 = au_tokens.to(torch.float32)
            out_f32 = self.transformer(au_f32)
            
        return out_f32.to(au_tokens.dtype)

