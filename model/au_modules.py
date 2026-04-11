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
        
    def forward(self, visual_tokens, region_tokens, distance_mask=None):
        """
        visual_tokens: [B, num_patches, embed_dim] (Query)
        region_tokens: [B, num_regions, embed_dim] (Key/Value)
        distance_mask: [B, num_patches, num_regions] (Additive logit mask or multiplicative mask)
        """
        if distance_mask is not None:
            # MultiHeadAttention takes mask of shape [B * num_heads, L, S] or [L, S]
            # distance_mask is [B, num_patches, num_regions] -> repeat for heads
            B = visual_tokens.shape[0]
            num_heads = self.cross_attn.num_heads
            mask = distance_mask.unsqueeze(1).repeat(1, num_heads, 1, 1) # [B, H, patches, regions]
            mask = mask.view(B * num_heads, visual_tokens.shape[1], region_tokens.shape[1])
        else:
            mask = None
            
        attended_tokens, _ = self.cross_attn(visual_tokens, region_tokens, region_tokens, attn_mask=mask)
        return attended_tokens + visual_tokens # Residual connection


class PatchTokenAttention(nn.Module):
    """
    PTA: Weighted patch pooling for subtle muscle activations.
    """
    def __init__(self, in_dim=512, num_aus=12):
        super().__init__()
        # Produce a weight map over the visual patches for each AU
        self.au_queries = nn.Parameter(torch.randn(1, num_aus, in_dim))
        self.attention_proj = nn.Linear(in_dim, in_dim)
        
    def forward(self, patch_tokens):
        """
        patch_tokens: [B, num_patches, embed_dim]
        returns: [B, num_aus, embed_dim]
        """
        B = patch_tokens.shape[0]
        q = self.au_queries.expand(B, -1, -1) # [B, 12, 512]
        
        # [B, 12, 512] @ [B, 512, patches] -> [B, 12, patches]
        attn_logits = torch.bmm(q, self.attention_proj(patch_tokens).transpose(1, 2))
        attn_weights = F.softmax(attn_logits / (patch_tokens.shape[-1] ** 0.5), dim=-1)
        
        # [B, 12, patches] @ [B, patches, 512] -> [B, 12, 512]
        au_embeddings = torch.bmm(attn_weights, patch_tokens)
        return au_embeddings


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
        return self.transformer(au_tokens)

