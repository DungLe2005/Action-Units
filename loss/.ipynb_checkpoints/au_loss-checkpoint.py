import torch
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for AU detection.
    Handels class imbalance by providing pos_weight.
    """
    def __init__(self, pos_weight=None):
        super(WeightedBCELoss, self).__init__()
        # BCEWithLogitsLoss combines Sigmoid and BCE for numerical stability
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        """
        logits: [Batch, 12]
        targets: [Batch, 12]
        """
        # --- Handle Invalid Labels (NaN or -1) ---
        valid_mask = (targets >= 0) & (~torch.isnan(targets))
        
        # Replace invalid target values temporarily to compute BCE without throwing NaN
        # (We will mask them out right after)
        safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
        
        if isinstance(logits, list):
            # If multiple classifiers output (like in CLIP-ReID)
            loss = 0.0
            for l in logits:
                # Use reduction='none' to apply valid_mask correctly
                l_loss = nn.functional.binary_cross_entropy_with_logits(
                    l, safe_targets, pos_weight=self.bce.pos_weight, reduction='none'
                )
                l_loss = (l_loss * valid_mask.float()).mean()
                loss += l_loss
            return loss
        
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, safe_targets, pos_weight=self.bce.pos_weight, reduction='none'
        )
        return (loss * valid_mask.float()).mean()
