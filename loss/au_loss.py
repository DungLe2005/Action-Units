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
        if isinstance(logits, list):
            # If multiple classifiers output (like in CLIP-ReID)
            loss = sum([self.bce(l, targets) for l in logits])
            return loss
        return self.bce(logits, targets)
