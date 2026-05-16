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
        targets = targets.float()
        if isinstance(logits, list):
            # Keep dual-head training on the same loss scale as a single AU head.
            losses = [self.bce(logit.float(), targets) for logit in logits]
            return torch.stack(losses).mean()
        return self.bce(logits.float(), targets)
