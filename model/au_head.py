import torch
import torch.nn as nn
from torch.nn import init

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

class AUHead(nn.Module):
    def __init__(self, in_planes, num_aus=12):
        super(AUHead, self).__init__()
        self.classifier = nn.Linear(in_planes, num_aus, bias=True)
        self.classifier.apply(weights_init_classifier)

    def init_bias_from_pos_weight(self, pos_weight):
        if self.classifier.bias is None:
            return

        with torch.no_grad():
            values = torch.as_tensor(
                pos_weight,
                dtype=self.classifier.bias.dtype,
                device=self.classifier.bias.device,
            )
            if values.numel() != self.classifier.bias.numel():
                raise ValueError(
                    "AU prior bias expected {} values, got {}".format(
                        self.classifier.bias.numel(), values.numel()
                    )
                )
            values = values.reshape_as(self.classifier.bias).clamp_min(1e-6)
            self.classifier.bias.copy_(-torch.log(values))

    def forward(self, x):
        return self.classifier(x)
