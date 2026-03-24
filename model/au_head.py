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
        self.classifier = nn.Linear(in_planes, num_aus, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        return self.classifier(x)
