from .vgg_block import vgg13_model
import torch.nn as nn
from .conv_utils import Conv2d
import torch.nn.functional as F


class VGG13_Crowd(nn.Module):
    def __init__(self, pretrained=False, bn=True):
        super(VGG13_Crowd, self).__init__()
        self.feature_c4 = vgg13_model(pretrained)
        self.de_pred = nn.Sequential(Conv2d(512, 128, 1, same_padding=True, NL='relu', bn=bn),
                                     Conv2d(128, 1, 1, same_padding=True, NL='None'))

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.feature_c4(x)
        x = self.de_pred(x[-1])
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x
