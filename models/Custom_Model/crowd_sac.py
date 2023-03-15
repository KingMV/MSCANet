import torch
import torch.nn as nn
import torch.nn.functional as F


class SAConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 use_deform=False
                 ):
        super(SAConv2d, self).__init__()
        self.use_deform = use_deform
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.switch = torch.nn.Conv2d(self.in_channels, 1,
                                      kernel_size=1,
                                      stride=stride,
                                      bias=True)
        self.pre_context = torch.nn.Conv2d(self.in_channels,
                                           self.in_channels,
                                           kernel_size=1,
                                           bias=True)
        self.post_context = torch.nn.Conv2d(self.out_channels,
                                            self.out_channels,
                                            kernel_size=1,
                                            bias=True)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        avg_x = F.adaptive_avg_pool2d(x, [1, 1])
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x += avg_x

        avg_x = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        print(avg_x.shape)
        switch = self.switch(avg_x)

        return switch

# if __name__ == '__main__':
#     print(1)
