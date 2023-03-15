import torch.nn as nn
import torch
import torch.nn.functional as F


# from models.Custom_Model.DCNV2.modules import DeformConvPack, DeformConv, ModulatedDeformConvPack


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False,
                 dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = []
        if dilation == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation,
                                  bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation,
                                  bias=False)
        # initialize the weight of Convolution layer
        self.conv.weight.data.normal_(0, 0.01)
        # self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        # initialize the parameter of bn layer
        if self.bn is not None:
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if NL == 'relu':
            self.relu = nn.ReLU()
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        elif NL == 'sigmoid':
            self.relu = nn.Sigmoid()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# class DCNV2(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False,
#                  dilation=1):
#         super(DCNV2, self).__init__()
#         padding = int((kernel_size - 1) / 2) if same_padding else 0
#         self.conv = []
#         if dilation == 1:
#             self.conv = ModulatedDeformConvPack(in_channels, out_channels, kernel_size, stride, padding=padding,
#                                                 dilation=dilation)
#         else:
#             self.conv = ModulatedDeformConvPack(in_channels, out_channels, kernel_size, stride, padding=dilation,
#                                                 dilation=dilation)
#         # initialize the weight of Convolution layer
#         self.conv.weight.data.normal_(0, 0.01)
#         self.conv.bias.data.zero_()
#         self.bn = nn.BatchNorm2d(out_channels) if bn else None
#         # initialize the parameter of bn layer
#         if self.bn is not None:
#             self.bn.weight.data.fill_(1)
#             self.bn.bias.data.zero_()
#         if NL == 'relu':
#             self.relu = nn.ReLU()
#         elif NL == 'prelu':
#             self.relu = nn.PReLU()
#         elif NL == 'sigmoid':
#             self.relu = nn.Sigmoid()
#         else:
#             self.relu = None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x


class MSA_Block_01(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, bilinear=False, dr=1):
        super(MSA_Block_01, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn = bn
        self.bilinear = bilinear

        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv1 = Conv2d(self.in_channels, self.in_channels // 2, 1, 1, NL='relu', same_padding=True, bn=self.bn)
            self.conv2 = Conv2d(self.in_channels // 2, self.out_channels, 3, 1, NL='relu', same_padding=True,
                                bn=self.bn)
        else:
            self.up = nn.ConvTranspose2d(self.in_channels, self.in_channels // 2, kernel_size=2, stride=2)
            self.conv1 = Conv2d(self.in_channels, self.in_channels // 2, 1, 1, NL='relu', dilation=1, bn=self.bn,
                                same_padding=True)

            # self.conv2 = Conv2d(self.in_channels // 2, self.out_channels, 3, 1, NL='relu', dilation=dr, bn=self.bn)
            self.conv2_1 = Conv2d(self.in_channels // 2, self.out_channels, 1, 1, NL='relu', dilation=1, bn=self.bn)
            self.conv2_2 = Conv2d(self.in_channels // 2, self.out_channels, 3, 1, NL='relu', dilation=dr, bn=self.bn)

    def forward(self, x1, x2):
        # X1 is upsampling  featuremap and X2 is org_feature_map
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)
        # print(x.size())
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x = x1 + x2
        return x


class Up_MLFN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up_MLFN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = Conv2d(in_channel, out_channel, 1, 1, NL='relu', same_padding=True, bn=True)
        self.conv2 = Conv2d(out_channel, out_channel, 3, 1, NL='relu', same_padding=True, bn=True)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x = x1 + x2
        return x


class Up_Block(nn.Module):
    def __init__(self, in_channel, out_channel, residual=False):
        super(Up_Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = Conv2d(in_channel, in_channel // 2, 1, 1, NL='relu', same_padding=True, bn=True)
        self.conv2 = Conv2d(in_channel // 2, out_channel, 3, 1, NL='relu', same_padding=True, bn=True)
        if residual:
            self.downsample = Conv2d(in_channel, out_channel, 1, 1, NL='relu', same_padding=True, bn=True)
            self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)

        if self.residual and self.in_channel != self.out_channel:
            identity = self.downsample(identity)
        if self.residual:
            x += identity
            x = self.relu(x)
            return x
        return x


class RF_Module(nn.Module):
    def __init__(self, input_channel, channels, dilated_rate=1, bn=True, mode=1):
        super(RF_Module, self).__init__()
        self.mode = mode
        self.bn = bn
        if self.mode == 1:
            self.rf_block = nn.Sequential(
                Conv2d(input_channel, channels[0], 3, 1, 'relu', same_padding=True, bn=self.bn, dilation=dilated_rate),
                Conv2d(channels[0], channels[1], 3, 1, 'relu', same_padding=True, bn=self.bn, dilation=dilated_rate),
                Conv2d(channels[1], channels[2], 3, 1, 'relu', same_padding=True, bn=self.bn, dilation=dilated_rate)
            )

    def forward(self, input):
        input = self.rf_block(input)
        return input


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, add_max=False):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.add_max = add_max
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if self.add_max:
            self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(self.input_channels, self.input_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.input_channels // reduction_ratio, self.input_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        if self.add_max:
            max_out = self.fc(self.max_pool(x))
            out = avg_out + max_out
        else:
            out = avg_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, input_planes, kernel_size=3, add_max=False):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        self.add_max = add_max
        padding = 3 if kernel_size == 7 else 1
        if self.add_max:
            self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        else:
            self.conv1 = nn.Conv2d(input_planes, 1, 1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        if self.add_max:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            x = self.conv1(x)
        else:
            x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, input_planes, reduction_ratio=16, res_skip=False):
        super(CBAM, self).__init__()
        self.res_skip = res_skip
        self.ca = ChannelAttention(input_planes, reduction_ratio, add_max=False)
        self.sa = SpatialAttention(input_planes=input_planes, kernel_size=3, add_max=False)

    def forward(self, x):
        residual = x
        x = self.ca(x) * x
        if self.res_skip:
            x += residual
        x = self.sa(x) * x
        if self.res_skip:
            x += residual
        return x


class SAM(nn.Module):
    def __init__(self, input_planes):
        super(SAM, self).__init__()
        self.input_planes = input_planes

    def forward(self, x, seg):
        mask_size = x.size()[2:]
        seg = F.interpolate(seg, size=mask_size, mode='bilinear', align_corners=True)
        x = x * seg

        return x


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
