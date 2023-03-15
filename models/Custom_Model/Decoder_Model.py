import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from .conv_utils import BaseConv, Conv2d, Up_Block, make_layers, MSA_Block_01, Up_MLFN
from .MLFPN import CFFM

from .sknet import SKUnit


class Decoder01(nn.Module):
    def __init__(self, filters, res=False):
        '''
        :param filters:  [32,64,128,256,256]
        '''
        super(Decoder01, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_4_1 = Up_Block(filters[3] + filters[4], filters[2], residual=res)
        self.conv_3_2 = Up_Block(filters[2] + filters[2], filters[1], residual=res)
        self.conv_2_3 = Up_Block(filters[1] + filters[1], filters[0], residual=res)
        # self.conv4_reduce_dim = Conv2d(512, filters[-2], 1, 1, NL='relu', same_padding=True, bn=True)
        self.conv4_reduce_dim = nn.Conv2d(512, filters[-2], 1, 1, padding=0)
        # self.conv3_reduce_dim = Conv2d(256, filters[-3], 1, 1, NL='relu', same_padding=True, bn=True)
        self.conv3_reduce_dim = nn.Conv2d(256, filters[-2], 1, 1, padding=0)
        # self.conv2_reduce_dim = Conv2d(128, filters[-4], 1, 1, NL='relu', same_padding=True, bn=True)
        self.conv2_reduce_dim = nn.Conv2d(128, filters[-2], 1, 1, padding=0)
        self.H = nn.Sequential(Conv2d(filters[0], filters[0], 3, 1, NL='relu', same_padding=True, bn=True),
                               Conv2d(filters[0], filters[0], 3, 1, NL='relu', same_padding=True, bn=True)
                               )

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_x = input
        x4_1 = self.conv_4_1(torch.cat([self.conv4_reduce_dim(conv4_3), conv5_x], 1))
        x3_2 = self.conv_3_2(torch.cat([self.conv3_reduce_dim(conv3_3), self.up(x4_1)], 1))
        x2_3 = self.conv_2_3(torch.cat([self.conv2_reduce_dim(conv2_2), self.up(x3_2)], 1))
        output = self.H(x2_3)
        return output


class Decoder02(nn.Module):
    def __init__(self, filters, res=False):
        '''
        :param filters:  [32,64,128,256,256]
        '''
        super(Decoder02, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_4_1 = Up_Block(filters[3] + filters[4], filters[3], residual=res)
        self.conv_3_2 = Up_Block(filters[2] + filters[3], filters[2], residual=res)
        self.conv_2_3 = Up_Block(filters[1] + filters[2], filters[1], residual=res)
        self.conv4_reduce_dim = Conv2d(512, filters[-2], 1, 1, NL='relu', same_padding=True, bn=True)
        # self.conv4_reduce_dim = nn.Conv2d(512, filters[-2], 1, 1, padding=0)
        self.conv3_reduce_dim = Conv2d(256, filters[-3], 1, 1, NL='relu', same_padding=True, bn=True)
        # self.conv3_reduce_dim = nn.Conv2d(256, filters[-3], 1, 1, padding=0)
        self.conv2_reduce_dim = Conv2d(128, filters[-4], 1, 1, NL='relu', same_padding=True, bn=True)
        # self.conv2_reduce_dim = nn.Conv2d(128, filters[-4], 1, 1, padding=0)
        self.H = nn.Sequential(Conv2d(filters[1], filters[0], 1, 1, NL='relu', same_padding=True, bn=True),
                               Conv2d(filters[0], filters[0], 3, 1, NL='relu', same_padding=True, bn=True),
                               Conv2d(filters[0], filters[0], 3, 1, NL='relu', same_padding=True, bn=True)
                               )

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_x = input
        x4_1 = self.conv_4_1(torch.cat([self.conv4_reduce_dim(conv4_3), conv5_x], 1))
        x3_2 = self.conv_3_2(torch.cat([self.conv3_reduce_dim(conv3_3), self.up(x4_1)], 1))
        x2_3 = self.conv_2_3(torch.cat([self.conv2_reduce_dim(conv2_2), self.up(x3_2)], 1))
        output = self.H(x2_3)
        return output


class Decoder_MSB_UNet(nn.Module):
    def __init__(self, filters, bn=True):
        '''
        :param filters:  [128,256,512]
        '''
        super(Decoder_MSB_UNet, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.M3_5 = MSA_Block_01(filters[1] + filters[2], filters[1], bn=bn)
        self.M2_3 = MSA_Block_01(filters[0] + filters[1], filters[0], bn=bn)

    def forward(self, *input):
        conv2_2, conv3_3, conv5_x = input
        # print(conv5_x.size())
        # print(conv3_3.size())

        x3 = self.M3_5(conv3_3, self.up(conv5_x))
        x2 = self.M2_3(conv2_2, self.up(x3))
        return x2, x3


class Decoder_UNet(nn.Module):
    def __init__(self, filters, bn=True):
        '''
        :param filters:  [128,256,512]
        '''
        super(Decoder_UNet, self).__init__()
        self.M3_5 = MSA_Block_01(filters[2], filters[1], bn=bn, dr=2, bilinear=False)
        self.M2_3 = MSA_Block_01(filters[1], filters[0], bn=bn, dr=2, bilinear=False)

    def forward(self, *input):
        conv2_2, conv3_3, conv5_x = input
        x3 = self.M3_5(conv5_x, conv3_3)
        x2 = self.M2_3(x3, conv2_2)
        return x2, x3


class D_Module(nn.Module):
    def __init__(self, in_planes, is_dilation=False, bn=True, is_out=False, bilinear=False, NL='None'):
        '''
        :param filters:  [128,256,512]
        '''
        if is_dilation:
            self.dr = 2
        else:
            self.dr = 1
        self.is_out = is_out
        super(D_Module, self).__init__()
        self.conv1 = Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, same_padding=True, bn=bn,
                            dilation=self.dr)
        self.conv2 = Conv2d(in_channels=in_planes, out_channels=in_planes // 2, kernel_size=3, same_padding=True,
                            bn=bn, dilation=self.dr)
        if self.is_out:
            self.conv3 = Conv2d(in_channels=in_planes // 2, out_channels=1, kernel_size=1, same_padding=True,
                                bn=bn, dilation=1, NL=NL)
        self.bilinear = bilinear
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_planes // 2, in_planes // 2, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.up(x)
        if self.is_out:
            y = self.conv3(x)
        else:
            y = x
        return x, y


class Decoder_MLFN(nn.Module):
    def __init__(self, filters):
        '''
        :param filters:  [128,256,512]
        '''
        super(Decoder_MLFN, self).__init__()
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # nn.Upsample()

        self.M5 = Up_MLFN(filters[2], 64)
        self.M3 = Up_MLFN(filters[1], 64)
        self.M2 = Up_MLFN(filters[0], 64)

        self.fushion = Conv2d(64 * 3, 128, 1, 1, NL='relu', same_padding=True, bn=True)

    def forward(self, *input):
        conv2_2, conv3_3, conv5_x = input
        out_size = conv2_2.size()[2:]
        x5 = self.M5(conv5_x)
        x5 = F.interpolate(x5, size=out_size, mode='bilinear', align_corners=True)
        x3 = self.M3(conv3_3)
        x3 = F.interpolate(x3, size=out_size, mode='bilinear', align_corners=True)
        x2 = self.M2(conv2_2)

        x = torch.cat([x2, x3, x5], dim=1)
        x = self.fushion(x)
        return x


class Decoder_MSF(nn.Module):
    def __init__(self, filters):
        '''
        :param filters:  [128,256,512]
        '''
        super(Decoder_MSF, self).__init__()
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # nn.Upsample()

        self.M5 = Up_MLFN(filters[2], 64)
        self.M3 = Up_MLFN(filters[1], 64)
        self.M2 = Up_MLFN(filters[0], 64)

        self.fushion = Conv2d(64 * 3, 128, 1, 1, NL='relu', same_padding=True, bn=True)

    def forward(self, *input):
        conv2_2, conv3_3, conv5_x = input
        out_size = conv2_2.size()[2:]
        x5 = self.M5(conv5_x)
        x5 = F.interpolate(x5, size=out_size, mode='bilinear', align_corners=True)
        x3 = self.M3(conv3_3)
        x3 = F.interpolate(x3, size=out_size, mode='bilinear', align_corners=True)
        x2 = self.M2(conv2_2)

        x = torch.cat([x2, x3, x5], dim=1)
        x = self.fushion(x)
        return x


class Decoder03(nn.Module):
    def __init__(self, filters, res=False, h_out=True):
        '''
        :param filters:  [32,64,128,256,256]
        '''
        super(Decoder03, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.h_out = h_out
        self.conv_4_1 = Up_Block(filters[3] + filters[4], filters[3], residual=res)
        self.conv_3_2 = Up_Block(filters[2] + filters[3], filters[2], residual=res)
        self.conv_2_3 = Up_Block(filters[1] + filters[2], filters[1], residual=res)
        self.conv4_reduce_dim = Conv2d(512, filters[-2], 1, 1, NL='relu', same_padding=True, bn=True)
        # self.conv4_reduce_dim = nn.Conv2d(512, filters[-2], 1, 1, padding=0)
        self.conv3_reduce_dim = Conv2d(256, filters[-3], 1, 1, NL='relu', same_padding=True, bn=True)
        # self.conv3_reduce_dim = nn.Conv2d(256, filters[-3], 1, 1, padding=0)
        self.conv2_reduce_dim = Conv2d(128, filters[-4], 1, 1, NL='relu', same_padding=True, bn=True)
        # self.conv2_reduce_dim = nn.Conv2d(128, filters[-4], 1, 1, padding=0)
        if self.h_out:
            self.H = nn.Sequential(Conv2d(filters[1], filters[0], 1, 1, NL='relu', same_padding=True, bn=True),
                                   Conv2d(filters[0], filters[0], 3, 1, NL='relu', same_padding=True, bn=True),
                                   Conv2d(filters[0], filters[0], 3, 1, NL='relu', same_padding=True, bn=True)
                                   )

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_x = input
        x4_1 = self.conv_4_1(torch.cat([self.conv4_reduce_dim(conv4_3), conv5_x], 1))
        x3_2 = self.conv_3_2(torch.cat([self.conv3_reduce_dim(conv3_3), self.up(x4_1)], 1))
        x2_3 = self.conv_2_3(torch.cat([self.conv2_reduce_dim(conv2_2), self.up(x3_2)], 1))
        if self.h_out:
            x2_3 = self.H(x2_3)
        return x2_3, x3_2, x4_1


class Decoder_simple_seg(nn.Module):
    def __init__(self, filters, res=False):
        '''
        :param filters:  [64,128,256,512,512]
        '''
        super(Decoder_simple_seg, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_4_1 = Up_Block(filters[3] + filters[4], filters[3], residual=res)
        self.conv_3_2 = Up_Block(filters[2] + filters[3], filters[2], residual=res)
        self.conv_2_3 = Up_Block(filters[1] + filters[2], filters[1], residual=res)
        self.H = nn.Sequential(Conv2d(filters[1], filters[0], 1, 1, NL='relu', same_padding=True, bn=True),
                               Conv2d(filters[0], filters[0], 3, 1, NL='relu', same_padding=True, bn=True),
                               Conv2d(filters[0], filters[0] // 2, 3, 1, NL='relu', same_padding=True, bn=True)
                               )
        self.Seg = nn.Sequential(Conv2d(filters[1], filters[0], 1, 1, NL='relu', same_padding=True, bn=True),
                                 Conv2d(filters[0], filters[0], 3, 1, NL='relu', same_padding=True, bn=True),
                                 Conv2d(filters[0], filters[0] // 2, 3, 1, NL='relu', same_padding=True, bn=True)
                                 )
        back_feature = [512, 512, 512]
        self.rf_module = make_layers(back_feature, in_channels=512, dilation=True)

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3 = input
        conv5_3 = self.rf_module(conv4_3)
        x4_1 = self.conv_4_1(torch.cat([conv4_3, conv5_3], 1))
        x3_2 = self.conv_3_2(torch.cat([conv3_3, self.up(x4_1)], 1))
        x2_3 = self.conv_2_3(torch.cat([conv2_2, self.up(x3_2)], 1))
        output = self.H(x2_3)
        seg_out = self.Seg(x2_3)
        return [output, seg_out]


class DCM(nn.Module):
    def __init__(self, channel):
        super(DCM, self).__init__()
        self.DCM1 = DCMLayer(1, channel)
        self.DCM3 = DCMLayer(3, channel)
        self.DCM5 = DCMLayer(5, channel)
        # self.conv_1x1 = nn.Conv2d(channel * 4, channel, 1, padding=0, bias=True)
        self.conv_1x1 = Conv2d(channel * 4, channel, 1, bn=True)

    def forward(self, x):
        dcm1 = self.DCM1(x)
        dcm3 = self.DCM3(x)
        dcm5 = self.DCM5(x)
        out = torch.cat([x, dcm1, dcm3, dcm5], dim=1)
        out = self.conv_1x1(out)
        return out


class DCMLayer(nn.Module):
    def __init__(self, k, channel=512):
        super(DCMLayer, self).__init__()
        self.k = k
        self.channel = channel  # 512
        # self.conv_1x1 = nn.Conv2d(channel, channel // 4, 1, padding=0, bias=True)
        self.conv_1x1_01 = Conv2d(channel, channel // 4, 1, bn=True)
        self.conv_1x1_02 = Conv2d(channel, channel // 4, 1, bn=True)
        self.fuse = Conv2d(channel // 4, channel, 1, bn=True)

        # self.fuse = nn.Conv2d(channel // 4, channel, 1, padding=0, bias=True)
        self.dw_conv = nn.Conv2d(channel // 4, channel // 4, self.k, padding=(self.k - 1) // 2, groups=channel // 4)
        # self.bn = nn.BatchNorm2d(channel // 4)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(k)
        # print(self.dw_conv.weight.size())

    def forward(self, x):
        N, C, H, W = x.size()
        # [N * C/4 * H * W]
        f = self.conv_1x1_01(x)
        # [N * C/4 * K * K]
        g = self.conv_1x1_02(self.pool(x))

        f_list = torch.split(f, 1, 0)
        g_list = torch.split(g, 1, 0)
        out = []
        for i in range(N):
            # [1* C/4 * H * W]
            f_one = f_list[i]
            # [C/4 * 1 * K * K]
            # print(g_list[i].size())
            g_one = g_list[i].squeeze(0).unsqueeze(1)
            # print(g_one.size())
            self.dw_conv.weight = nn.Parameter(g_one)

            # [1* C/4 * H * W]
            o = self.dw_conv(f_one)
            o = self.relu(o)
            out.append(o)

        # [N * C/4 * H * W]
        y = torch.cat(out, dim=0)
        y = self.fuse(y)

        return y


class FPN_Deocder(nn.Module):
    def __init__(self, filters, res=False, reduction_dim=64):
        '''
        :param filters:  [64,128,256,512,512]
        '''
        super(FPN_Deocder, self).__init__()
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # top layer
        self.toplayer = nn.Conv2d(512, reduction_dim, kernel_size=1, stride=1, padding=0)
        # self.conv_4_1 = Up_Block(filters[3] + filters[4], filters[3], residual=res)
        # self.conv_3_2 = Up_Block(filters[2] + filters[3], filters[2], residual=res)
        # self.conv_2_3 = Up_Block(filters[1] + filters[2], filters[1], residual=res)
        # self.conv5_reduce_dim = nn.Conv2d(512, 64, 1, 1, padding=0)
        self.conv4_reduce_dim = Conv2d(512, reduction_dim, 1, 1, same_padding=True, bn=True)
        self.conv3_reduce_dim = Conv2d(256, reduction_dim, 1, 1, same_padding=True, bn=True)
        self.conv2_reduce_dim = Conv2d(128, reduction_dim, 1, 1, same_padding=True, bn=True)

        # smooth layers
        self.smooth1 = Conv2d(reduction_dim, reduction_dim, kernel_size=3, stride=1, same_padding=True, bn=True)
        self.smooth2 = Conv2d(reduction_dim, reduction_dim, kernel_size=3, stride=1, same_padding=True, bn=True)
        self.smooth3 = Conv2d(reduction_dim, reduction_dim, kernel_size=3, stride=1, same_padding=True, bn=True)

        # self.H = nn.Sequential(Conv2d(filters[1], filters[0], 1, 1, NL='relu', same_padding=True, bn=True),
        #                        Conv2d(filters[0], filters[0], 3, 1, NL='relu', same_padding=True, bn=True),
        #                        Conv2d(filters[0], filters[0], 3, 1, NL='relu', same_padding=True, bn=True)
        #                        )

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_x = input

        # top-down
        p5 = self.toplayer(conv5_x)
        p4 = self._upsample_add(p5, self.conv4_reduce_dim(conv4_3))
        p3 = self._upsample_add(p4, self.conv3_reduce_dim(conv3_3))
        p2 = self._upsample_add(p3, self.conv2_reduce_dim(conv2_2))

        # smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # x4_1 = self.conv_4_1(torch.cat([self.conv4_reduce_dim(conv4_3), conv5_x], 1))
        # x3_2 = self.conv_3_2(torch.cat([self.conv3_reduce_dim(conv3_3), self.up(x4_1)], 1))
        # x2_3 = self.conv_2_3(torch.cat([self.conv2_reduce_dim(conv2_2), self.up(x3_2)], 1))
        # output = self.H(x2_3)
        return p2, p3, p4, p5


class UNet_Decoder(nn.Module):
    def __init__(self, filters, res=False, add_rf=False):
        '''
        :param filters:  [64,128,256,512,512]
        '''
        super(UNet_Decoder, self).__init__()
        self.add_rf = add_rf
        # if self.add_rf:
        #
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_4_1 = Up_Block(filters[3] + filters[4], filters[3], residual=res)
        self.conv_3_2 = Up_Block(filters[2] + filters[3], filters[2], residual=res)
        self.conv_2_3 = Up_Block(filters[1] + filters[2], filters[1], residual=res)
        self.H = nn.Sequential(Conv2d(filters[1], filters[0], 1, 1, NL='relu', same_padding=True, bn=True),
                               Conv2d(filters[0], filters[0], 3, 1, NL='relu', same_padding=True, bn=True),
                               Conv2d(filters[0], filters[0], 3, 1, NL='relu', same_padding=True, bn=True),
                               Conv2d(filters[0], filters[0] // 2, 3, 1, NL='relu', same_padding=True, bn=True),
                               Conv2d(filters[0] // 2, filters[0] // 2, 3, 1, NL='relu', same_padding=True, bn=True),
                               )

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input
        if self.add_rf:
            x4_1 = self.conv_4_1(torch.cat([conv4_3, conv5_3], 1))
        else:
            x4_1 = self.conv_4_1(torch.cat([conv4_3, self.up(conv5_3)], 1))
        x3_2 = self.conv_3_2(torch.cat([conv3_3, self.up(x4_1)], 1))
        x2_3 = self.conv_2_3(torch.cat([conv2_2, self.up(x3_2)], 1))
        output = self.H(x2_3)
        return output


class UNet_Official_Decoder(nn.Module):
    def __init__(self, filters, res=False, add_rf=False):
        '''
        :param filters:  [64,128,256,512,512]
        '''
        super(UNet_Official_Decoder, self).__init__()
        self.add_rf = add_rf
        # if self.add_rf:
        #
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_4_1 = Up_Block(filters[3] + filters[4], filters[3], residual=res)
        self.conv_3_2 = Up_Block(filters[2] + filters[3], filters[2], residual=res)
        self.conv_2_3 = Up_Block(filters[1] + filters[2], filters[1], residual=res)
        self.H = nn.Sequential(Conv2d(filters[1], filters[0], 1, 1, NL='relu', same_padding=True, bn=True),
                               Conv2d(filters[0], filters[0], 3, 1, NL='relu', same_padding=True, bn=True)
                               )

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input
        if self.add_rf:
            x4_1 = self.conv_4_1(torch.cat([conv4_3, conv5_3], 1))
        else:
            x4_1 = self.conv_4_1(torch.cat([conv4_3, self.up(conv5_3)], 1))
        x3_2 = self.conv_3_2(torch.cat([conv3_3, self.up(x4_1)], 1))
        x2_3 = self.conv_2_3(torch.cat([conv2_2, self.up(x3_2)], 1))
        output = self.H(x2_3)
        return output


class Multi_Scale_Module(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Multi_Scale_Module, self).__init__()
        self.branch_1 = nn.Sequential(
            Conv2d(inchannel, inchannel // 2, kernel_size=1, stride=1, same_padding=True, bn=True, dilation=1),
            Conv2d(inchannel // 2, inchannel // 2, kernel_size=3, stride=1, same_padding=True, bn=True, dilation=1),
            Conv2d(inchannel // 2, inchannel // 2, kernel_size=3, stride=1, same_padding=True, bn=True, dilation=1)
        )

        self.branch_2 = nn.Sequential(
            Conv2d(inchannel, inchannel // 2, kernel_size=1, stride=1, same_padding=True, bn=True, dilation=1),
            Conv2d(inchannel // 2, inchannel // 2, kernel_size=3, stride=1, same_padding=True, bn=True, dilation=2),
            Conv2d(inchannel // 2, inchannel // 2, kernel_size=3, stride=1, same_padding=True, bn=True, dilation=2)
        )
        self.branch_3 = nn.Sequential(
            Conv2d(inchannel, inchannel // 2, kernel_size=1, stride=1, same_padding=True, bn=True, dilation=1),
            Conv2d(inchannel // 2, inchannel // 2, kernel_size=3, stride=1, same_padding=True, bn=True, dilation=3),
            Conv2d(inchannel // 2, inchannel // 2, kernel_size=3, stride=1, same_padding=True, bn=True, dilation=3)
        )
        # self.sfam = CFFM(planes=inchannel // 2 * 3)
        self.conv_1x1 = Conv2d(inchannel // 2 * 3, outchannel, kernel_size=1, stride=1, same_padding=True, bn=True,
                               dilation=1)

        self.conv_3x3 = Conv2d(outchannel, outchannel, kernel_size=3, stride=1, same_padding=True, bn=True, dilation=1)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv_1x1(x)
        x = self.conv_3x3(x)

        return x


class SE(nn.Module):
    def __init__(self, planes, compress_ratio=16, add_sa=False):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.add_sa = add_sa
        # self.sigmoid = nn.Sigmoid()
        self.planes = planes
        self.fc = nn.Sequential(
            Conv2d(self.planes, self.planes // compress_ratio, 1, 1, same_padding=True),
            Conv2d(self.planes // compress_ratio, self.planes, 1, 1, same_padding=True, NL='sigmoid'))
        if self.add_sa:
            self.conv1 = Conv2d(self.planes, 1, 1, 1, same_padding=True)

    def forward(self, x):
        channel_factor = self.avgpool(x)
        channel_factor = self.fc(channel_factor)
        channel_map = channel_factor * x

        if self.add_sa:
            spatial_map = self.conv1(x)
            mask_1 = torch.ge(spatial_map, channel_map)
            mask_1 = mask_1.float()
            x = mask_1 * spatial_map + (1 - mask_1) * channel_map
        else:
            x = channel_map

        return x


class MSB_01(nn.Module):
    def __init__(self, inchannel, cffm=False):
        super(MSB_01, self).__init__()
        self.branch_1 = nn.Sequential(
            Conv2d(inchannel, inchannel // 2, kernel_size=1, stride=1, same_padding=True, bn=True, dilation=1),
            Conv2d(inchannel // 2, inchannel // 2, kernel_size=3, stride=1, same_padding=True, bn=True, dilation=1)
        )
        self.branch_2 = nn.Sequential(
            Conv2d(inchannel, inchannel // 2, kernel_size=1, stride=1, same_padding=True, bn=True, dilation=1),
            Conv2d(inchannel // 2, inchannel // 2, kernel_size=3, stride=1, same_padding=True, bn=True, dilation=2)
        )
        self.branch_3 = nn.Sequential(
            Conv2d(inchannel, inchannel // 2, kernel_size=1, stride=1, same_padding=True, bn=True, dilation=1),
            Conv2d(inchannel // 2, inchannel // 2, kernel_size=3, stride=1, same_padding=True, bn=True, dilation=3)
        )
        # self.scale_1 = Conv2d(inchannel // 2, 1, kernel_size=3, stride=1, same_padding=True)
        # self.scale_2 = Conv2d(inchannel // 2, 1, kernel_size=3, stride=1, same_padding=True)
        # self.scale_3 = Conv2d(inchannel // 2, 1, kernel_size=3, stride=1, same_padding=True)
        self.is_cffm = cffm
        if self.is_cffm:
            self.sfam = CFFM(planes=inchannel // 2 * 3)
        self.conv_1x1 = Conv2d(inchannel // 2 * 3, inchannel, kernel_size=3, stride=1, same_padding=True)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        # x1 = self.scale_1(x1)
        # x2 = self.scale_2(x2)
        # x3 = self.scale_3(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        if self.is_cffm:
            x = self.sfam(x)
        x = self.conv_1x1(x)
        return x


class SAM(nn.Module):
    # def __init__(self, inchannel, cffm=False):
    def __init__(self, inchannel, dilation_ratio_list, branch_number=3, planes_channel=256):
        super(SAM, self).__init__()
        self.red_ratio = inchannel // planes_channel

        self.branch_1 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[0])
        )
        self.branch_2 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[1])
        )
        self.branch_3 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[2])
        )
        self.branch_4 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[3])
        )
        self.fuse = Conv2d(inchannel // self.red_ratio * branch_number, inchannel, kernel_size=3, stride=1,
                           same_padding=True, bn=True)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.branch_4(x)
        # x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = self.fuse(x)
        return x


class PSM(nn.Module):
    # def __init__(self, inchannel, cffm=False):
    def __init__(self, inchannel, dilation_ratio_list, branch_number=3, red_ratio=4):
        super(PSM, self).__init__()
        self.red_ratio = red_ratio
        self.branch_1 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[0])
        )
        self.branch_2 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[1])
        )
        self.branch_3 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[2])
        )
        self.branch_4 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[3])
        )
        self.fuse = Conv2d(inchannel // self.red_ratio * branch_number, inchannel, kernel_size=3, stride=1,
                           same_padding=True, bn=True)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.branch_4(x)
        # x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.fuse(x)
        return x


class MCN(nn.Module):
    # def __init__(self, inchannel, cffm=False):
    def __init__(self, inchannel, dilation_ratio_list, branch_number=4, red_ratio=4):
        super(MCN, self).__init__()
        self.red_ratio = red_ratio
        self.branch_1 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[0])
        )
        self.branch_2 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[1])
        )
        self.branch_3 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[2])
        )
        self.branch_4 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[3])
        )
        self.fuse = Conv2d(inchannel // self.red_ratio * branch_number, inchannel, kernel_size=3, stride=1,
                           same_padding=True, bn=True)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.branch_4(x)
        # x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.fuse(x)
        return x, [x1, x2, x3, x4]


class SiT(nn.Module):
    # def __init__(self, inchannel, cffm=False):
    def __init__(self, inchannel, dilation_ratio_list, group=4, group_channels=64):
        super(SiT, self).__init__()
        self.group = group

        for i, idx in enumerate(range(self.group)):
            setattr(self, 'conv_1x1_' + str(idx),
                    Conv2d(in_channels=inchannel, out_channels=group_channels, kernel_size=1, stride=1, bn=True))
        self.branch_1 = nn.Sequential(
            Conv2d(group_channels, group_channels, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[0])
        )
        self.branch_2 = nn.Sequential(
            Conv2d(group_channels, group_channels, kernel_size=3, stride=1, same_padding=True, bn=True,
                   dilation=dilation_ratio_list[1])
        )
        self.branch_3 = nn.Sequential(
            Conv2d(group_channels, group_channels, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[2])
        )
        self.branch_4 = nn.Sequential(
            Conv2d(group_channels, group_channels, kernel_size=3, stride=1, same_padding=True, bn=True,
                   dilation=dilation_ratio_list[3])
        )
        self.fuse = Conv2d(group_channels * self.group, inchannel, kernel_size=3, stride=1,
                           same_padding=True, bn=True)

    def forward(self, x, train=True):
        # N, C, H, W = x.size()
        # x_split = x.view(N, self.group, int(C / self.group), H, W)
        # x1 = x_split[:, 0, :, :, :]
        # x2 = x_split[:, 1, :, :, :]
        # x3 = x_split[:, 2, :, :, :]
        # x4 = x_split[:, 3, :, :, :]
        x_groups = []
        for i in range(self.group):
            m = getattr(self, 'conv_1x1_' + str(i))
            x_groups.append(m(x))
        x0 = x_groups[0]
        x1 = self.branch_1(x_groups[1])
        x2 = self.branch_2(x_groups[2])
        if train:
            w1 = torch.rand(1)[0]
        else:
            w1 = 0.5
        x2 = w1 * x1 + (1 - w1) * x2
        x3 = self.branch_3(x_groups[3])
        if train:
            w2 = torch.rand(1)[0]
        else:
            w2 = 0.5
        x3 = w2 * x2 + (1 - w2) * x3
        x4 = self.branch_4(x_groups[4])
        if train:
            w3 = torch.rand(1)[0]
        else:
            w3 = 0.5
        x4 = w3 * x3 + (1 - w3) * x4

        # x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = torch.cat([x0, x1, x2, x3, x4], dim=1)
        x = self.fuse(x)
        return x


class PSM01(nn.Module):
    # def __init__(self, inchannel, cffm=False):
    def __init__(self, inchannel, dilation_ratio_list, branch_number=3, red_ratio=4):
        super(PSM01, self).__init__()
        self.red_ratio = red_ratio
        self.branch_1 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[0])
        )
        self.branch_2 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[1])
        )
        self.branch_3 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[2])
        )
        self.branch_4 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[3])
        )

        self.conv1_3x3 = Conv2d(inchannel // self.red_ratio * 2, inchannel // self.red_ratio, kernel_size=3, stride=1,
                                same_padding=True, bn=True)
        self.conv2_3x3 = Conv2d(inchannel // self.red_ratio * 2, inchannel // self.red_ratio, kernel_size=3, stride=1,
                                same_padding=True, bn=True)
        self.conv3_3x3 = Conv2d(inchannel // self.red_ratio * 2, inchannel // self.red_ratio, kernel_size=3, stride=1,
                                same_padding=True, bn=True)

        self.fuse = Conv2d(inchannel // self.red_ratio * branch_number, inchannel, kernel_size=3, stride=1,
                           same_padding=True, bn=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        r = 16
        self.sefc = nn.Sequential(
            Conv2d(inchannel, inchannel // r, 1, 1, same_padding=True),
            Conv2d(inchannel // r, inchannel, 1, 1, same_padding=True, NL='sigmoid')
        )

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x12 = torch.cat([x1, x2], dim=1)
        x12 = self.conv1_3x3(x12)
        x3 = self.branch_3(x)
        x23 = torch.cat([x12, x3], dim=1)
        x23 = self.conv2_3x3(x23)
        x4 = self.branch_4(x)

        x34 = torch.cat([x23, x4], dim=1)
        x34 = self.conv3_3x3(x34)

        gc = self.avgpool(x)
        weight = self.sefc(gc)
        # x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = torch.cat([x1, x12, x23, x34], dim=1)
        x = self.fuse(x)
        x = x * weight
        return x


class SAM01(nn.Module):
    def __init__(self, inchannel, outchannel, dilation_ratio_list, branch_number=3, planes_channel=256, add_res=False):
        super(SAM01, self).__init__()
        self.red_ratio = inchannel // planes_channel
        self.d_ratio = 1
        self.branch_number = branch_number
        self.add_res = add_res
        assert len(dilation_ratio_list) == branch_number
        for i, idx in enumerate(range(branch_number)):
            setattr(self, 'branch_' + str(idx),
                    self.make_branch(inchannel, inchannel // self.red_ratio, d_rate=dilation_ratio_list[i]))
        if self.add_res:
            self.fuse = Conv2d(inchannel // self.red_ratio * branch_number + inchannel, outchannel, kernel_size=1,
                               stride=1, same_padding=True, bn=True)
        else:
            self.fuse = Conv2d(inchannel // self.red_ratio * branch_number, outchannel, kernel_size=1, stride=1,
                               same_padding=True, bn=True)

    def make_branch(self, in_channels, out_channels, d_rate):
        layers = []
        layers += [Conv2d(in_channels, out_channels, 1, same_padding=True, bn=True, dilation=1)]
        layers += [Conv2d(out_channels, out_channels, 3, same_padding=True, bn=True, dilation=d_rate)]
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []
        if self.add_res:
            outputs.append(x)
        for i in range(self.branch_number):
            m = getattr(self, 'branch_' + str(i))
            outputs.append(m(x))
        x = torch.cat(outputs, dim=1)
        x = self.fuse(x)
        return x


class SAM_LW(nn.Module):
    def __init__(self, inchannel, dilation_ratio_list, branch_number=3, planes_channel=256, add_res=False):
        super(SAM_LW, self).__init__()
        self.red_ratio = inchannel // planes_channel
        self.d_ratio = 1
        self.branch_number = branch_number
        self.add_res = add_res
        assert len(dilation_ratio_list) == branch_number
        for i, idx in enumerate(range(branch_number)):
            setattr(self, 'branch_' + str(idx),
                    self.make_branch(inchannel, inchannel // self.red_ratio, d_rate=dilation_ratio_list[i]))
        if self.add_res:
            self.fuse = Conv2d(inchannel // self.red_ratio * branch_number + inchannel, inchannel, kernel_size=1,
                               stride=1, same_padding=True, bn=True)
        else:
            self.fuse = Conv2d(inchannel // self.red_ratio * branch_number, inchannel, kernel_size=1, stride=1,
                               same_padding=True, bn=True)

    def make_branch(self, in_channels, out_channels, d_rate):
        layers = []
        layers += [Conv2d(in_channels, out_channels, 1, same_padding=True, bn=True, dilation=1)]
        layers += [Conv2d(out_channels, out_channels, 3, same_padding=True, bn=True, dilation=d_rate)]
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []
        if self.add_res:
            outputs.append(x)
        for i in range(self.branch_number):
            m = getattr(self, 'branch_' + str(i))
            outputs.append(m(x))
        x = torch.cat(outputs, dim=1)
        x = self.fuse(x)
        return x


class trident_block(nn.Module):
    def __init__(self, inchannel, dilation_ratio, planes_channel=128):
        super(trident_block, self).__init__()
        self.dilatedratio = dilation_ratio
        self.planes_channel = planes_channel
        self.inchannel = inchannel
        self.Conv_block = nn.Sequential(
            Conv2d(self.inchannel, self.planes_channel, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(self.planes_channel, self.planes_channel, kernel_size=3, stride=1, same_padding=True, bn=True,
                   dilation=dilation_ratio),
            Conv2d(self.planes_channel, self.inchannel, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1, NL='None')
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.Conv_block(x)
        x += input
        x = self.relu(x)
        return x


class SX(nn.Module):
    # def __init__(self, inchannel, cffm=False):
    def __init__(self, inchannel, dilation_ratio_list, planes_channel=256):
        super(SX, self).__init__()
        self.tb1 = trident_block(inchannel=512, planes_channel=planes_channel, dilation_ratio=dilation_ratio_list[0])
        self.tb2 = trident_block(inchannel=512, planes_channel=planes_channel, dilation_ratio=dilation_ratio_list[1])
        self.tb3 = trident_block(inchannel=512, planes_channel=planes_channel, dilation_ratio=dilation_ratio_list[2])

        self.fuse = Conv2d(inchannel * 3, inchannel, kernel_size=3, stride=1, same_padding=True, bn=True)

    def forward(self, x):
        x1 = self.tb1(x)
        x2 = self.tb2(x)
        x3 = self.tb3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fuse(x)
        return x


class MSB_DCN(nn.Module):
    # def __init__(self, inchannel, cffm=False):
    def __init__(self, inchannel, dilation_ratio_list, reduction_ratio=2, add_x=False):
        super(MSB_DCN, self).__init__()
        self.red_ratio = reduction_ratio
        self.add_x = add_x
        # dilation_ratio_list = [1, 2, 3]
        # dilation_ratio_list = [2, 4, 6]
        self.branch_1 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[0])
        )
        self.branch_2 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[1])
        )
        self.branch_3 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=dilation_ratio_list[2])
        )
        # self.scale_1 = Conv2d(inchannel // 2, 1, kernel_size=3, stride=1, same_padding=True)
        # self.scale_2 = Conv2d(inchannel // 2, 1, kernel_size=3, stride=1, same_padding=True)
        # self.scale_3 = Conv2d(inchannel // 2, 1, kernel_size=3, stride=1, same_padding=True)
        # self.is_cffm = cffm
        # if self.is_cffm:
        #     self.sfam = CFFM(planes=inchannel // self.red_ratio * 3)
        if self.add_x:
            self.conv_1x1 = Conv2d(inchannel // self.red_ratio * 3 + inchannel, inchannel, kernel_size=1, stride=1,
                                   same_padding=True, bn=True)
        else:
            self.conv_1x1 = Conv2d(inchannel // self.red_ratio * 3, inchannel, kernel_size=1, stride=1,
                                   same_padding=True, bn=True)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        # x1 = self.scale_1(x1)
        # x2 = self.scale_2(x2)
        # x3 = self.scale_3(x3)
        if self.add_x:
            x = torch.cat([x, x1, x2, x3], dim=1)
            # x = self.sfam(x)
        else:
            x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv_1x1(x)
        return x


class MSB_AB_DCN(nn.Module):
    # def __init__(self, inchannel, cffm=False):
    def __init__(self, inchannel, dilation_ratio_list, reduction_ratio=2):
        super(MSB_AB_DCN, self).__init__()
        self.red_ratio = reduction_ratio
        # dilation_ratio_list = [1, 2, 3]
        # dilation_ratio_list = [2, 4, 6]
        self.branch_1 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True,
                   bn=True, dilation=dilation_ratio_list[0])
        )
        self.branch_2 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True,
                   bn=True, dilation=dilation_ratio_list[1])
        )
        self.branch_3 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                   same_padding=True,
                   bn=True, dilation=dilation_ratio_list[2])
        )
        self.conv_1x1 = Conv2d(inchannel // self.red_ratio * 3 + inchannel, inchannel, kernel_size=1, stride=1,
                               same_padding=True)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        # x1 = self.scale_1(x1)
        # x2 = self.scale_2(x2)
        # x3 = self.scale_3(x3)
        x = torch.cat([x, x1, x2, x3], dim=1)
        if self.is_cffm:
            x = self.sfam(x)
        x = self.conv_1x1(x)
        return x


# class SKNet(nn.Module):
#     def __init__(self, inchannel, dilation_ratio_list, reduction_ratio=2, cffm=False):
#         super(SKNet, self).__init__()
#         self.red_ratio = reduction_ratio
#         self.branch_1 = nn.Sequential(
#             Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
#                    dilation=1),
#             Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
#                    same_padding=True,
#                    bn=True, dilation=dilation_ratio_list[0])
#         )
#         self.branch_2 = nn.Sequential(
#             Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
#                    dilation=1),
#             Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
#                    same_padding=True,
#                    bn=True, dilation=dilation_ratio_list[1])
#         )
#         self.branch_3 = nn.Sequential(
#             Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
#                    dilation=1),
#             Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
#                    same_padding=True,
#                    bn=True, dilation=dilation_ratio_list[2])
#         )
#         self.is_cffm = cffm
#         if self.is_cffm:
#             self.sfam = CFFM(planes=inchannel // self.red_ratio * 3)
#         self.conv_1x1 = Conv2d(inchannel // self.red_ratio * 3 + inchannel, inchannel, kernel_size=1, stride=1,
#                                same_padding=True)
#
#     def forward(self, x):
#         x1 = self.branch_1(x)
#         x2 = self.branch_2(x)
#         x3 = self.branch_3(x)
#         x = torch.cat([x, x1, x2, x3], dim=1)
#         if self.is_cffm:
#             x = self.sfam(x)
#         x = self.conv_1x1(x)
#         return x


class MSB_DCNV2(nn.Module):
    def __init__(self, inchannel, cffm=False, reduction_ratio=2):
        super(MSB_DCNV2, self).__init__()
        self.red_ratio = reduction_ratio
        self.branch_1 = nn.Sequential(
            DCNV2(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                  dilation=1),
            DCNV2(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1, same_padding=True,
                  bn=True, dilation=1)
        )
        self.branch_2 = nn.Sequential(
            DCNV2(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                  dilation=1),
            DCNV2(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1, same_padding=True,
                  bn=True, dilation=2)
        )
        self.branch_3 = nn.Sequential(
            DCNV2(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                  dilation=1),
            DCNV2(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1, same_padding=True,
                  bn=True, dilation=3)
        )
        # self.scale_1 = Conv2d(inchannel // 2, 1, kernel_size=3, stride=1, same_padding=True)
        # self.scale_2 = Conv2d(inchannel // 2, 1, kernel_size=3, stride=1, same_padding=True)
        # self.scale_3 = Conv2d(inchannel // 2, 1, kernel_size=3, stride=1, same_padding=True)
        self.is_cffm = cffm
        if self.is_cffm:
            self.sfam = CFFM(planes=inchannel // 2 * 3)
        self.conv_1x1 = Conv2d(inchannel // self.red_ratio * 3 + inchannel, inchannel, kernel_size=1, stride=1,
                               same_padding=True, bn=True)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        # x1 = self.scale_1(x1)
        # x2 = self.scale_2(x2)
        # x3 = self.scale_3(x3)
        x = torch.cat([x, x1, x2, x3], dim=1)
        if self.is_cffm:
            x = self.sfam(x)
        x = self.conv_1x1(x)
        return x


class MSB_MRFB(nn.Module):
    def __init__(self, inchannel, reduction_ratio=2, cffm=False):
        super(MSB_MRFB, self).__init__()
        self.red_ratio = reduction_ratio
        self.branch_1 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1, same_padding=True,
                   bn=True, dilation=1)
        )
        self.branch_2 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            # 3*3
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1, same_padding=True,
                   bn=True, dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1, same_padding=True,
                   bn=True, dilation=2)

        )
        self.branch_3 = nn.Sequential(
            Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            # 5*5-->2 3*3
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1, same_padding=True,
                   bn=True, dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1, same_padding=True,
                   bn=True, dilation=1),
            Conv2d(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1, same_padding=True,
                   bn=True, dilation=3)
        )
        # self.scale_1 = Conv2d(inchannel // 2, 1, kernel_size=3, stride=1, same_padding=True)
        # self.scale_2 = Conv2d(inchannel // 2, 1, kernel_size=3, stride=1, same_padding=True)
        # self.scale_3 = Conv2d(inchannel // 2, 1, kernel_size=3, stride=1, same_padding=True)
        self.is_cffm = cffm
        if self.is_cffm:
            self.sfam = CFFM(planes=inchannel // self.red_ratio * 3)
        self.relu = nn.ReLU(inplace=False)
        self.conv_reduction_input = Conv2d(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, bn=True)
        self.conv_1x1 = Conv2d(inchannel // self.red_ratio * 4, inchannel, kernel_size=1, stride=1, same_padding=True,
                               bn=True)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        # x1 = self.scale_1(x1)
        # x2 = self.scale_2(x2)
        # x3 = self.scale_3(x3)
        x = torch.cat([self.conv_reduction_input(x), x1, x2, x3], dim=1)
        if self.is_cffm:
            x = self.sfam(x)
        x = self.conv_1x1(x)
        return x


class MSB_SPP(nn.Module):
    def __init__(self, inchannel, sizes, outchannel=256, cffm=False, reduction_factor=4):
        super(MSB_SPP, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.AvgPool2d(sizes[0]),
            Conv2d(inchannel, inchannel // reduction_factor, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // reduction_factor, inchannel // reduction_factor, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=1)
        )
        self.branch_2 = nn.Sequential(
            nn.AvgPool2d(sizes[1]),
            Conv2d(inchannel, inchannel // reduction_factor, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // reduction_factor, inchannel // reduction_factor, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=1)
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(sizes[2]),
            Conv2d(inchannel, inchannel // reduction_factor, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // reduction_factor, inchannel // reduction_factor, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=1)
        )
        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(sizes[3]),
            Conv2d(inchannel, inchannel // reduction_factor, kernel_size=1, stride=1, same_padding=True, bn=True,
                   dilation=1),
            Conv2d(inchannel // reduction_factor, inchannel // reduction_factor, kernel_size=3, stride=1,
                   same_padding=True, bn=True, dilation=1)
        )
        self.is_cffm = cffm
        if self.is_cffm:
            self.sfam = CFFM(planes=inchannel // reduction_factor * reduction_factor)
        self.conv_1x1 = Conv2d(inchannel // reduction_factor * 4 + inchannel, outchannel, kernel_size=1,
                               stride=1, same_padding=True)

    def forward(self, x):
        input_size = x.size()[2:]
        x1 = self.branch_1(x)
        x1 = F.interpolate(x1, size=input_size, mode='bilinear', align_corners=True)
        x2 = self.branch_2(x)
        x2 = F.interpolate(x2, size=input_size, mode='bilinear', align_corners=True)
        x3 = self.branch_3(x)
        x3 = F.interpolate(x3, size=input_size, mode='bilinear', align_corners=True)
        x4 = self.branch_4(x)
        x4 = F.interpolate(x4, size=input_size, mode='bilinear', align_corners=True)
        x = torch.cat([x, x1, x2, x3, x4], dim=1)
        if self.is_cffm:
            x = self.sfam(x)
        x = self.conv_1x1(x)
        return x


class MSB_02(nn.Module):
    def __init__(self, inchannel, cffm=False, seg_output=True):
        super(MSB_02, self).__init__()
        self.branch_1 = nn.Sequential(
            Conv2d(inchannel, inchannel // 2, kernel_size=1, stride=1, same_padding=True, bn=True, dilation=1),
            Conv2d(inchannel // 2, inchannel // 2, kernel_size=3, stride=1, same_padding=True, bn=True, dilation=1)
        )
        self.branch_2 = nn.Sequential(
            Conv2d(inchannel, inchannel // 2, kernel_size=1, stride=1, same_padding=True, bn=True, dilation=1),
            Conv2d(inchannel // 2, inchannel // 2, kernel_size=3, stride=1, same_padding=True, bn=True, dilation=2)
        )
        self.branch_3 = nn.Sequential(
            Conv2d(inchannel, inchannel // 2, kernel_size=1, stride=1, same_padding=True, bn=True, dilation=1),
            Conv2d(inchannel // 2, inchannel // 2, kernel_size=3, stride=1, same_padding=True, bn=True, dilation=3)
        )
        # self.scale_1 = Conv2d(inchannel // 2, 1, kernel_size=3, stride=1, same_padding=True)
        # self.scale_2 = Conv2d(inchannel // 2, 1, kernel_size=3, stride=1, same_padding=True)
        # self.scale_3 = Conv2d(inchannel // 2, 1, kernel_size=3, stride=1, same_padding=True)
        self.is_cffm = cffm
        self.seg_output = seg_output
        if self.is_cffm:
            self.sfam = CFFM(planes=inchannel // 2 * 3)
        self.conv_1x1 = Conv2d(inchannel // 2 * 3, inchannel, kernel_size=3, stride=1, same_padding=True)

    def forward(self, x):
        # seg_feature = None
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        # x1 = self.scale_1(x1)
        # x2 = self.scale_2(x2)
        # x3 = self.scale_3(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        seg_feature = x
        if self.is_cffm:
            x = self.sfam(x)
        x = self.conv_1x1(x)
        # if self.seg_output:
        #     return x, seg_feature
        return x
