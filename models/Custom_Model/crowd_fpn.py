from .vgg_block import vgg13_model, vgg16_model, VGG16_NoBN
import torch.nn as nn
import torch.nn.functional as F
import torch
from config import cfg
from .conv_utils import Conv2d, make_layers, CBAM, SAM
from .Decoder_Model import Decoder01, Decoder02, UNet_Decoder, FPN_Deocder, Decoder03, Multi_Scale_Module, MSB_01, \
    MSB_02, DCM, MSB_DCN, Decoder_simple_seg, MSB_SPP, MSB_MRFB, MSB_DCNV2, Decoder_MSB_UNet, Decoder_MLFN
from .sknet import SKUnit, SKUnit_ab
from .MLFPN import MLFPN, MSDecoder, MSN, CFFM
from torchvision import models


class Crowd_FPN_Baseline(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_FPN_Baseline, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512, 256, 256, 256], 512, batch_norm=bn, dilation=True)
        self.decoder = Decoder01(filters=[32, 64, 128, 256, 256])
        self.density_pred = Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False,
                                   NL='None')

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        reg_map = self.decoder(C2, C3, C4, C5)
        reg_map = self.density_pred(reg_map)
        if cfg.density_stride == 1:
            reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_DCM(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_DCM, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512, 256, 256, 256], 512, batch_norm=bn, dilation=True)
        self.decoder = DCM(256)
        # self.density_pred = Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False,
        #                            NL='None')

        self.density_pred = nn.Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=False, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        reg_map = self.decoder(C5)
        reg_map = self.density_pred(reg_map)
        if cfg.density_stride == 1:
            reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_MSB_CBAM(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_MSB_CBAM, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        self.cbam = CBAM(512, reduction_ratio=16, res_skip=True)
        self.decoder = DCM(512)
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        C5 = self.cbam(C5)
        reg_map = self.decoder(C5)
        reg_map = self.density_pred(reg_map)
        if cfg.density_stride == 1:
            reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_VGG_NoBN(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_VGG_NoBN, self).__init__()
        if backbone == 'vgg':
            # self.feature_c4 = VGG16_NoBN(True)
            self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
            self.features = make_layers(self.frontend_feat)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=False, dilation=True)
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=3, same_padding=True, bn=False),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=False),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )
        self._initialize_weights()
        if pretrained:
            mod = models.vgg16(pretrained=True)
            self.features.load_state_dict(mod.features[0:23].state_dict())

    def forward(self, x):
        input_size = x.size()[2:]
        C4 = self.features(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        reg_map = self.density_pred(C5)
        if cfg.density_stride == 1:
            reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Crowd_VGG(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_VGG, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
            # self.feature_c4 = VGG16_NoBN(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        reg_map = self.density_pred(C5)
        if cfg.density_stride == 1:
            reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_MRCM(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False, M=2):
        super(Crowd_MRCM, self).__init__()
        self.bn = bn
        if backbone == 'vgg':
            if bn:
                self.feature_c4 = vgg13_model(pretrained)
            else:
                self.feature_c4 = VGG16_NoBN(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        self.msb = SKUnit(inplanes=512, M=M, G=32, stride=1, L=32)
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        if self.bn:
            C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        else:
            C4 = self.feature_c4(x)
        C5 = self.back_encoder(C4)
        features = self.msb(C5)
        reg_map = self.density_pred(features)
        if cfg.density_stride == 1:
            reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_AB(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False, D=1):
        super(Crowd_AB, self).__init__()
        self.bn = bn
        if backbone == 'vgg':
            if bn:
                self.feature_c4 = vgg13_model(pretrained)
            else:
                self.feature_c4 = VGG16_NoBN(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        self.msb = SKUnit_ab(inplanes=512, D=D, M=1, G=32, stride=1, L=32)
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=3, same_padding=True, bn=bn),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=bn),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        if self.bn:
            C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        else:
            C4 = self.feature_c4(x)
        C5 = self.back_encoder(C4)
        features = self.msb(C5)
        reg_map = self.density_pred(features)
        if cfg.density_stride == 1:
            reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_MSB_DCN(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False, use_dcnv2=False, reduction_ratio=2):
        super(Crowd_MSB_DCN, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        if use_dcnv2:
            self.decoder = MSB_DCNV2(512, reduction_ratio=reduction_ratio)
        else:
            self.decoder = MSB_DCN(512, reduction_ratio=reduction_ratio, dilation_ratio_list=[1, 2, 3], add_x=False)
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        # print(C4.shape)
        x = self.decoder(C5)

        reg_map = self.density_pred(x)

        # if cfg.density_stride == 1:
        #     reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_MSB_UNet(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False, use_dcnv2=False, reduction_ratio=2):
        super(Crowd_MSB_UNet, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        if use_dcnv2:
            self.msb = MSB_DCNV2(512, reduction_ratio=reduction_ratio)
        else:
            self.msb = MSB_DCN(512, reduction_ratio=reduction_ratio, dilation_ratio_list=[2, 4, 6])
        self.decoder = Decoder_MSB_UNet([128, 256, 512])
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        C5 = self.msb(C5)
        reg_map, _ = self.decoder(C2, C3, C5)
        reg_map = self.density_pred(reg_map)
        if cfg.density_stride == 1:
            reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_MSB_MRFB(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False, reduction_ratio=2):
        super(Crowd_MSB_MRFB, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        self.decoder = MSB_MRFB(512, reduction_ratio=reduction_ratio)
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        reg_map = self.decoder(C5)
        reg_map = self.density_pred(reg_map)
        if cfg.density_stride == 1:
            reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_MSB_SCF(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_MSB_SCF, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        self.decoder = MSB_DCN(512)
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        reg_map = self.decoder(C5)
        reg_map = self.density_pred(reg_map)
        if cfg.density_stride == 1:
            reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


# class Crowd_MSB_Seg(nn.Module):
#     def __init__(self, backbone='vgg', pretrained=True, bn=False):
#         super(Crowd_MSB_Seg, self).__init__()
#         if backbone == 'vgg':
#             self.feature_c4 = vgg13_model(pretrained)
#         self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
#         # self.decoder = Decoder01(filters=[32, 64, 128, 256, 256])
#         self.decoder = MSB_DCN(512)
#         self.seg_pred = nn.Sequential(
#             Conv2d(in_channels=512, out_channels=128, kernel_size=3, same_padding=True, bn=True),
#             Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
#             Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='sigmoid')
#         )
#
#     def forward(self, x):
#         input_size = x.size()[2:]
#         C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
#         C5 = self.back_encoder(C4)
#         seg_map = self.decoder(C5)
#         seg_map = self.seg_pred(seg_map)
#         if cfg.density_stride == 1:
#             seg_map = F.interpolate(seg_map, size=input_size, mode='bilinear', align_corners=True)
#         return seg_map


class Crowd_MSB_SAM(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_MSB_SAM, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        self.sam = SAM(512)
        self.decoder = DCM(512)
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x, seg):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        C5 = self.sam(C5, seg)
        reg_map = self.decoder(C5)
        reg_map = self.density_pred(reg_map)
        if cfg.density_stride == 1:
            reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_MSB_DCN_Scale(nn.Module):
    def __init__(self, dilation_ratio, backbone='vgg', pretrained=True,
                 bn=False, reduction_ratio=2):
        super(Crowd_MSB_DCN_Scale, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        self.decoder = MSB_DCN(512, reduction_ratio=reduction_ratio, dilation_ratio_list=dilation_ratio)
        self.spp_can = MSB_SPP(512, sizes=[1, 2, 3, 4], outchannel=512)
        self.scale_pred = nn.Sequential(
            Conv2d(512, 32, 1, 1, same_padding=True, bn=True),
            Conv2d(32, 16, 3, 1, same_padding=True, bn=True),
            Conv2d(16, 4, 1, 1, same_padding=True, bn=False, NL='None')
        )

        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        # seg_size = (input_size[0] // 2, input_size[1] // 2)
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        # scale_feature = self.spp_can(C4)

        output = self.decoder(C5)
        scale_map = self.scale_pred(output)
        reg_map = self.density_pred(output)
        reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        scale_map = F.interpolate(scale_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map, scale_map


class Crowd_MSB_DCN_Scale_01(nn.Module):
    def __init__(self, dilation_ratio, backbone='vgg', pretrained=True,
                 bn=False, reduction_ratio=2):
        super(Crowd_MSB_DCN_Scale_01, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        self.decoder = MSB_DCN(512, reduction_ratio=reduction_ratio, dilation_ratio_list=dilation_ratio)
        self.spp_can = MSB_SPP(512, sizes=[1, 2, 3, 4], outchannel=512)
        self.scale_pred = nn.Sequential(
            Conv2d(512, 32, 1, 1, same_padding=True, bn=True),
            Conv2d(32, 16, 3, 1, same_padding=True, bn=True),
            Conv2d(16, 4, 1, 1, same_padding=True, bn=False, NL='None')
        )
        self.mlfn = Decoder_MLFN(filters=[128, 256, 512])
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        # seg_size = (input_size[0] // 2, input_size[1] // 2)
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        # scale_feature = self.spp_can(C4)
        output = self.decoder(C5)
        scale_map = self.scale_pred(output)
        # scale_aware_mask
        output = self.mlfn(C2, C3, C5)
        reg_map = self.density_pred(output)
        reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        scale_map = F.interpolate(scale_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map, scale_map


class Crowd_MSB_DCN_Scale_02(nn.Module):
    def __init__(self, dilation_ratio, backbone='vgg', pretrained=True,
                 bn=False, reduction_ratio=2):
        super(Crowd_MSB_DCN_Scale_02, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        self.decoder = MSB_DCN(512, reduction_ratio=reduction_ratio, dilation_ratio_list=dilation_ratio)
        self.spp_can = MSB_SPP(512, sizes=[1, 2, 3, 4], outchannel=512)
        self.scale_pred = nn.Sequential(
            Conv2d(512, 32, 1, 1, same_padding=True, bn=True),
            Conv2d(32, 16, 3, 1, same_padding=True, bn=True),
            Conv2d(16, 4, 1, 1, same_padding=True, bn=False, NL='None')
        )
        # self.mlfn = Decoder_MLFN(filters=[128, 256, 512])

        self.head_pred_01 = nn.Sequential(
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None'))

        self.head_pred_02 = nn.Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None'))

        self.head_pred_03 = nn.Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None'))

        self.density_pred = nn.Sequential(
            Conv2d(in_channels=3, out_channels=3, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=3, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]

        # seg_size = (input_size[0] // 2, input_size[1] // 2)
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        # scale_feature = self.spp_can(C4)
        output = self.decoder(C5)
        scale_map = self.scale_pred(output)

        fusion_size = C2.size()[2:]

        D3 = self.head_pred_03(output)
        D3 = F.interpolate(D3, size=fusion_size, mode='bilinear', align_corners=True)
        D2 = self.head_pred_02(C3)
        D2 = F.interpolate(D2, size=fusion_size, mode='bilinear', align_corners=True)
        D1 = self.head_pred_01(C2)

        # scale_aware_mask
        # output = self.mlfn(C2, C3, C5)

        reg_map = torch.cat([D1, D2, D3], dim=1)
        reg_map = self.density_pred(reg_map)
        reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        scale_map = F.interpolate(scale_map, size=input_size, mode='bilinear', align_corners=True)
        return [reg_map, D1, D2, D3], scale_map


class Crowd_MSB_DCN_Seg(nn.Module):
    def __init__(self, dilation_ratio, use_mask_feature=False, use_spp_seg=False, backbone='vgg', pretrained=True,
                 bn=False, reduction_ratio=2):
        super(Crowd_MSB_DCN_Seg, self).__init__()
        self.use_spp = use_spp_seg
        self.use_mask_feature = use_mask_feature
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        # self.decoder = Decoder01(filters=[32, 64, 128, 256, 256])
        self.decoder = MSB_DCN(512, reduction_ratio=reduction_ratio, dilation_ratio_list=dilation_ratio)
        # self.density_pred = Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False,
        #                            NL='None')
        if self.use_spp:
            self.spp_can = MSB_SPP(512, sizes=[1, 2, 3, 4], outchannel=512)
        self.seg_pred = nn.Sequential(
            Conv2d(512, 32, 1, 1, same_padding=True, bn=True),
            Conv2d(32, 16, 3, 1, same_padding=True, bn=True),
            Conv2d(16, 1, 1, 1, same_padding=True, bn=False, NL='sigmoid')
        )
        if use_mask_feature:
            self.density_pred = nn.Sequential(
                Conv2d(in_channels=1024, out_channels=128, kernel_size=3, same_padding=True, bn=True),
                Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
                Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
            )
        else:
            self.density_pred = nn.Sequential(
                Conv2d(in_channels=512, out_channels=128, kernel_size=3, same_padding=True, bn=True),
                Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
                Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
            )

    def forward(self, x):
        input_size = x.size()[2:]
        # seg_size = (input_size[0] // 2, input_size[1] // 2)
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)

        # seg_feature = self.spp_can(C4)
        # seg_feature = self.spp_can(C4)
        if self.use_spp:
            seg_feature = self.spp_can(C4)
            seg_map = self.seg_pred(seg_feature)
        else:
            seg_map = self.seg_pred(C4)

        output = self.decoder(C5)
        if 0:
            reg_map = (1 + seg_map) * output
        else:
            masked_seg_features = seg_map * output
        if self.use_mask_feature:
            reg_map = torch.cat([masked_seg_features, output], dim=1)
        else:
            reg_map = masked_seg_features
        reg_map = self.density_pred(reg_map)
        seg_map = F.interpolate(seg_map, size=input_size, mode='bilinear', align_corners=True)
        reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map, seg_map


class Crowd_MSB_UNet_Seg(nn.Module):
    def __init__(self, dilation_ratio, use_spp_seg=False, backbone='vgg', pretrained=True, bn=False, use_dcnv2=False,
                 reduction_ratio=2):
        super(Crowd_MSB_UNet_Seg, self).__init__()
        self.use_spp = use_spp_seg
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        if use_dcnv2:
            self.msb = MSB_DCNV2(512, reduction_ratio=reduction_ratio)
        else:
            self.msb = MSB_DCN(512, reduction_ratio=reduction_ratio, dilation_ratio_list=dilation_ratio)
        self.decoder = Decoder_MSB_UNet([128, 256, 512])
        if self.use_spp:
            self.spp_can = MSB_SPP(128, sizes=[1, 2, 3, 4], outchannel=128)
        self.seg_pred = nn.Sequential(
            Conv2d(128, 64, 1, 1, same_padding=True, bn=True),
            Conv2d(64, 16, 3, 1, same_padding=True, bn=True),
            Conv2d(16, 1, 1, 1, same_padding=True, bn=False, NL='sigmoid')
        )
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        C5 = self.msb(C5)
        features, _ = self.decoder(C2, C3, C5)
        if self.use_spp:
            seg_map = self.spp_can(features)
            seg_map = self.seg_pred(seg_map)
        else:
            seg_map = self.seg_pred(features)
        reg_map = features * seg_map
        reg_map = self.density_pred(reg_map)
        if cfg.density_stride == 1:
            reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
            seg_map = F.interpolate(seg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map, seg_map


class Crowd_MSB_SPP(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_MSB_SPP, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        # self.decoder = Decoder01(filters=[32, 64, 128, 256, 256])
        self.spp_can = MSB_SPP(512, sizes=[1, 2, 3, 4], outchannel=512)
        self.multi_scale_encoder = MSB_DCN(512)
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        seg_size = (input_size[0] // 2, input_size[1] // 2)
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        C5 = self.multi_scale_encoder(C5)
        output = self.spp_can(C5)
        # seg_map = self.seg_pred(output)
        # reg_map = seg_map * output
        # reg_map =
        reg_map = self.density_pred(output)
        # seg_map = F.interpolate(seg_map, size=seg_size, mode='bilinear', align_corners=True)
        if cfg.density_stride == 1:
            reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map
