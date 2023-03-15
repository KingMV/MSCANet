from .vgg_block import vgg13_model, vgg16_model, VGG16_NoBN
import torch.nn as nn
import torch.nn.functional as F
import torch
from config import cfg
from .conv_utils import Conv2d, make_layers
from .Decoder_Model import SAM, SX, SAM01, SE, PSM, PSM01, MCN, SiT
from .dense_aspp import DenseAspp, DenseAspp_03, DenseContext
from .pyconv import PyConvBlock


class Crowd_CSRNet(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_CSRNet, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        # self.se = SE(512, 32, add_sa=True)
        # self.ssb = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)

        # self.back_encoder = PyConvBlock(512, 128, 1, pyconv_kernels=[3, 5, 7, 9], pyconv_groups=[1, 4, 8, 16])
        # blocks = 2
        # self.inplanes = 512
        # self.back_encoder = self._make_layer(PyConvBlock, 256, blocks, pyconv_kernels=[3, 5, 7, 9],
        #                                      pyconv_groups=[1, 4, 8, 16])
        # self.density_pred = nn.Sequential(
        #     Conv2d(in_channels=512, out_channels=256, kernel_size=3, same_padding=True, dilation=1, bn=bn),
        #     Conv2d(in_channels=256, out_channels=128, kernel_size=3, same_padding=True, dilation=1, bn=bn),
        #     Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, dilation=1, bn=bn),
        #     Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        # )
        # self.density_pred = nn.Sequential(
        #     Conv2d(in_channels=512, out_channels=64, kernel_size=3, same_padding=True, dilation=1, bn=bn),
        #     Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        # )

        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=256, kernel_size=3, same_padding=True, dilation=2, bn=bn),
            Conv2d(in_channels=256, out_channels=128, kernel_size=3, same_padding=True, dilation=2, bn=bn),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, dilation=2, bn=bn),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        # x = self.se(C5)
        # x = self.ssb(C5)
        reg_map = self.density_pred(C5)
        # if cfg.density_stride == 1:
        #     reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None, pyconv_kernels=[3], pyconv_groups=[1]):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, norm_layer=norm_layer,
                            pyconv_kernels=pyconv_kernels, pyconv_groups=pyconv_groups))
        # self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer,
                                pyconv_kernels=pyconv_kernels, pyconv_groups=pyconv_groups))

        return nn.Sequential(*layers)


class Crowd_MCNN(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=True):
        super(Crowd_MCNN, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.psm_1 = MCN(inchannel=512, dilation_ratio_list=[1, 2, 3, 4], branch_number=4, red_ratio=2)

        # self.psm_1 = PSM01(inchannel=512, dilation_ratio_list=[1, 2, 3, 4], branch_number=4, red_ratio=2)
        # self.psm_2 = PSM01(inchannel=512, dilation_ratio_list=[1, 2, 3, 4], branch_number=4, red_ratio=2)
        # self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)

        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=256, kernel_size=3, same_padding=True, dilation=1, bn=bn),
            Conv2d(in_channels=256, out_channels=128, kernel_size=3, same_padding=True, dilation=1, bn=bn),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, dilation=1, bn=bn),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None'),
        )

    def forward(self, x):
        input_size = x.size()[2:]
        dense_skip = False
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        out, fmps = self.psm_1(C4)
        reg_map = self.density_pred(out)
        return reg_map, fmps


class Crowd_PSNet(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=True):
        super(Crowd_PSNet, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        # self.sam_1 = SAM(inchannel=512, dilation_ratio_list=[1, 2], branch_number=2, planes_channel=256)
        # self.psm_1 = PSM(inchannel=512, dilation_ratio_list=[1, 2, 3, 4], branch_number=4, red_ratio=2)
        self.sit_1 = SiT(inchannel=512, group=5, group_channels=64, dilation_ratio_list=[1, 2, 3, 4])
        # self.psm_2 = PSM(inchannel=512, dilation_ratio_list=[1, 2, 3, 4], branch_number=4)
        # self.psm_3 = PSM(inchannel=512, dilation_ratio_list=[1, 2, 3, 4], branch_number=4)
        # self.psm_4 = PSM(inchannel=512, dilation_ratio_list=[1, 2, 3, 4], branch_number=4)

        # self.psm_1 = PSM01(inchannel=512, dilation_ratio_list=[1, 2, 3, 4], branch_number=4, red_ratio=2)

        # self.psm_2 = PSM01(inchannel=512, dilation_ratio_list=[1, 2, 3, 4], branch_number=4, red_ratio=2)
        # self.psm_3 = PSM01(inchannel=512, dilation_ratio_list=[1, 2, 3, 4], branch_number=4, red_ratio=2)
        # self.psm_4 = PSM01(inchannel=512, dilation_ratio_list=[1, 2, 3, 4], branch_number=4)
        # self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        # self.se1 = SE(512, 32, add_sa=True)
        # self.se2 = SE(512, 32, add_sa=True)

        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=256, kernel_size=3, same_padding=True, dilation=1, bn=bn),
            Conv2d(in_channels=256, out_channels=128, kernel_size=3, same_padding=True, dilation=1, bn=bn),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, dilation=1, bn=bn),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None'),
        )

        # self.density_pred = nn.Sequential(
        #     Conv2d(in_channels=512, out_channels=256, kernel_size=3, same_padding=True, bn=bn),
        #     Conv2d(in_channels=256, out_channels=128, kernel_size=3, same_padding=True, bn=bn),
        #     Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=bn),
        #     Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        # )

    def forward(self, x, train=True):
        input_size = x.size()[2:]
        dense_skip = False
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        # C5 = self.back_encoder(C4)
        # C5 = self.se1(C5)
        # x = self.sam_1(C5)
        # x = self.sam_2(x)
        # x = self.sam_3(x)

        x = C4
        x1_raw = self.sit_1(x, train)
        # x1_raw = self.psm_1(C5)
        if dense_skip:
            x1 = x + x1_raw
        else:
            x1 = x1_raw
        # x2_raw = self.psm_2(x1)
        # if dense_skip:
        #     x2 = x1 + x2_raw
        # else:
        #     x2 = x2_raw
        # x3_raw = self.psm_3(x2)
        #
        # if dense_skip:
        #     x3 = x1_raw + x2_raw + x3_raw + x
        # else:
        #     x3 = x3_raw

        # x4_raw = self.psm_4(x3)
        #
        # if dense_skip:
        #     x4 = x1_raw + x2_raw + x3_raw + x4_raw + x
        # else:
        #     x4 = x4_raw
        # out = self.se1(x4)
        reg_map = self.density_pred(x1)
        # if cfg.density_stride == 1:
        #     reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_SAM(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=True):
        super(Crowd_SAM, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        # self.sam_1 = SAM(inchannel=512, dilation_ratio_list=[1, 2], branch_number=2, planes_channel=256)
        self.sam_1 = SAM01(inchannel=512, outchannel=512, dilation_ratio_list=[1, 2, 3], branch_number=3,
                           planes_channel=256, add_res=False)
        self.sam_2 = SAM01(inchannel=512, outchannel=512, dilation_ratio_list=[1, 2, 3], branch_number=3,
                           planes_channel=256, add_res=False)
        self.sam_3 = SAM01(inchannel=512, outchannel=512, dilation_ratio_list=[1, 2, 3], branch_number=3,
                           planes_channel=256, add_res=False)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        # self.se1 = SE(512, 32, add_sa=True)
        # self.se2 = SE(512, 32, add_sa=True)

        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=64, kernel_size=3, same_padding=True, dilation=1, bn=bn),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

        # self.density_pred = nn.Sequential(
        #     Conv2d(in_channels=512, out_channels=256, kernel_size=3, same_padding=True, bn=bn),
        #     Conv2d(in_channels=256, out_channels=128, kernel_size=3, same_padding=True, bn=bn),
        #     Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=bn),
        #     Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        # )

    def forward(self, x):
        input_size = x.size()[2:]
        dense_skip = True
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        # C5 = self.se1(C5)
        # x = self.sam_1(C5)
        # x = self.sam_2(x)
        # x = self.sam_3(x)

        x = C5
        x1_raw = self.sam_1(C5)
        if dense_skip:
            x1 = x + x1_raw
        else:
            x1 = x1_raw
        x2_raw = self.sam_2(x1)
        if dense_skip:
            x2 = x1 + x2_raw
        else:
            x2 = x2_raw
        x3_raw = self.sam_3(x2)

        if dense_skip:
            out = x1_raw + x2_raw + x3_raw + x
        else:
            out = x3_raw
        # out = self.se2(out)
        reg_map = self.density_pred(out)
        # if cfg.density_stride == 1:
        #     reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_trident(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=True):
        super(Crowd_trident, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.sam_1 = SAM(inchannel=512, dilation_ratio_list=[2, 4, 8, 12], branch_number=4, planes_channel=256)
        # self.scale_block = SX(inchannel=512, dilation_ratio_list=[1, 2, 3], planes_channel=256)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=256, kernel_size=3, same_padding=True, bn=bn),
            Conv2d(in_channels=256, out_channels=128, kernel_size=3, same_padding=True, bn=bn),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=bn),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        x = self.sam_1(C5)
        reg_map = self.density_pred(x)
        # if cfg.density_stride == 1:
        #     reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_DSAM(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=True):
        super(Crowd_DSAM, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        # self.sam_1 = SAM(inchannel=512, dilation_ratio_list=[1, 2, 3, 4], branch_number=3, planes_channel=256)
        # self.dsam_1 = DenseAspp(512, 256, 512)
        # self.dsam_1 = DenseAspp_03(512, 256, 512)

        self.dsam_1 = DenseContext(512, 256, 512, d_lists=[2, 4, 6])
        # self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=256, kernel_size=3, same_padding=True, bn=bn),
            Conv2d(in_channels=256, out_channels=128, kernel_size=3, same_padding=True, bn=bn),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=bn),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        # C5 = self.back_encoder(C4)
        x = self.dsam_1(C4)
        reg_map = self.density_pred(x)
        # if cfg.density_stride == 1:
        #     reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_DDSAM(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=True):
        super(Crowd_DDSAM, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        # self.sam_1 = SAM(inchannel=512, dilation_ratio_list=[1, 2, 3, 4], branch_number=3, planes_channel=256)
        # self.dsam_1 = DenseAspp_03(512, 256, 512)
        # self.dsam_2 = DenseAspp_03(512, 256, 512)
        # self.dsam_3 = DenseAspp_03(512, 256, 512)

        self.dsam_1 = DenseContext(512, 256, 512, d_lists=[2, 4, 6])
        self.dsam_2 = DenseContext(512, 256, 512, d_lists=[2, 4, 6])
        self.dsam_3 = DenseContext(512, 256, 512, d_lists=[2, 4, 6])
        # self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=256, kernel_size=3, same_padding=True, bn=bn),
            Conv2d(in_channels=256, out_channels=128, kernel_size=3, same_padding=True, bn=bn),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=bn),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )
        # self.conv_1x1 = Conv2d(in_channels=512, out_channels=512, kernel_size=1, same_padding=True, bn=True)

    def forward(self, x):
        dense_skip = True
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        # C5 = self.back_encoder(C4)

        x = C4
        x1_raw = self.dsam_1(C4)
        if dense_skip:
            x1 = x + x1_raw
        else:
            x1 = x1_raw
        x2_raw = self.dsam_2(x1)
        if dense_skip:
            x2 = x1 + x2_raw
        else:
            x2 = x2_raw
        x3_raw = self.dsam_3(x2)

        if dense_skip:
            out = x1_raw + x2_raw + x3_raw + x
        else:
            out = x3_raw
        # # x3 = x1_raw + x2_raw + x3_raw + x
        # # out = x2
        # out = x3_raw
        # out = self.conv_1x1(out)

        reg_map = self.density_pred(out)

        # if cfg.density_stride == 1:
        #     reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map
