from .vgg_block import vgg13_model, vgg16_model
import torch.nn as nn
import torch.nn.functional as F
import torch
from config import cfg
from .conv_utils import Conv2d, make_layers
from .Decoder_Model import Decoder01, Decoder02, UNet_Decoder, FPN_Deocder, Decoder03, Multi_Scale_Module, MSB_01, \
    MSB_02, DCM, MSB_DCN, Decoder_simple_seg, MSB_SPP, Decoder_MSB_UNet
from .MLFPN import MLFPN, MSDecoder, MSN, CFFM


class FEM(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(FEM, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        self.msb = MSB_DCN(512)

    def forward(self, x):
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        C5 = self.msb(C5)
        return C2, C3, C4, C5  # C4 is vgg feature


class Crowd_MSB_FPN(nn.Module):
    def __init__(self, backbone='vgg', bn=True):
        super(Crowd_MSB_FPN, self).__init__()
        self.backbone = backbone
        self.bn = bn
        self.multi_features = FEM(backbone=self.backbone, bn=True)
        self.decoder = Decoder_MSB_UNet(filters=[128, 256, 512])  # C2,C3,C5

        back_filters = [128, 64, 32, 1]
        self.density_pred = nn.Sequential(
            Conv2d(back_filters[0], back_filters[1], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[1], back_filters[2], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[2], back_filters[3], 1, 1, NL='None', same_padding=True, bn=False)
        )

    def forward(self, x):
        input_size = x.size()[2:]
        conv2, conv3, _, conv5 = self.multi_features(x)
        out, _ = self.decoder(conv2, conv3, conv5)
        den_map = self.density_pred(out)
        den_map = F.interpolate(den_map, size=input_size, mode='bilinear', align_corners=True)
        return den_map


class Crowd_MSB_FPN_Seg(nn.Module):
    def __init__(self, backbone='vgg', bn=True):
        super(Crowd_MSB_FPN_Seg, self).__init__()
        self.backbone = backbone
        self.bn = bn
        self.multi_features = FEM(backbone=self.backbone, bn=True)
        self.decoder = Decoder_MSB_UNet(filters=[128, 256, 512])  # C2,C3,C5
        back_filters = [128, 64, 32, 1]
        self.density_pred = nn.Sequential(
            Conv2d(back_filters[0], back_filters[1], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[1], back_filters[2], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[2], back_filters[3], 1, 1, NL='None', same_padding=True, bn=False)
        )
        self.seg_pred = nn.Sequential(
            Conv2d(128, 32, 1, 1, same_padding=True, bn=True),
            Conv2d(32, 16, 3, 1, same_padding=True, bn=True),
            Conv2d(16, 1, 1, 1, same_padding=True, bn=False, NL='sigmoid')
        )
        # self.density_pred = nn.Sequential(
        #     Conv2d(in_channels=128, out_channels=1, kernel_size=1, same_padding=True, bn=False,
        #            NL='None')
        # )

    def forward(self, x):
        den_map = None
        seg_map = None
        input_size = x.size()[2:]
        conv2, conv3, conv5 = self.multi_features(x)
        out, _ = self.decoder(conv2, conv3, conv5)
        seg_map = self.seg_pred(out)
        mask = torch.zeros_like(seg_map)
        mask[seg_map > 0.3] = 1
        # seg_map[seg_map > 0.3] = 1
        # seg_map[seg_map <= 0.3] = 0
        den_map = mask * out
        den_map = self.density_pred(den_map)
        den_map = F.interpolate(den_map, size=input_size, mode='bilinear', align_corners=True)
        seg_map = F.interpolate(seg_map, size=input_size, mode='bilinear', align_corners=True)
        return den_map, seg_map


class Crowd_MSB_SC(nn.Module):
    def __init__(self, backbone='vgg', bn=True):
        super(Crowd_MSB_SC, self).__init__()
        self.backbone = backbone
        self.bn = bn

        # multi-scale feature encoder
        self.multi_features = FEM(backbone=self.backbone, bn=True)
        # scale classification
        self.scale_classify = nn.Sequential(
            Conv2d(512, 256, 3, 1, same_padding=True, bn=True),
            Conv2d(256, 128, 3, 1, same_padding=True, bn=True),
            nn.AdaptiveAvgPool2d(1),
            Conv2d(128, 4, 1, 1, same_padding=True, bn=False, NL='None')
        )

        # self.decoder = Decoder_MSB_UNet(filters=[128, 256, 512])  # C2,C3,C5
        back_filters = [512, 256, 128, 1]
        self.normal_pred = nn.Sequential(
            Conv2d(back_filters[0], back_filters[1], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[1], back_filters[2], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[2], back_filters[3], 1, 1, NL='None', bn=False)
        )

    def forward(self, x):
        den_map = None
        input_size = x.size()[2:]
        conv2, conv3, conv4, conv5 = self.multi_features(x)
        scale_c = self.scale_classify(conv5)  # !!!!!!!!!!!!!1conv4
        scale_c = torch.flatten(scale_c, 1)
        den_map = self.normal_pred(conv5)
        den_map = F.interpolate(den_map, size=input_size, mode='bilinear')
        return den_map, scale_c


class Crowd_MSB_SC_0325(nn.Module):
    def __init__(self, backbone='vgg', bn=True):
        super(Crowd_MSB_SC_0325, self).__init__()
        self.backbone = backbone
        self.bn = bn

        # multi-scale feature encoder
        self.multi_features = FEM(backbone=self.backbone, bn=True)
        # scale classification
        self.scale_branch = nn.Sequential(
            Conv2d(512, 256, 3, 1, same_padding=True, bn=True, dilation=2),
            Conv2d(256, 128, 3, 1, same_padding=True, bn=True, dilation=2)
        )
        self.scale_classify = nn.Sequential(
            Conv2d(128, 64, 3, 1, same_padding=True, bn=True, dilation=1),
            nn.AdaptiveAvgPool2d(1),
            Conv2d(64, 4, 1, 1, same_padding=True, bn=False, NL='None')
        )

        # self.decoder = Decoder_MSB_UNet(filters=[128, 256, 512])  # C2,C3,C5
        back_filters = [512 + 128, 256, 128, 1]
        self.normal_pred = nn.Sequential(
            Conv2d(back_filters[0], back_filters[1], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[1], back_filters[2], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[2], back_filters[3], 1, 1, NL='None', bn=False)
        )

    def forward(self, x):
        den_map = None
        input_size = x.size()[2:]
        conv2, conv3, conv4, conv5 = self.multi_features(x)
        scale_feature = self.scale_branch(conv4)
        # x = torch.cat([x, x1, x2, x3], dim=1)
        conv5 = torch.cat([conv5, scale_feature], dim=1)
        scale_c = self.scale_classify(scale_feature)
        scale_c = torch.flatten(scale_c, 1)
        den_map = self.normal_pred(conv5)
        den_map = F.interpolate(den_map, size=input_size, mode='bilinear')
        return den_map, scale_c


class Crowd_MSB_Class(nn.Module):
    def __init__(self, backbone='vgg', bn=True):
        super(Crowd_MSB_Class, self).__init__()
        self.backbone = backbone
        self.bn = bn

        # multi-scale feature encoder
        self.multi_features = FEM(backbone=self.backbone, bn=True)
        # scale classification
        self.scale_classify = nn.Sequential(
            Conv2d(512, 256, 3, 1, same_padding=True, bn=True),
            Conv2d(256, 128, 3, 1, same_padding=True, bn=True),
            nn.AdaptiveAvgPool2d(1),
            Conv2d(128, 4, 1, 1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        den_map = None
        input_size = x.size()[2:]
        conv2, conv3, conv4, conv5 = self.multi_features(x)
        scale_c = self.scale_classify(conv4)
        scale_c = torch.flatten(scale_c, 1)
        return scale_c


class Crowd_MSB_SC_01(nn.Module):
    def __init__(self, backbone='vgg', bn=True):
        super(Crowd_MSB_SC_01, self).__init__()
        self.backbone = backbone
        self.bn = bn

        # multi-scale feature encoder
        self.multi_features = FEM(backbone=self.backbone, bn=True)
        self.spp_can = MSB_SPP(512, sizes=[1, 2, 3, 4], outchannel=512)
        # scale classification
        self.scale_classify = nn.Sequential(
            Conv2d(512, 256, 3, 1, same_padding=True, bn=True),
            Conv2d(256, 128, 3, 1, same_padding=True, bn=True),
            nn.AdaptiveAvgPool2d(1),
            Conv2d(128, 4, 1, 1, same_padding=True, bn=False, NL='None')
        )

        # self.decoder = Decoder_MSB_UNet(filters=[128, 256, 512])  # C2,C3,C5
        back_filters = [512, 256, 128, 1]
        self.normal_pred = nn.Sequential(
            Conv2d(back_filters[0], back_filters[1], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[1], back_filters[2], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[2], back_filters[3], 1, 1, NL='None', bn=False)
        )

    def forward(self, x):
        input_size = x.size()[2:]
        conv2, conv3, conv4, conv5 = self.multi_features(x)
        scale_c = self.scale_classify(conv4)
        scale_c = torch.flatten(scale_c, 1)
        conv5 = self.spp_can(conv5)
        den_map = self.normal_pred(conv5)
        den_map = F.interpolate(den_map, size=input_size, mode='bilinear')
        return den_map, scale_c


class Crowd_MSB_SC_02(nn.Module):
    def __init__(self, backbone='vgg', bn=True):
        super(Crowd_MSB_SC_02, self).__init__()
        self.backbone = backbone
        self.bn = bn

        # multi-scale feature encoder
        self.multi_features = FEM(backbone=self.backbone, bn=True)
        # self.spp_can = MSB_SPP(512, sizes=[1, 2, 3, 4], outchannel=512)

        # density_level classification
        self.scale_classify = nn.Sequential(
            Conv2d(512, 256, 3, 1, same_padding=True, bn=True),
            Conv2d(256, 128, 3, 1, same_padding=True, bn=True),
            nn.AdaptiveAvgPool2d(1),
            Conv2d(128, 4, 1, 1, same_padding=True, bn=False, NL='None')
        )

        # self.decoder = Decoder_MSB_UNet(filters=[128, 256, 512])  # C2,C3,C5
        back_filters = [512, 256, 128, 1]
        self.normal_pred = nn.Sequential(
            Conv2d(back_filters[0], back_filters[1], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[1], back_filters[2], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[2], back_filters[3], 1, 1, NL='None', bn=False)
        )
        self.hd_branch = Decoder_MSB_UNet(filters=[128, 256, 512])  # C2,C3,C5
        back_filters = [128, 64, 32, 1]
        self.hd_pred = nn.Sequential(
            Conv2d(back_filters[0], back_filters[1], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[1], back_filters[2], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[2], back_filters[3], 1, 1, NL='None', bn=False)
        )

    def forward(self, x):
        input_size = x.size()[2:]
        conv2, conv3, conv4, mfs = self.multi_features(x)

        density_level = self.scale_classify(conv4)
        density_level = torch.flatten(density_level, 1)

        # high density
        hd, _ = self.hd_branch(conv2, conv3, mfs)

        # hd_size = hd.size()[2:]
        # print(hd.size())
        sd = self.normal_pred(mfs)
        hd = self.hd_pred(hd)
        sd = F.interpolate(sd, size=input_size, mode='bilinear')
        hd = F.interpolate(hd, size=input_size, mode='bilinear')
        return sd, hd, density_level


class Crowd_MSB_SC_04(nn.Module):
    def __init__(self, backbone='vgg', bn=True):
        super(Crowd_MSB_SC_04, self).__init__()
        self.backbone = backbone
        self.bn = bn

        # multi-scale feature encoder
        self.multi_features = FEM(backbone=self.backbone, bn=True)
        # self.spp_can = MSB_SPP(512, sizes=[1, 2, 3, 4], outchannel=512)

        # density_level classification
        self.scale_classify = nn.Sequential(
            Conv2d(512, 256, 3, 1, same_padding=True, bn=True),
            Conv2d(256, 128, 3, 1, same_padding=True, bn=True),
            nn.AdaptiveAvgPool2d(1),
            Conv2d(128, 4, 1, 1, same_padding=True, bn=False, NL='None')
        )

        self.hd_branch = Decoder_MSB_UNet(filters=[128, 256, 512])  # C2,C3,C5
        back_filters = [128, 64, 32]
        self.hd_pred = nn.Sequential(
            Conv2d(back_filters[0], back_filters[1], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[1], back_filters[2], 3, 1, same_padding=True, bn=True)
        )
        # self.decoder = Decoder_MSB_UNet(filters=[128, 256, 512])  # C2,C3,C5
        back_filters = [512, 256, 32]
        self.normal_pred = nn.Sequential(
            Conv2d(back_filters[0], back_filters[1], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[1], back_filters[2], 3, 1, same_padding=True, bn=True)
        )

        self.final_pred = Conv2d(64, 1, 1, NL='None', bn=False)

    def forward(self, x):
        input_size = x.size()[2:]
        conv2, conv3, conv4, mfs = self.multi_features(x)

        density_level = self.scale_classify(conv4)
        density_level = torch.flatten(density_level, 1)

        # high density
        hd, _ = self.hd_branch(conv2, conv3, mfs)

        # hd_size = hd.size()[2:]
        # print(hd.size())
        sd = self.normal_pred(mfs)
        hd = self.hd_pred(hd)
        sd = F.interpolate(sd, size=input_size, mode='bilinear')
        hd = F.interpolate(hd, size=input_size, mode='bilinear')

        pred_map = torch.cat([sd, hd], dim=1)
        pred_map = self.final_pred(pred_map)
        return pred_map, density_level


class Crowd_MSB_FPN_SPP(nn.Module):
    def __init__(self, backbone='vgg', bn=True):
        super(Crowd_MSB_FPN_SPP, self).__init__()
        self.backbone = backbone
        self.bn = bn
        self.multi_features = FEM(backbone=self.backbone, bn=True)
        self.decoder = Decoder_MSB_UNet(filters=[128, 256, 512])  # C2,C3,C5
        self.spp = MSB_SPP(128, sizes=[1, 2, 3, 4])
        back_filters = [128, 64, 32, 1]
        self.density_pred = nn.Sequential(
            Conv2d(back_filters[0], back_filters[1], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[1], back_filters[2], 3, 1, same_padding=True, bn=True),
            Conv2d(back_filters[2], back_filters[3], 1, 1, NL='None', same_padding=True, bn=False)
        )
        # self.seg_pred = nn.Sequential(
        #     Conv2d(128, 32, 1, 1, same_padding=True, bn=True),
        #     Conv2d(32, 16, 3, 1, same_padding=True, bn=True),
        #     Conv2d(16, 1, 1, 1, same_padding=True, bn=False, NL='sigmoid')
        # )
        # self.density_pred = nn.Sequential(
        #     Conv2d(in_channels=128, out_channels=1, kernel_size=1, same_padding=True, bn=False,
        #            NL='None')
        # )

    def forward(self, x):
        den_map = None
        seg_map = None
        input_size = x.size()[2:]
        conv2, conv3, conv5 = self.multi_features(x)
        out, _ = self.decoder(conv2, conv3, conv5)
        out = self.spp(out)
        # seg_map = self.seg_pred(out)
        # mask = torch.zeros_like(seg_map)
        # mask[seg_map > 0.3] = 1
        # seg_map[seg_map > 0.3] = 1
        # seg_map[seg_map <= 0.3] = 0
        # den_map = mask * out
        den_map = self.density_pred(out)
        den_map = F.interpolate(den_map, size=input_size, mode='bilinear', align_corners=True)
        # seg_map = F.interpolate(seg_map, size=input_size, mode='bilinear', align_corners=True)
        return den_map
