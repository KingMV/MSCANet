from .vgg_block import vgg13_model
import torch.nn.functional as F
import torch.nn as nn
import torch
from .conv_utils import Conv2d, make_layers
from .Decoder_Model import SAM, Decoder_MSB_UNet, SAM01, MSB_DCN, Decoder_UNet, D_Module
from .dense_aspp import DenseAspp, DenseAspp_03, DenseContext


class Crowd_Seg_UNet(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_Seg_UNet, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        # self.msb = MSB_DCN(512, reduction_ratio=2, dilation_ratio_list=[1, 2, 3])
        # self.L5 = DenseAspp_03(512, 256, 512)
        # self.dsam_1 = DenseAspp_03(512, 256, 512)
        # self.dsam_2 = DenseAspp_03(512, 256, 512)
        # self.dsam_3 = DenseAspp_03(512, 256, 512)
        # self.decoder = Decoder_MSB_UNet([128, 256, 512])
        # self.dsam = DenseContext(512, 256, 512, d_lists=[2, 4, 6])
        self.decoder_01 = Decoder_UNet([128, 256, 512])
        self.decoder_02 = Decoder_UNet([128, 256, 512])

        self.density_pred = nn.Sequential(
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, same_padding=True, bn=True)
        )

        self.seg_pred = nn.Sequential(
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='sigmoid')
        )
        self.conv_1x1 = Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)

        # C5 = self.back_encoder(C4)

        DL2, DL3 = self.decoder_01(C2, C3, C5)
        SL2, SL3 = self.decoder_02(C2, C3, C5)
        # L2, L3 = self.decoder(C2, C3, C5)

        seg_map = self.seg_pred(SL2)
        density_map = self.density_pred(DL2)

        density_map = seg_map * density_map
        # L2 = L2 * seg_map
        density_map = self.conv_1x1(density_map)
        # reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        # seg_map = F.interpolate(seg_map, size=input_size, mode='bilinear', align_corners=True)
        return density_map, seg_map


class Crowd_MSPNet(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_MSPNet, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        # self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)

        self.dme_1 = D_Module(512, is_dilation=False, bilinear=False, is_out=False)
        self.dme_2 = D_Module(256, is_dilation=True, bilinear=False, is_out=False)
        self.dme_3 = D_Module(128, is_dilation=True, bilinear=False, is_out=False)

        self.ame_1 = D_Module(512, is_dilation=False, bilinear=False, NL='sigmoid', is_out=True)
        self.ame_2 = D_Module(256, is_dilation=True, bilinear=False, NL='sigmoid', is_out=True)
        self.ame_3 = D_Module(128, is_dilation=True, bilinear=False, NL='sigmoid', is_out=True)

        # self.dme_1_1x1 = Conv2d(256, 1, 1, 1, NL='None', same_padding=True, bn=False)
        # self.dme_2_1x1 = Conv2d(128, 1, 1, 1, NL='None', same_padding=True, bn=False)
        # self.dme_3_1x1 = Conv2d(64, 1, 1, 1, NL='None', same_padding=True, bn=False)

        # self.density_pred = nn.Sequential(
        #     Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
        #     Conv2d(in_channels=64, out_channels=32, kernel_size=3, same_padding=True, bn=True)
        # )

        # self.seg_pred = nn.Sequential(
        #     Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
        #     Conv2d(in_channels=64, out_channels=32, kernel_size=3, same_padding=True, bn=True),
        #     Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='sigmoid')
        # )
        self.conv1_1x1 = Conv2d(in_channels=256, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        self.conv2_1x1 = Conv2d(in_channels=128, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        self.conv3_1x1 = Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        # C5 = self.back_encoder(C4)
        # print(input_size)
        # C5 = self.back_encoder(C4)
        y, D1 = self.dme_1(C4)
        # print(y.size())
        y, D2 = self.dme_2(y)
        # print(y.size())
        y, D3 = self.dme_3(y)
        # print(y.size())
        s, A1 = self.ame_1(C4)
        s, A2 = self.ame_2(s)
        s, A3 = self.ame_3(s)

        D1 = D1*A1
        D1 = self.conv1_1x1(D1)

        D2 = D2*A2
        D2 = self.conv2_1x1(D2)

        D3 = D3*A3
        D3 = self.conv3_1x1(D3)

        # L2 = L2 * seg_map
        # reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        # seg_map = F.interpolate(seg_map, size=input_size, mode='bilinear', align_corners=True)
        seg_maps = [A1, A2, A3]
        reg_maps = [D1, D2, D3]
        return reg_maps, seg_maps


class Crowd_Seg_Decoder(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_Seg_Decoder, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        # self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        # self.msb = MSB_DCN(512, reduction_ratio=2, dilation_ratio_list=[1, 2, 3])
        # self.dsam = DenseContext(512, 256, 512, d_lists=[2, 4, 6])

        self.dsam_1 = DenseContext(512, 256, 512, d_lists=[2, 4, 6])
        self.dsam_2 = DenseContext(512, 256, 512, d_lists=[2, 4, 6])
        self.dsam_3 = DenseContext(512, 256, 512, d_lists=[2, 4, 6])
        self.decoder = Decoder_UNet([128, 256, 512])

        self.density_pred = nn.Sequential(
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

        self.seg_pred_1 = nn.Sequential(
            Conv2d(in_channels=512, out_channels=32, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='sigmoid')
        )

        self.seg_pred_2 = nn.Sequential(
            Conv2d(in_channels=256, out_channels=32, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='sigmoid')
        )
        self.seg_pred_3 = nn.Sequential(
            Conv2d(in_channels=128, out_channels=32, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='sigmoid')
        )

    def forward(self, x):
        dense_skip = True
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        # C5 = self.back_encoder(C4)

        # C5 = self.dsam(C4)

        ######DDSAM
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

        seg_map_1 = self.seg_pred_1(out)

        L2, L3 = self.decoder(C2, C3, out)
        seg_map_2 = self.seg_pred_2(L3)

        seg_map_3 = self.seg_pred_3(L2)
        L2 = L2 * seg_map_3
        reg_map = self.density_pred(L2)
        # reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        # seg_map = F.interpolate(seg_map, size=input_size, mode='bilinear', align_corners=True)
        seg_map = [seg_map_3, seg_map_2, seg_map_1]
        return reg_map, seg_map


# class Crowd_Seg(nn.Module):
#     def __init__(self, backbone='vgg', pretrained=True, bn=False):
#         super(Crowd_Seg, self).__init__()
#         if backbone == 'vgg':
#             self.feature_c4 = vgg13_model(pretrained)
#         self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
#         # self.msb = MSB_DCN(512, reduction_ratio=2, dilation_ratio_list=[1, 2, 3])
#         # self.L5 = DenseAspp_03(512, 256, 512)
#         # self.dsam_1 = DenseAspp_03(512, 256, 512)
#         # self.dsam_2 = DenseAspp_03(512, 256, 512)
#         # self.dsam_3 = DenseAspp_03(512, 256, 512)
#         # self.decoder = Decoder_MSB_UNet([128, 256, 512])
#         # self.decoder = Decoder_UNet([128, 256, 512])
#
#         self.density_pred = nn.Sequential(
#             Conv2d(in_channels=512, out_channels=256, kernel_size=3, same_padding=True, bn=bn),
#             Conv2d(in_channels=256, out_channels=128, kernel_size=3, same_padding=True, bn=bn),
#             Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=bn)
#         )
#         self.conv1x1 = Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
#
#         self.seg_pred = nn.Sequential(
#             Conv2d(in_channels=512, out_channels=256, kernel_size=3, same_padding=True, bn=bn),
#             Conv2d(in_channels=256, out_channels=128, kernel_size=3, same_padding=True, bn=bn),
#             Conv2d(in_channels=128, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='sigmoid')
#         )
#
#     def forward(self, x):
#         input_size = x.size()[2:]
#         C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
#         C5 = self.back_encoder(C4)
#
#         # C5 = self.msb(C5)
#
#         # L2, L3 = self.decoder(C2, C3, C5)
#
#         seg_map = self.seg_pred(C5)
#
#         reg_map = self.density_pred(C5)
#         reg_map = seg_map * reg_map
#         reg_map = self.conv1x1(reg_map)
#         # reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
#         # seg_map = F.interpolate(seg_map, size=input_size, mode='bilinear', align_corners=True)
#         return reg_map, seg_map


class Crowd_Seg(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_Seg, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        # self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        # self.msb = MSB_DCN(512, reduction_ratio=2, dilation_ratio_list=[1, 2, 3])
        # self.L5 = DenseAspp_03(512, 256, 512)
        self.dsam = DenseContext(512, 256, 512, d_lists=[2, 4, 6])
        # self.dsam_2 = DenseAspp_03(512, 256, 512)
        # self.dsam_3 = DenseAspp_03(512, 256, 512)
        # self.decoder = Decoder_MSB_UNet([128, 256, 512])
        # self.decoder = Decoder_UNet([128, 256, 512])

        self.density_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=256, kernel_size=3, same_padding=True, bn=bn),
            Conv2d(in_channels=256, out_channels=128, kernel_size=3, same_padding=True, bn=bn),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=bn)
        )
        self.conv1x1 = Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')

        self.seg_pred = nn.Sequential(
            Conv2d(in_channels=512, out_channels=256, kernel_size=3, same_padding=True, bn=bn),
            Conv2d(in_channels=256, out_channels=128, kernel_size=3, same_padding=True, bn=bn),
            Conv2d(in_channels=128, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='sigmoid')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        # C5 = self.back_encoder(C4)

        # C5 = self.msb(C5)

        # L2, L3 = self.decoder(C2, C3, C5)
        C5 = self.dsam(C4)

        seg_map = self.seg_pred(C5)

        reg_map = self.density_pred(C5)
        reg_map = seg_map * reg_map
        reg_map = self.conv1x1(reg_map)
        # reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        # seg_map = F.interpolate(seg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map, seg_map

# class Crowd_Seg_UNet(nn.Module):
#     def __init__(self, backbone='vgg', pretrained=True, bn=False):
#         super(Crowd_Seg_UNet, self).__init__()
#         if backbone == 'vgg':
#             self.feature_c4 = vgg13_model(pretrained)
#         self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
#
#         # self.L5 = DenseAspp_03(512, 256, 512)
#         self.dsam_1 = DenseAspp_03(512, 256, 512)
#         self.dsam_2 = DenseAspp_03(512, 256, 512)
#         self.dsam_3 = DenseAspp_03(512, 256, 512)
#         self.decoder = Decoder_MSB_UNet([128, 256, 512])
#
#         self.density_pred = nn.Sequential(
#             Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
#             Conv2d(in_channels=64, out_channels=32, kernel_size=3, same_padding=True, bn=True),
#             Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
#         )
#
#         self.seg_pred = nn.Sequential(
#             Conv2d(in_channels=128, out_channels=32, kernel_size=3, same_padding=True, bn=True),
#             Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='sigmoid')
#         )
#
#     def forward(self, x):
#         input_size = x.size()[2:]
#         C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
#         C5 = self.back_encoder(C4)
#         dense_skip = False
#         # DDSAM
#         x = C5
#         x1_raw = self.dsam_1(C5)
#         if dense_skip:
#             x1 = x + x1_raw
#         else:
#             x1 = x1_raw
#         x2_raw = self.dsam_2(x1)
#         if dense_skip:
#             x2 = x1 + x2_raw
#         else:
#             x2 = x2_raw
#         out = x2
#         ##############
#
#         C5 = out
#
#         L2, L3 = self.decoder(C2, C3, C5)
#
#         seg_map = self.seg_pred(L2)
#         L2 = L2 * (1 + seg_map)
#         reg_map = self.density_pred(L2)
#         # reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
#         # seg_map = F.interpolate(seg_map, size=input_size, mode='bilinear', align_corners=True)
#         return reg_map, seg_map
