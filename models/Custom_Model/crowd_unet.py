from .vgg_block import vgg13_model
import torch.nn.functional as F
import torch.nn as nn
from .conv_utils import Conv2d, make_layers
from .Decoder_Model import SAM, Decoder_MSB_UNet, SAM01,Decoder_UNet
from .dense_aspp import DenseAspp, DenseAspp_03,DenseContext


class Crowd_UNet(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_UNet, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        # self.back_encoder = DenseContext(512, 256, 512, d_lists=[2, 4, 6])
        self.decoder = Decoder_UNet([128, 256, 512])
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        # print(x.size())
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)

        features, _ = self.decoder(C2, C3, C5)
        # print(features.size())
        reg_map = self.density_pred(features)
        # reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_SAM_UNet(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_SAM_UNet, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        # self.sam_L5 = SAM(inchannel=512, dilation_ratio_list=[1, 2, 3, 4], branch_number=3, planes_channel=256)
        # self.sam_L5 = SAM01(inchannel=512, branch_number=3, planes_channel=256, add_res=False)
        self.sam_L5 = DenseAspp(512, 256, 512)
        # self.sam_L3 = SAM01(inchannel=256, branch_number=2, planes_channel=128, add_res=False)
        # self.sam_L2 = SAM01(inchannel=128, branch_number=2, planes_channel=64, add_res=False)
        self.decoder = Decoder_UNet([128, 256, 512])
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)

        # C2 = self.sam_L2(C2)
        C5 = self.sam_L5(C5)
        # C3 = self.sam_L3(C3)

        features, _ = self.decoder(C2, C3, C5)
        reg_map = self.density_pred(features)
        reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_DSAM_UNet(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_DSAM_UNet, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        # self.sam_L5 = SAM(inchannel=512, dilation_ratio_list=[1, 2, 3, 4], branch_number=3, planes_channel=256)
        # self.sam_L5 = SAM01(inchannel=512, branch_number=3, planes_channel=256, add_res=False)
        # self.sam_L5 = DenseAspp(512, 256, 512)
        self.sam_L5 = DenseAspp_03(512, 256, 512)
        # self.sam_L3 = SAM01(inchannel=256, branch_number=2, planes_channel=128, add_res=False)
        # self.sam_L2 = SAM01(inchannel=128, branch_number=2, planes_channel=64, add_res=False)
        self.decoder = Decoder_UNet([128, 256, 512])
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)

        # C2 = self.sam_L2(C2)
        C5 = self.sam_L5(C5)
        # C3 = self.sam_L3(C3)

        features, _ = self.decoder(C2, C3, C5)
        reg_map = self.density_pred(features)
        reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map


class Crowd_DDSAM_UNet(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Crowd_DDSAM_UNet, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        # self.sam_L5 = SAM(inchannel=512, dilation_ratio_list=[1, 2, 3, 4], branch_number=3, planes_channel=256)
        # self.sam_L5 = SAM01(inchannel=512, branch_number=3, planes_channel=256, add_res=False)
        # self.sam_L5 = DenseAspp(512, 256, 512)
        # self.sam_L5 = DenseAspp_03(512, 256, 512)
        self.dsam_1 = DenseAspp_03(512, 256, 512)
        self.dsam_2 = DenseAspp_03(512, 256, 512)
        self.dsam_3 = DenseAspp_03(512, 256, 512)
        # self.sam_L3 = SAM01(inchannel=256, branch_number=2, planes_channel=128, add_res=False)
        # self.sam_L2 = SAM01(inchannel=128, branch_number=2, planes_channel=64, add_res=False)
        self.decoder = Decoder_UNet([128, 256, 512])
        self.density_pred = nn.Sequential(
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, same_padding=True, bn=True),
            Conv2d(in_channels=32, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        )

    def forward(self, x):
        input_size = x.size()[2:]
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        dense_skip = False

        x = C5
        x1_raw = self.dsam_1(C5)
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
            x3 = x2 + x3_raw
        else:
            x3 = x3_raw
        C5 = x3
        # C2 = self.sam_L2(C2)
        # C5 = self.sam_L5(C5)
        # C3 = self.sam_L3(C3)
        features, _ = self.decoder(C2, C3, C5)
        reg_map = self.density_pred(features)
        # reg_map = F.interpolate(reg_map, size=input_size, mode='bilinear', align_corners=True)
        return reg_map
