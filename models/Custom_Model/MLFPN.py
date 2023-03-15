import torch
import torch.nn as nn
from .conv_utils import Conv2d
import torch.nn.functional as F

from .conv_utils import Up_Block


class MLFPN(nn.Module):
    def __init__(self, input_size, planes, backbone_type='vgg', smooth=True, num_levels=3, num_scales=6, sfam=True):
        super(MLFPN, self).__init__()
        self.input_size = input_size
        self.smooth = smooth
        self.num_levels = num_levels
        self.num_scales = num_scales
        self.backbone_type = backbone_type
        self.planes = planes
        self.sfam = sfam

        # if self.backbone_type == 'vgg':
        shallow_in, shallow_out = 512, 256
        deep_in, deep_out = 512, 256
        self.reduce_shallow = Conv2d(shallow_in, shallow_out, 3, 1, same_padding=True, bn=True)
        self.reduce_deep = Conv2d(deep_in, deep_out, 3, 1, same_padding=True, bn=True)

        # Build tum
        tums = []
        for i in range(self.num_levels):
            if i == 0:
                # input:(b,256,H,W)--->(b,256,H,W)
                tums.append(Tum(first_level=True, input_planes=self.planes // 2, is_smooth=self.smooth,
                                scales=self.num_scales, side_channel=self.planes))
            else:
                tums.append(Tum(first_level=False, input_planes=self.planes // 2, is_smooth=self.smooth,
                                scales=self.num_scales, side_channel=self.planes))
        self.tums = nn.ModuleList(tums)

        # Build FFM2
        self.FFM1_Reduction = Conv2d(shallow_out + deep_out, self.planes, 3, 1, same_padding=True, bn=True)
        self.leach = nn.ModuleList(
            [Conv2d(shallow_out + deep_out, self.planes // 2, 1, 1, same_padding=True, bn=True)] * (
                    self.num_levels - 1))
        if self.sfam:
            self.sfam_module = SFAM()

    def forward(self, x):
        x_shallow = self.reduce_shallow(x[0])
        x_deep = self.reduce_deep(x[1])
        # 512,H,W
        base_feature = torch.cat([x_shallow, F.interpolate(x_deep, scale_factor=2, mode='nearest')], 1)
        print('base_feature size is {0}'.format(base_feature.size()))
        tum_outs = [self.tums[0](self.FFM1_Reduction(base_feature), None)]
        print('the size of tum[0] is {0}'.format(tum_outs[0][-1].size()))
        for i in range(1, self.num_levels, 1):
            tum_outs.append(self.tums[i](self.leach[i - 1](base_feature), tum_outs[i - 1][-1]))

        sources = []
        for i in range(self.num_scales, 0, -1):
            sources.append(torch.cat([tum_out[i - 1] for tum_out in tum_outs], 1))

        # if self.sfam:
        #     sources = self.sfam_module(sources)

        return sources


class MSN(nn.Module):
    def __init__(self, input_planes=256, scales=6, is_smooth=True, output_planes=64):
        super(MSN, self).__init__()
        self.input_planes = input_planes
        self.scales = scales
        self.is_smooth = is_smooth
        self.planes = input_planes
        self.output_planes = output_planes

        self.encoder_layers = nn.Sequential()
        self.decoder_layers = nn.Sequential()

        for i in range(self.scales - 1):
            self.encoder_layers.add_module('{}'.format(len(self.encoder_layers)),
                                           Conv2d(self.planes, self.planes, 3, 2, same_padding=True, bn=True))
        for i in range(self.scales - 1):
            self.decoder_layers.add_module('{}'.format(len(self.decoder_layers)),
                                           Conv2d(self.planes, self.planes, 3, 1, same_padding=True, bn=True))

        if self.is_smooth:
            smooth = []
            for i in range(self.scales):
                if self.output_planes > 0:
                    smooth.append(Conv2d(self.planes, self.output_planes, 1, 1, bn=True))
                else:
                    smooth.append(Conv2d(self.planes, self.planes // 2, 1, 1, bn=True))
            self.smooth = nn.Sequential(*smooth)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='nearest') + y

    def forward(self, x):
        conv_feat = [x]
        for i in range(len(self.encoder_layers)):
            x = self.encoder_layers[i](x)
            conv_feat.append(x)
        deconved_feat = [conv_feat[-1]]
        for i in range(len(self.decoder_layers)):
            deconved_feat.append(self._upsample_add(self.decoder_layers[i](deconved_feat[i]), conv_feat[- 2 - i]))
        if self.is_smooth:
            smoothed_feat = [self.smooth[0](deconved_feat[0])]
            for i in range(1, len(self.smooth), 1):
                smoothed_feat.append(self.smooth[i](deconved_feat[i]))
            return smoothed_feat
        return deconved_feat


class MSDecoder(nn.Module):
    def __init__(self, filters, res=False, h_out=False):
        '''
        :param filters:  [32,64,128,256,256]
        '''
        super(MSDecoder, self).__init__()
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


class Tum(nn.Module):
    def __init__(self, first_level=True, input_planes=128, side_channel=128, scales=6, is_smooth=True,
                 sfam=False):
        super(Tum, self).__init__()
        self.planes = input_planes * 2  # 256
        self.first_level = first_level
        self.input_channles = input_planes + side_channel
        self.layers = nn.Sequential()
        self.scales = scales
        self.is_smooth = is_smooth
        self.sfam = sfam

        # FM reduction
        # self.layers.add_module('{}'.format(len(self.layers)),
        #                        Conv2d(self.input_channles, self.planes, 3, 2, same_padding=True, bn=True))
        for i in range(self.scales - 1):
            self.layers.add_module('{}'.format(len(self.layers)),
                                   Conv2d(self.planes, self.planes, 3, 2, same_padding=True, bn=True))
            # if i != self.scales - 2:
            #     self.layers.add_module('{}'.format(len(self.layers)),
            #                            Conv2d(self.planes, self.planes, 3, 2, same_padding=True, bn=True))
            # else:
            #     self.layers.add_module('{}'.format(len(self.layers)),
            #                            Conv2d(self.planes, self.planes, 3, 1, same_padding=True, bn=True))

        # self.toplayer = nn.Sequential(Conv2d(self.planes, self.planes, 1, 1, same_padding=True, bn=True))
        self.latlayer = nn.Sequential()
        for i in range(self.scales - 1):
            self.latlayer.add_module('{}'.format(len(self.latlayer)),
                                     Conv2d(self.planes, self.planes, 3, 1, same_padding=True, bn=True))

        # build sfam:
        if self.sfam:
            self.sfam_module = SFAM(self.planes,
                                    self.num_levels,
                                    self.num_scales)
        if self.is_smooth:
            smooth = []
            for i in range(self.scales):
                smooth.append(Conv2d(self.planes, self.planes // 2, 1, 1, bn=True))
            self.smooth = nn.Sequential(*smooth)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='nearest') + y

    def forward(self, x, y):
        if not self.first_level:
            x = torch.cat([x, y], dim=1)
        conv_feat = [x]
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            conv_feat.append(x)
        print('encoder size is {0}'.format(conv_feat[-1].size()))
        print('conv_feat size is {0}'.format(conv_feat.__len__()))
        # deconved_feat = [self._upsample_add(self.latlayer[0](conv_feat[-1]), conv_feat[-2])]
        deconved_feat = [conv_feat[-1]]
        for i in range(len(self.latlayer)):
            deconved_feat.append(self._upsample_add(self.latlayer[i](deconved_feat[i]), conv_feat[- 2 - i]))
        print('deconved_feat size is {0}'.format(len(deconved_feat)))
        # deconved_feat.append(self._upsample_add(dec))
        # deconved_feat = [self.toplayer[0](conv_feat[-1])]
        # for i in range(len(self.latlayer)):
        #     deconved_feat.append(
        #         self._upsample_add(deconved_feat[i], self.latlayer[i](conv_feat[len(self.layers) - 1 - i])))
        if self.is_smooth:
            smoothed_feat = [self.smooth[0](deconved_feat[0])]
            for i in range(1, len(self.smooth), 1):
                smoothed_feat.append(self.smooth[i](deconved_feat[i]))
            return smoothed_feat
        return deconved_feat


class CFFM(nn.Module):
    def __init__(self, planes, compress_ratio=16):
        super(CFFM, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.sigmoid = nn.Sigmoid()
        self.planes = planes
        self.fc = nn.Sequential(
            Conv2d(self.planes, self.planes // compress_ratio, 1, 1, same_padding=True),
            Conv2d(self.planes // compress_ratio, self.planes, 1, 1, same_padding=True, NL='sigmoid'))

    def forward(self, x):
        weight = self.avgpool(x)
        weight = self.fc(weight)
        x = weight * x
        return x


class SFAM(nn.Module):
    def __init__(self, planes=128, num_levels=3, num_scales=6, compress_ratio=16):
        super(SFAM, self).__init__()
        self.planes = planes
        self.num_levels = num_levels
        self.num_scales = num_scales
        self.compress_ratio = compress_ratio

        self.fc1 = nn.ModuleList([nn.Conv2d(self.planes * self.num_levels,
                                            self.planes * self.num_levels // 16, 1, 1, 0)] * self.num_scales)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.ModuleList([nn.Conv2d(self.planes * self.num_levels // 16,
                                            self.planes * self.num_levels, 1, 1, 0)] * self.num_scales)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        attention_feat = []
        for i, _mf in enumerate(x):
            _tmp_f = self.avgpool(_mf)
            _tmp_f = self.fc1[i](_tmp_f)
            _tmp_f = self.relu(_tmp_f)
            _tmp_f = self.fc2[i](_tmp_f)
            _tmp_f = self.sigmoid(_tmp_f)
            attention_feat.append(_mf * _tmp_f)
        return attention_feat
