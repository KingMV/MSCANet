import torch.nn as nn
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from .conv_utils import BaseConv, Conv2d, Up_Block, make_layers, MSA_Block_01, Up_MLFN


class dense_block(nn.Module):
    def __init__(self, in_channels, planes, out_channels, dilation):
        super(dense_block, self).__init__()
        self.block = nn.Sequential(Conv2d(in_channels, planes, kernel_size=1, same_padding=True, dilation=1, bn=True),
                                   Conv2d(planes, out_channels, kernel_size=3, same_padding=True, dilation=dilation,
                                          bn=True)
                                   )

    def forward(self, x):
        x = self.block(x)
        return x


class DenseAspp(nn.Module):
    def __init__(self, in_channels, planes, out_channels):
        super(DenseAspp, self).__init__()
        input_dims = in_channels
        d_feature0 = planes
        d_feature1 = planes
        # d_feature1 = planes // 2
        # d_lists = [1, 2, 3, 4]
        d_lists = [2, 3, 4, 5]
        # d_lists = [3, 6, 12, 18]
        self.assp1 = dense_block(input_dims, d_feature0, d_feature1, dilation=d_lists[0])
        self.assp2 = dense_block(input_dims + d_feature1 * 1, d_feature0, d_feature1, dilation=d_lists[1])
        self.assp3 = dense_block(input_dims + d_feature1 * 2, d_feature0, d_feature1, dilation=d_lists[2])
        self.assp4 = dense_block(input_dims + d_feature1 * 3, d_feature0, d_feature1, dilation=d_lists[3])

        self.conv_1x1 = Conv2d(input_dims, d_feature1, kernel_size=1, bn=True, same_padding=True)
        self.fuse = Conv2d(d_feature1 * 5, out_channels, kernel_size=1, stride=1, same_padding=True, bn=True)
        # self.fuse = Conv2d(d_feature1 * 4, out_channels, kernel_size=1, stride=1, same_padding=True, bn=True)

    def forward(self, x):
        input = x
        x1 = self.assp1(x)
        x = torch.cat([x, x1], dim=1)  # 512+128
        x2 = self.assp2(x)
        x = torch.cat([x, x2], dim=1)  # 512+128+128
        x3 = self.assp3(x)
        x = torch.cat([x, x3], dim=1)  # 512+128+128+128
        x4 = self.assp4(x)
        avg_input = F.adaptive_avg_pool2d(input, [1, 1])
        avg_input = avg_input.expand_as(input)
        avg_input = self.conv_1x1(avg_input)

        x = torch.cat([avg_input, x1, x2, x3, x4], dim=1)
        x = self.fuse(x)
        return x


class DenseAspp_03(nn.Module):
    def __init__(self, in_channels, planes, out_channels):
        super(DenseAspp_03, self).__init__()
        input_dims = in_channels
        d_feature0 = planes
        # d_feature1 = planes
        d_feature1 = planes // 2
        # d_lists = [1, 2, 3, 4]
        d_lists = [1, 2, 3]
        # d_lists = [3, 6, 12, 18]
        self.assp1 = dense_block(input_dims, d_feature0, d_feature1, dilation=d_lists[0])
        self.assp2 = dense_block(input_dims + d_feature1 * 1, d_feature0, d_feature1, dilation=d_lists[1])
        self.assp3 = dense_block(input_dims + d_feature1 * 2, d_feature0, d_feature1, dilation=d_lists[2])

        # self.conv_1x1 = Conv2d(input_dims, d_feature1, kernel_size=1, bn=True, same_padding=True)
        self.fuse = Conv2d(d_feature1 * 3, out_channels, kernel_size=1, stride=1, same_padding=True, bn=True)
        # self.fuse = Conv2d(d_feature1 * 4, out_channels, kernel_size=1, stride=1, same_padding=True, bn=True)

    def forward(self, x):
        input = x
        x1 = self.assp1(x)
        x = torch.cat([x, x1], dim=1)  # 512+128
        x2 = self.assp2(x)
        x = torch.cat([x, x2], dim=1)  # 512+128+128
        x3 = self.assp3(x)
        # x = torch.cat([x, x3], dim=1)  # 512+128+128+128
        # x4 = self.assp4(x)
        # avg_input = F.adaptive_avg_pool2d(input, [1, 1])
        # avg_input = avg_input.expand_as(input)
        # avg_input = self.conv_1x1(avg_input)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fuse(x)
        return x


class DenseContext(nn.Module):
    def __init__(self, in_channels, planes, out_channels, d_lists):
        super(DenseContext, self).__init__()
        input_dims = in_channels
        d_feature0 = planes
        # d_feature1 = planes
        d_feature1 = planes // 2
        # d_lists = [1, 2, 3, 4]
        # d_lists = [1, 2, 3]
        # d_lists = [3, 6, 12, 18]
        self.assp1 = dense_block(input_dims, d_feature0, d_feature1, dilation=d_lists[0])
        self.assp2 = dense_block(input_dims + d_feature1 * 1, d_feature0, d_feature1, dilation=d_lists[1])
        self.assp3 = dense_block(input_dims + d_feature1 * 2, d_feature0, d_feature1, dilation=d_lists[2])

        self.conv_1x1 = Conv2d(input_dims, d_feature1, kernel_size=1, bn=True, same_padding=True)
        # self.fuse = Conv2d(d_feature1 * 4, out_channels, kernel_size=1, stride=1, same_padding=True, bn=True)
        # self.fuse = Conv2d(d_feature1 * 3 + in_channels, out_channels, kernel_size=1, stride=1, same_padding=True,
        #                    bn=True)
        self.fuse = Conv2d(d_feature1 * 2 + in_channels, out_channels, kernel_size=1, stride=1, same_padding=True,
                           bn=True)

    def forward(self, x):
        input = x
        x1 = self.assp1(x)
        x = torch.cat([x, x1], dim=1)  # 512+128
        x2 = self.assp2(x)
        x = torch.cat([x, x2], dim=1)  # 512+128+128
        x3 = self.assp3(x)
        # x = torch.cat([x, x3], dim=1)  # 512+128+128+128
        # x4 = self.assp4(x)
        # avg_input = F.adaptive_avg_pool2d(input, [1, 1])
        # avg_input = avg_input.expand_as(input)
        # avg_input = self.conv_1x1(avg_input)

        # x = torch.cat([avg_input, input, x2, x3], dim=1)
        x = torch.cat([input, x2, x3], dim=1)
        # x = torch.cat([input, x1, x2, x3], dim=1)
        x = self.fuse(x)
        return x
