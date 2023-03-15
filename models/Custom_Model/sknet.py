import torch
from torch import nn
from functools import reduce

# from torchvision.models.resnet import resnext50_32x4d


class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, M=2, G=32, r=16, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        self.out_channels = out_channels
        d = max(int(in_channels / r), L)
        self.M = M
        self.convs = nn.ModuleList()
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=1 + i, padding=1 + i,
                          groups=G, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True)
                                 )
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output = []
        batch_size = x.size()[0]
        for i, conv in enumerate(self.convs):
            feas = conv(x)
            output.append(feas)
        U = reduce(lambda x, y: x + y, output)
        s = self.gap(U)
        s = self.fc1(s)
        #
        a_b = self.fc2(s)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        # print(a_b.shape)
        a_b = self.softmax(a_b)
        # print(a_b)
        # print(a_b[0, 1, 0])

        a_b = list(a_b.chunk(self.M, dim=1))  # split to a and b
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, output, a_b))
        V = reduce(lambda x, y: x + y, V)
        # fea_v = (feas * attention_vectors).sum(dim=1)
        return V


class SKConv_ab(nn.Module):
    def __init__(self, in_channels, out_channels, D=1, M=1, G=32, r=16, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv_ab, self).__init__()
        self.out_channels = out_channels
        d = max(int(in_channels / r), L)
        self.M = M
        self.convs = nn.ModuleList()
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=D, padding=D, groups=G,
                          bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True)
                                 )
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output = []
        batch_size = x.size()[0]
        for i, conv in enumerate(self.convs):
            feas = conv(x)
            output.append(feas)
        U = reduce(lambda x, y: x + y, output)
        s = self.gap(U)
        s = self.fc1(s)
        #
        a_b = self.fc2(s)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        # print(a_b.shape)
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(self.M, dim=1))  # split to a and b
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, output, a_b))
        V = reduce(lambda x, y: x + y, V)
        # fea_v = (feas * attention_vectors).sum(dim=1)
        return V


class SKUnit_ab(nn.Module):
    def __init__(self, inplanes, D=1, M=1, G=32, stride=1, L=32):
        """ Constructor
        Args:
            inplanes: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit_ab, self).__init__()
        expansion = 2
        planes = inplanes // expansion
        self.feas = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride=1, bias=False),  # 1*1 reduction dim
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            SKConv_ab(in_channels=planes, out_channels=planes, D=D, stride=stride, L=L, M=M, G=G),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes * expansion, 1, stride=1, bias=False),
            nn.BatchNorm2d(planes * expansion)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        fea = self.feas(x)
        fea += shortcut
        return self.relu(fea)


class SKUnit(nn.Module):
    def __init__(self, inplanes, M=2, G=32, stride=1, L=32):
        """ Constructor
        Args:
            inplanes: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        expansion = 2
        planes = inplanes // expansion
        self.feas = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride=1, bias=False),  # 1*1 reduction dim
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            SKConv(in_channels=planes, out_channels=planes, stride=stride, L=L, M=M, G=G),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes * expansion, 1, stride=1, bias=False),
            nn.BatchNorm2d(planes * expansion)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        fea = self.feas(x)
        fea += shortcut
        return self.relu(fea)


# class SKNet(nn.Module):
#     def __init__(self, class_num):
#         super(SKNet, self).__init__()
#         self.basic_conv = nn.Sequential(
#             nn.Conv2d(3, 64, 3, padding=1),
#             nn.BatchNorm2d(64)
#         )  # 32x32
#         self.stage_1 = nn.Sequential(
#             SKUnit(64, 256, 32, 2, 8, 2, stride=2),
#             nn.ReLU(),
#             SKUnit(256, 256, 32, 2, 8, 2),
#             nn.ReLU(),
#             SKUnit(256, 256, 32, 2, 8, 2),
#             nn.ReLU()
#         )  # 32x32
#         self.stage_2 = nn.Sequential(
#             SKUnit(256, 512, 32, 2, 8, 2, stride=2),
#             nn.ReLU(),
#             SKUnit(512, 512, 32, 2, 8, 2),
#             nn.ReLU(),
#             SKUnit(512, 512, 32, 2, 8, 2),
#             nn.ReLU()
#         )  # 16x16
#         self.stage_3 = nn.Sequential(
#             SKUnit(512, 1024, 32, 2, 8, 2, stride=2),
#             nn.ReLU(),
#             SKUnit(1024, 1024, 32, 2, 8, 2),
#             nn.ReLU(),
#             SKUnit(1024, 1024, 32, 2, 8, 2),
#             nn.ReLU()
#         )  # 8x8
#         self.pool = nn.AvgPool2d(8)
#         self.classifier = nn.Sequential(
#             nn.Linear(1024, class_num),
#             # nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         fea = self.basic_conv(x)
#         fea = self.stage_1(fea)
#         fea = self.stage_2(fea)
#         fea = self.stage_3(fea)
#         fea = self.pool(fea)
#         fea = torch.squeeze(fea)
#         fea = self.classifier(fea)
#         return fea


if __name__ == '__main__':
    x = torch.rand(8, 512, 32, 32)
    # conv = SKConv(64, 32, 3, 8, 2)
    # conv = SKConv(in_channels=512, out_channels=256, M=4, G=32, r=16, L=32)
    conv = SKUnit(inplanes=512, M=1, G=32, L=32)
    out = conv(x)
    criterion = nn.L1Loss()
    loss = criterion(out, x)
    loss.backward()
    print('out shape : {}'.format(out.shape))
    print('loss value : {}'.format(loss))
