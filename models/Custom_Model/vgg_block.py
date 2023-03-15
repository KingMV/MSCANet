import torch.nn as nn
from torch.utils import model_zoo
from .conv_utils import BaseConv
from torchvision import models


def vgg16_model(pretrained=True):
    net = VGG()
    if pretrained:
        net.load_state_dict(load_vgg16_weight())
    return net


def vgg13_model(pretrained=True):
    net = VGG13()
    if pretrained:
        net.load_state_dict(load_vgg13_weight())
    return net


def load_vgg16_weight():
    state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
    old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
    new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
    new_dict = {}
    for i in range(13):
        new_dict['conv' + new_name[i] + '.conv.weight'] = \
            state_dict['features.' + str(old_name[2 * i]) + '.weight']
        new_dict['conv' + new_name[i] + '.conv.bias'] = \
            state_dict['features.' + str(old_name[2 * i]) + '.bias']
        new_dict['conv' + new_name[i] + '.bn.weight'] = \
            state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
        new_dict['conv' + new_name[i] + '.bn.bias'] = \
            state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
        new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
            state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
        new_dict['conv' + new_name[i] + '.bn.running_var'] = \
            state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']
    return new_dict


def load_vgg13_weight():
    state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
    old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37]
    new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3']
    new_dict = {}

    for i in range(10):
        new_dict['conv' + new_name[i] + '.conv.weight'] = \
            state_dict['features.' + str(old_name[2 * i]) + '.weight']
        new_dict['conv' + new_name[i] + '.conv.bias'] = \
            state_dict['features.' + str(old_name[2 * i]) + '.bias']
        new_dict['conv' + new_name[i] + '.bn.weight'] = \
            state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
        new_dict['conv' + new_name[i] + '.bn.bias'] = \
            state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
        new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
            state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
        new_dict['conv' + new_name[i] + '.bn.running_var'] = \
            state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']
    return new_dict


class VGG16_NoBN(nn.Module):
    def __init__(self, load_weights=False):
        super(VGG16_NoBN, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.features = self.make_layers(self.frontend_feat)
        self._initialize_weights()
        if load_weights:
            mod = models.vgg16(pretrained=True)
            self.features.load_state_dict(mod.features[0:23].state_dict())

    def forward(self, x):

        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
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


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)

        input = self.pool(conv3_3)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)

        input = self.pool(conv4_3)
        input = self.conv5_1(input)
        input = self.conv5_2(input)
        conv5_3 = self.conv5_3(input)

        return conv2_2, conv3_3, conv4_3, conv5_3


class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        # self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        # self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        # self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)

        input = self.pool(conv3_3)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)

        return conv2_2, conv3_3, conv4_3


def print_state_dict(state_dict):
    for name in state_dict.keys():
        print(name)
    print('the lenght of state_dict is {0}'.format(len(state_dict.keys())))


def print_model_parameter(net):
    for name in net.state_dict():
        print(name)
        # print(p)


if __name__ == '__main__':
    net = vgg13_model()
    print_model_parameter(net)
