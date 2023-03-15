import torch
import torch.nn as nn
from .vgg_block import vgg13_model, vgg16_model
import torch.nn.functional as F
from config import cfg
from .conv_utils import Conv2d, make_layers, CBAM, SAM
from .Decoder_Model import MSB_DCN
import cv2
import numpy as np


class Auto_Scale_Net(nn.Module):
    def __init__(self, backbone='vgg', pretrained=True, bn=False):
        super(Auto_Scale_Net, self).__init__()
        if backbone == 'vgg':
            self.feature_c4 = vgg13_model(pretrained)
        self.back_encoder = make_layers([512, 512, 512], 512, batch_norm=bn, dilation=True)
        self.decoder = MSB_DCN(512)
        # self.density_pred = nn.Sequential(
        #     Conv2d(in_channels=512, out_channels=128, kernel_size=3, same_padding=True, bn=True),
        #     Conv2d(in_channels=128, out_channels=64, kernel_size=3, same_padding=True, bn=True),
        #     Conv2d(in_channels=64, out_channels=1, kernel_size=1, same_padding=True, bn=False, NL='None')
        # )
        fc_flatted = 16 * 16
        self.avg_pool_16 = nn.AdaptiveAvgPool2d(16)
        self.avg_pool_1 = nn.AdaptiveAvgPool2d(1)
        self.conv_layer = nn.Sequential(
            Conv2d(512, 256, 3, same_padding=True, bn=True),
            Conv2d(256, 128, 3, same_padding=True, bn=True),
            Conv2d(128, 64, 3, same_padding=True, bn=True)
        )
        self.fc = nn.Linear(64, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        C2, C3, C4 = self.feature_c4(x)  # conv2 ,conv3,conv4,conv5
        C5 = self.back_encoder(C4)
        features = self.decoder(C5)
        features = self.avg_pool_16(features)
        features = self.conv_layer(features)
        features = self.avg_pool_1(features)
        features = torch.flatten(features, 1)
        features = self.fc(features)
        y = self.relu(features)
        return y


def crop_arrange(scale_mask, scale_label):
    scale_mask_array = scale_mask.numpy()
    mask_array = np.zeros_like(scale_mask_array, dtype=np.uint8)
    mask_array[scale_mask_array == scale_label] = 255
    ret, thresh = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
    mask, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox_list = []
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        bbox_list.append([x, y, w, h])
        return bbox_list
    else:
        return 0, 0, 0, 0


def resized_crop_image(image, bboxs):
    crop_images_list = []
    for b in bboxs:
        x, y, w, h = b
        crop_image = image[y:y + h, x:x + w, :]
        crop_images_list.append(crop_image)
    return crop_images_list


class Scale_CC(nn.Module):
    def __init__(self):
        super(Scale_CC, self).__init__()

    def forward(self, image_tensor, den_tensor, scale_mask):
        bboxs = crop_arrange(scale_mask, 3)
        return bboxs
