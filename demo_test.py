from models.Custom_Model.crowd_sac import SAConv2d
import torch
from functools import reduce
from models.Custom_Model.Decoder_Model import SAM01
from models.Custom_Model.vgg_block import vgg13_model
from models.CC import CrowdCounter
from models.Custom_Model.dense_aspp import DenseAspp
import torch.nn.functional as F
from datasets.QNRF.loading_data import loading_data
import os
from PIL import Image
import glob

# root = r'../Crowd_Dataset/ShanghaiTech'
# kernel_size = 15
# method_name = 'fixed'
# method = 'fixed_sigma_{0}'.format(kernel_size)  # adaptive
# Part = 'B'
#
# mod = 16
#
# part_train = os.path.join(root, 'part_{0}/train_data'.format(Part), 'images')
# part_test = os.path.join(root, 'part_{0}/test_data'.format(Part), 'images')
#
# for path in glob.glob(os.path.join(part_train, '*.jpg')):
#     print(path)
#     img = Image.open(path)
#     print(img.size)
train_loader, test_loader, _ = loading_data()

for i, data in enumerate(train_loader):
    img, den = data
    print('batch:{0},shape of den {1}\t gt:{2}'.format(i, img.shape, torch.sum(den[0]) / 100.0))
# import torchsummary
# from torchvision.models.resnet import resnet50
#
# model = resnet50()
# model.cuda(0)
# torchsummary.summary(model, (3, 224, 224))


# def show_architecture(architecture):
# input: list of network architecture
#     for i in range(len(architecture)):
#         print(architecture[i])

# model = CrowdCounter([3], 'Crowd_SAM')
# for i,j in model.named_parameters():
#     print(i)
#
# m = SAM01(512, 3, 256).cuda()
# x = torch.rand((1, 512, 32, 32)).cuda()
# y = m(x)
# print(y.shape)
# x = torch.ones((1, 1, 10, 10))
# element_amount = reduce(lambda x, y: x * y, x.shape)
# print(element_amount)
# y = F.avg_pool2d(x, (2, 2))*4
# print(y)
# print(y.shape)
# model = DenseAspp(512, 256, 512)
#
# y = model(x)
# print(y.shape)
# import numpy as np
# import cv2
#
# x = np.random.random((100, 100))
# print(np.sum(x))
# y = cv2.resize(x, (x.shape[1] // 2, x.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
# print(np.sum(y))
