import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .custom_loss import Balanced_loss, Custom_MSE_loss, Custom_Log_MSE_loss
from config import cfg
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
from functools import reduce
import math

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
restore_transform = standard_transforms.Compose([
    own_transforms.DeNormalize(*mean_std),
    standard_transforms.ToPILImage()
])
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])


class CrowdClass(nn.Module):
    def __init__(self, gpus, model_name):
        super(CrowdClass, self).__init__()
        self.model_name = model_name
        if model_name == 'Crowd_Class':  # Crowd_MSB
            from .Custom_Model.crowd_baseline import Crowd_MSB_Class as net
            self.CCN = net(bn=True)
        else:
            raise ValueError('error model name')
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_class = torch.FloatTensor([0]).cuda()
        # self.loss_class_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(np.asarray([1, 1, 1, 1.5])).float()).cuda()
        self.loss_class_fn = nn.CrossEntropyLoss().cuda()
        #
        self.batch_num = 1

    @property
    def loss(self):
        return self.loss_class

    def forward(self, img, gt_den_class):
        den_class = self.CCN(img)
        self.loss_class = self.loss_class_fn(den_class, gt_den_class)
        return den_class

    def test_forward(self, img):
        den_class = self.CCN(img)
        return den_class

    def patch_forward(self, img):
        img_numpy = np.squeeze(img.data.cpu().numpy(), axis=0)
        imgs_numpy = self.crop_image(img_numpy)
        # print(imgs_numpy.shape)
        imgs_tensor = torch.from_numpy(np.transpose(imgs_numpy, (0, 1, 4, 2, 3))).float().cuda()
        m, n = imgs_numpy.shape[0], imgs_numpy.shape[1]
        pred_class = -1
        for i in range(m):
            for j in range(n):
                den_class_pro = self.CCN(torch.unsqueeze(imgs_tensor[i][j], dim=0))
                den_class_pro = F.softmax(den_class_pro)

                pred_class = torch.argmax(den_class_pro, dim=1).cpu().item()
                print(pred_class)
                # if pred_class == 3:
                #     print(pred_class)
                #     factor = 2
                #     resize_img = imgs_numpy[i][j]
                #     resize_img = cv2.resize(resize_img,
                #                             (int(resize_img.shape[1] * factor), int(resize_img.shape[0] * factor)),
                #                             cv2.INTER_LINEAR)
                #     resize_img_tensor = torch.from_numpy(
                #         np.transpose(resize_img, (2, 0, 1))[np.newaxis, :, :, :]).float().cuda()
                #     resize_den = self.CCN(resize_img_tensor)
                #
                #     resize_den = resize_den.data.cpu().numpy()[0][0]
                #     scale_factor = (resize_den.shape[0] / 224) * (resize_den.shape[1] / 224)
                #     resize_den = cv2.resize(resize_den, (224, 224), cv2.INTER_LINEAR) * scale_factor
                #     predmaps_numpy[i][j] = resize_den
                # else:
                #     den_map = self.CCN(torch.unsqueeze(imgs_tensor[i][j], dim=0))
                #     predmaps_numpy[i][j] = den_map.data.cpu().numpy()[0][0]
                #
                #     pred_cnt += np.sum(predmaps_numpy[i][j])
                #     gt_cnt += np.sum(dens_numpy[i][j])
        # pred_map = image_concat(predmaps_numpy)
        # pred_map = pred_map[np.newaxis, :, :]
        # den_map = image_concat(dens_numpy)
        # den_map = den_map[np.newaxis, :, :]
        # self.loss_mse = self.loss_mse_fn(torch.from_numpy(pred_map), torch.from_numpy(den_map))
        return pred_class

    def crop_image(self, img, patch_size=224):
        img = np.transpose(img, (1, 2, 0))
        img_w, img_h = img.shape[1], img.shape[0]
        # patch_size
        pad_dw = patch_size - img.shape[1] % patch_size
        pad_dh = patch_size - img.shape[0] % patch_size

        img = cv2.copyMakeBorder(img, pad_dh // 2, pad_dh - pad_dh // 2, pad_dw // 2, pad_dw - pad_dw // 2,
                                 cv2.BORDER_CONSTANT, value=0)
        cols = img.shape[1] // patch_size
        rows = img.shape[0] // patch_size

        gx, gy = np.meshgrid(np.linspace(0, img.shape[1], cols + 1),
                             np.linspace(0, img.shape[0], rows + 1))
        gx = gx.astype(np.int)
        gy = gy.astype(np.int)
        divide_image = np.zeros([rows, cols, patch_size, patch_size, 3])
        for r in range(rows):
            for c in range(cols):
                divide_image[r, c] = img[gy[r][c]:gy[r + 1][c + 1], gx[r][c]:gx[r + 1][c + 1], :]
        return divide_image


class Crowd_Seg_Counter(nn.Module):
    def __init__(self, gpus, model_name):
        super(Crowd_Seg_Counter, self).__init__()
        self.model_name = model_name
        if model_name == 'AlexNet':
            from .SCC_Model.AlexNet import AlexNet as net
        elif model_name == 'Crowd_MSB':  # Crowd_MSB
            from .Custom_Model.crowd_fpn import Crowd_MSB_DCN as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_Baseline_Seg':  # Crowd_MSB
            from .Custom_Model.crowd_fpn import Crowd_MSB_DCN_Seg as net
            self.CCN = net(bn=True, dilation_ratio=[2, 4, 6], reduction_ratio=4, use_spp_seg=True)
        elif model_name == 'Crowd_Baseline_Seg_01':  # Crowd_MSB
            from .Custom_Model.crowd_fpn import Crowd_MSB_DCN_Seg as net
            self.CCN = net(bn=True, dilation_ratio=[2, 4, 6], reduction_ratio=4, use_spp_seg=True,
                           use_mask_feature=True)
        elif model_name == 'Crowd_Baseline_UNet_Seg':  # Crowd_MSB
            from .Custom_Model.crowd_fpn import Crowd_MSB_UNet_Seg as net
            self.CCN = net(bn=True, dilation_ratio=[2, 4, 6], reduction_ratio=4, use_spp_seg=True)
        elif model_name == 'Crowd_Scale_Baseline':  # Crowd_MSB FPN
            from .Custom_Model.crowd_fpn import Crowd_MSB_DCN_Scale as net
            self.CCN = net(bn=True, dilation_ratio=[2, 4, 6], reduction_ratio=4)
        elif model_name == 'Crowd_Scale_Baseline_01':  # Crowd_MSB FPN
            from .Custom_Model.crowd_fpn import Crowd_MSB_DCN_Scale_01 as net
            self.CCN = net(bn=True, dilation_ratio=[2, 4, 6], reduction_ratio=4)
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_seg = torch.FloatTensor([0]).cuda()
        self.loss_mse = torch.FloatTensor([0]).cuda()
        self.loss_class = torch.FloatTensor([0]).cuda()
        self.loss_s = torch.FloatTensor([0]).cuda()
        self.loss_h = torch.FloatTensor([0]).cuda()
        if cfg.BL_LOSS:
            print('use the custom balance loss')
            self.loss_mse_fn = Balanced_loss(reduction='sum').cuda()
        else:
            self.loss_mse_fn = nn.MSELoss(reduction='sum').cuda()
        # self.loss_class_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(np.asarray([1, 1, 1, 1.5])).float()).cuda()
        self.loss_class_fn = nn.CrossEntropyLoss().cuda()
        # self.loss_bce_fn = nn.BCELoss().cuda()
        self.loss_bce_fn = nn.CrossEntropyLoss().cuda()

        self.batch_num = 1

    def build_spp_loss(self, pred_map, gt_map, K=3):
        sal_loss = 0
        print(pred_map.shape)
        for i in range(K):
            if i == 0:
                sal_loss += self.loss_mse_fn(pred_map, gt_map)
            pred_map = F.max_pool2d(pred_map, (2, 2))
            gt_map = F.max_pool2d(gt_map, (2, 2))
            sal_loss += self.loss_mse_fn(pred_map, gt_map)
        return sal_loss / K

    def build_custom_spp_loss(self, pred_map, gt_map, K=3):
        # print('build_sal_loss')
        # print('the pred_map shape is {0}'.format(pred_map.shape))
        # print('the gt_map shape is {0}'.format(gt_map.shape))

        sal_loss_1 = self.loss_mse_fn(pred_map, gt_map)
        sal_loss_2 = self.loss_mse_fn(F.max_pool2d(pred_map, (2, 2)), F.max_pool2d(gt_map, (2, 2)))
        sal_loss_3 = self.loss_mse_fn(F.max_pool2d(pred_map, (4, 4)), F.max_pool2d(gt_map, (4, 4)))

        sal_loss = sal_loss_1 + 0.5 * sal_loss_2 + 0.5 * sal_loss_3
        # for i in range(K):
        #     if i == 0:
        #         sal_loss += self.loss_mse_fn(pred_map, gt_map)
        #     pred_map = F.max_pool2d(pred_map, (2, 2))
        #     gt_map = F.max_pool2d(gt_map, (2, 2))
        #     sal_loss += self.loss_mse_fn(pred_map, gt_map)
        return sal_loss

    @property
    def loss(self):
        return self.loss_mse / self.batch_num + self.loss_seg

    def get_loss(self):
        return self.loss_mse / self.batch_num, self.loss_seg

    # def get_sc_loss(self):
    #     return self.loss_s / self.batch_num, self.loss_h / self.batch_num, self.loss_class

    def forward(self, img, gt_map, gt_seg=None, is_train=True):
        self.batch_num = img.shape[0]
        if self.model_name == 'Crowd_Baseline_Seg' or self.model_name == 'Crowd_Baseline_UNet_Seg' or \
                self.model_name == 'Crowd_Baseline_Seg_01' or self.model_name == 'Crowd_Scale_Baseline' or \
                self.model_name == 'Crowd_Scale_Baseline_01' or self.model_name == 'Crowd_Scale_Baseline_02':
            density_map, seg_map = self.CCN(img)
            if cfg.use_sal and is_train:
                self.loss_mse = self.build_spp_loss(density_map, gt_map, K=3)
            elif cfg.use_sal and is_train:
                self.loss_mse = self.build_custom_spp_loss(density_map, gt_map, K=3)
            else:
                self.loss_mse = self.loss_mse_fn(density_map.squeeze(), gt_map.squeeze())
            if gt_seg is not None:
                self.loss_seg = self.loss_bce_fn(seg_map.squeeze(), gt_seg.squeeze())
            return density_map, seg_map

    def infernece_forward(self, img):
        density_map, _ = self.CCN(img)
        return density_map


class Crowd_Scale_Counter(nn.Module):
    def __init__(self, gpus, model_name):
        super(Crowd_Scale_Counter, self).__init__()
        self.model_name = model_name
        if model_name == 'Crowd_Scale_Baseline_02':  # Crowd_MSB FPN
            from .Custom_Model.crowd_fpn import Crowd_MSB_DCN_Scale_02 as net
            self.CCN = net(bn=True, dilation_ratio=[2, 4, 6], reduction_ratio=4)
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_scale_map = torch.FloatTensor([0]).cuda()
        self.loss_mse = torch.FloatTensor([0]).cuda()
        if cfg.BL_LOSS:
            print('use the custom balance loss')
            self.loss_mse_fn = Balanced_loss(reduction='sum').cuda()
        else:
            self.loss_mse_fn = nn.MSELoss(reduction='sum').cuda()
        # self.loss_class_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(np.asarray([1, 1, 1, 1.5])).float()).cuda()
        # self.loss_bce_fn = nn.BCELoss().cuda()
        self.loss_ce_fn = nn.CrossEntropyLoss().cuda()
        self.batch_num = 1

    def build_spp_loss(self, pred_map, gt_map, K=3):
        sal_loss = 0
        print(pred_map.shape)
        for i in range(K):
            if i == 0:
                sal_loss += self.loss_mse_fn(pred_map, gt_map)
            pred_map = F.max_pool2d(pred_map, (2, 2))
            gt_map = F.max_pool2d(gt_map, (2, 2))
            sal_loss += self.loss_mse_fn(pred_map, gt_map)
        return sal_loss / K

    def build_custom_spp_loss(self, pred_map, gt_map, K=3):
        sal_loss_1 = self.loss_mse_fn(pred_map, gt_map)
        sal_loss_2 = self.loss_mse_fn(F.max_pool2d(pred_map, (2, 2)), F.max_pool2d(gt_map, (2, 2)))
        sal_loss_3 = self.loss_mse_fn(F.max_pool2d(pred_map, (4, 4)), F.max_pool2d(gt_map, (4, 4)))
        sal_loss = sal_loss_1 + 0.5 * sal_loss_2 + 0.5 * sal_loss_3
        return sal_loss

    @property
    def loss(self):
        return self.loss_mse / self.batch_num + self.loss_scale_map

    def get_loss(self):
        return self.loss_mse / self.batch_num, self.loss_scale_map

    def cacluate_multi_stage_loss(self, pred_maps, gt_map, gt_scale_map=None):
        # print('caculate the multi-stage loss')
        assert len(pred_maps) == 4
        D, D1, D2, D3 = pred_maps

        gt_map_x = F.interpolate(gt_map, size=D1.size()[2:], mode='bilinear', align_corners=True)

        # multi_scale loss
        loss_mse_1 = self.loss_mse_fn(gt_map_x, D1)
        loss_mse_2 = self.loss_mse_fn(gt_map_x, D2)
        loss_mse_3 = self.loss_mse_fn(gt_map_x, D3)
        loss_multi_scale = loss_mse_1 + loss_mse_2 + loss_mse_3
        loss_mse = self.loss_mse_fn(D, gt_map)
        return loss_mse + 0.1 * loss_multi_scale

    def forward(self, img, gt_map, gt_scale_map=None, is_train=True):
        self.batch_num = img.shape[0]
        if self.model_name == 'Crowd_Scale_Baseline_02':
            density_maps, seg_map = self.CCN(img)
            self.loss_mse = self.cacluate_multi_stage_loss(density_maps, gt_map)
            if gt_scale_map is not None:
                self.loss_scale_map = self.loss_ce_fn(seg_map.squeeze(), gt_scale_map.squeeze())
            density_map = density_maps[0]
            return density_map, seg_map
        else:
            return ValueError('error model name')

    def infernece_forward(self, img):
        density_maps, scale_map = self.CCN(img)
        return density_maps[0], scale_map


class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name):
        super(CrowdCounter, self).__init__()
        self.model_name = model_name
        if model_name == 'AlexNet':
            from .SCC_Model.AlexNet import AlexNet as net
        elif model_name == 'VGG':
            from .SCC_Model.VGG import VGG as net
            self.CCN = net()
        elif model_name == 'CSRNet':
            from .SCC_Model.CSRNet import CSRNet as net
            self.CCN = net()
        elif model_name == 'Res101_SFCN':
            from .SCC_Model.Res101_SFCN import Res101_SFCN as net
            self.CCN = net()
        elif model_name == 'SANet':
            from .M2TCC_Model.SANet import SANet as net
            self.CCN = net()
        elif model_name == 'Crowd_CSRNet':
            from .Custom_Model.crowd_csrnet import Crowd_CSRNet as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_PSNet':
            from .Custom_Model.crowd_csrnet import Crowd_PSNet as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_MCNN':
            from .Custom_Model.crowd_csrnet import Crowd_MCNN as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_SAM':
            from .Custom_Model.crowd_csrnet import Crowd_SAM as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_DSAM':
            from .Custom_Model.crowd_csrnet import Crowd_DSAM as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_DDSAM':
            from .Custom_Model.crowd_csrnet import Crowd_DDSAM as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_trident':
            from .Custom_Model.crowd_csrnet import Crowd_trident as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_UNet':
            from .Custom_Model.crowd_unet import Crowd_UNet as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_SAM_UNet':
            from .Custom_Model.crowd_unet import Crowd_SAM_UNet as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_DSAM_UNet':
            from .Custom_Model.crowd_unet import Crowd_DSAM_UNet as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_DDSAM_UNet':
            from .Custom_Model.crowd_unet import Crowd_DDSAM_UNet as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_Seg_UNet':
            from .Custom_Model.crowd_segmentation import Crowd_Seg_UNet as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_Seg_Decoder':
            from .Custom_Model.crowd_segmentation import Crowd_Seg_Decoder as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_Seg':
            from .Custom_Model.crowd_segmentation import Crowd_Seg as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_MSPNet':
            from .Custom_Model.crowd_segmentation import Crowd_MSPNet as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_MSB':  # Crowd_MSB
            from .Custom_Model.crowd_fpn import Crowd_MSB_DCN as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_VGG':  # Crowd_MSB
            from .Custom_Model.crowd_fpn import Crowd_VGG as net
            self.CCN = net(pretrained=True, bn=True)
        elif model_name == 'Crowd_VGG_NoBN':  # Crowd_MSB
            from .Custom_Model.crowd_fpn import Crowd_VGG_NoBN as net
            self.CCN = net(pretrained=True, bn=False)
        elif model_name == 'Crowd_MRCM':
            from .Custom_Model.crowd_fpn import Crowd_MRCM as net
            self.CCN = net(pretrained=True, bn=True, M=3)
            # self.CCN = net(pretrained=True, bn=True, M=3)
        elif model_name == 'Crowd_MRCM_AB':
            from .Custom_Model.crowd_fpn import Crowd_AB as net
            self.CCN = net(pretrained=True, bn=False, D=3)
        elif model_name == 'Crowd_SC':  # Crowd_MSB
            from .Custom_Model.crowd_baseline import Crowd_MSB_SC as net
            self.CCN = net(bn=True)
        elif model_name == 'Crowd_SC_0325':  # Crowd_MSB
            from .Custom_Model.crowd_baseline import Crowd_MSB_SC_0325 as net
            self.CCN = net(bn=True)
        elif model_name == 'Crowd_high':  # Crowd_MSB
            from .Custom_Model.crowd_baseline import Crowd_MSB_SC as net
            self.CCN = net(bn=True)
        elif model_name == 'Crowd_Baseline':  # Crowd_MSB
            from .Custom_Model.crowd_fpn import Crowd_MSB_DCN as net
            self.CCN = net(bn=True, reduction_ratio=4)
        elif model_name == 'Crowd_Baseline_UNet':  # Crowd_MSB FPN
            from .Custom_Model.crowd_fpn import Crowd_MSB_UNet as net
            self.CCN = net(bn=True, reduction_ratio=4)
        elif model_name == 'Crowd_Scale_Baseline':  # Crowd_MSB FPN
            from .Custom_Model.crowd_fpn import Crowd_MSB_DCN_Scale as net
            self.CCN = net(bn=True, dilation_ratio=[2, 4, 6], reduction_ratio=4)
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_seg = torch.FloatTensor([0]).cuda()
        self.loss_mse = torch.FloatTensor([0]).cuda()
        self.loss_class = torch.FloatTensor([0]).cuda()
        self.loss_s = torch.FloatTensor([0]).cuda()
        self.loss_h = torch.FloatTensor([0]).cuda()
        self.loss_d = torch.FloatTensor([0]).cuda()
        self.loss_mse_fn = nn.MSELoss(reduction='mean').cuda()
        # self.loss_class_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(np.asarray([1, 1, 1, 1.5])).float()).cuda()
        # self.loss_class_fn = nn.CrossEntropyLoss().cuda()
        self.loss_bce_fn = nn.BCELoss(reduction='mean').cuda()

        self.batch_num = 1

    @property
    def loss(self):
        return self.loss_mse / self.batch_num + self.loss_class

    def get_loss(self):
        return self.loss_mse, self.loss_seg

    def get_all_loss(self):
        return self.loss_mse, self.loss_seg, self.loss_d

    def build_seg_loss(self, gt_seg, pred_segs):
        p_seg_3, p_seg_2, p_seg_1 = pred_segs

        gt_seg_3 = gt_seg
        gt_seg_2 = F.interpolate(gt_seg, scale_factor=0.5, mode='bilinear', align_corners=True)
        gt_seg_1 = F.interpolate(gt_seg, scale_factor=0.25, mode='bilinear', align_corners=True)
        # print(gt_seg_1.shape)
        # print(p_seg_1.shape)
        # print(torch.max(gt_seg_1))
        # print(torch.min(gt_seg_1))
        l_1 = self.loss_bce_fn(p_seg_1.squeeze(), gt_seg_1.squeeze())
        l_2 = self.loss_bce_fn(p_seg_2.squeeze(), gt_seg_2.squeeze())
        l_3 = self.loss_bce_fn(p_seg_3.squeeze(), gt_seg_3.squeeze())
        L = 0.01 * l_1 + 0.1 * l_2 + l_3
        L = l_1 + l_2 + l_3
        print('seg_1:{0},seg_2:{1},seg_3:{2}'.format(l_1, l_2, l_3))
        return L

    def calculate_f_cos(self, fmps):
        vectors = []
        vector_sum = torch.mean(fmps[0], dim=1).view(fmps[0].shape[0], -1)
        for i, f in enumerate(fmps):
            v = torch.mean(f, dim=1).view(f.shape[0], -1)
            if i > 0:
                vector_sum += v
            vectors.append(v)
        d_list = []
        distance = 0
        for v in vectors:
            d = torch.mean(torch.cosine_similarity(vector_sum, (vector_sum - v), dim=1))
            d_list.append(d)
            distance += d
        distance = distance / len(d_list)

        return distance

    def build_msp_loss(self, gt_den, gt_seg, pred_dens, pred_segs):
        p_den_1, p_den_2, p_den_3 = pred_dens
        p_seg_1, p_seg_2, p_seg_3 = pred_segs

        gt_seg_3 = gt_seg
        gt_seg_2 = F.interpolate(gt_seg, scale_factor=0.5, mode='bilinear', align_corners=True)
        gt_seg_1 = F.interpolate(gt_seg, scale_factor=0.25, mode='bilinear', align_corners=True)

        gt_den_3 = gt_den
        gt_den_2 = F.interpolate(gt_den, scale_factor=0.5, mode='bilinear', align_corners=True)
        gt_den_1 = F.interpolate(gt_den, scale_factor=0.25, mode='bilinear', align_corners=True)

        # print(gt_seg_1.shape)
        # print(p_seg_1.shape)
        # print(torch.max(gt_seg_1))
        # print(torch.min(gt_seg_1))
        # print('gt{0}\t{1}\t{2}'.format(gt_den_1.shape, gt_den_2.shape, gt_den_3.shape))
        # print('pre{0}\t{1}\t{2}'.format(p_den_1.shape, p_den_2.shape, p_den_3.shape))
        l_mse_1 = self.loss_mse_fn(p_den_1.squeeze(), gt_den_1.squeeze())
        l_mse_2 = self.loss_mse_fn(p_den_2.squeeze(), gt_den_2.squeeze())
        l_mse_3 = self.loss_mse_fn(p_den_3.squeeze(), gt_den_3.squeeze())

        l_bce_1 = self.loss_bce_fn(p_seg_1.squeeze(), gt_seg_1.squeeze())
        l_bce_2 = self.loss_bce_fn(p_seg_2.squeeze(), gt_seg_2.squeeze())
        l_bce_3 = self.loss_bce_fn(p_seg_3.squeeze(), gt_seg_3.squeeze())

        s1 = 0.0001 * l_bce_1 + l_mse_1
        s2 = 0.0001 * l_bce_2 + l_mse_2
        s3 = 0.0001 * l_bce_3 + l_mse_3

        # L = s1 / 8 + s2 / 4 + s3 / 2
        L = s3
        # self.loss_seg = l_bce_1 + l_bce_2 + l_bce_3
        self.loss_seg = l_bce_3
        # L = l_1 + l_2 + l_3
        # print('seg_1:{0},seg_2:{1},seg_3:{2}'.format(l_1, l_2, l_3))
        return L

    def build_spp_loss(self, pred_map, gt_map, K=3):
        # print('build_sal_loss')
        sal_loss = 0
        # print('the pred_map shape is {0}'.format(pred_map.shape))
        # print('the gt_map shape is {0}'.format(gt_map.shape))
        sal_loss += self.loss_mse_fn(pred_map, gt_map)
        for i in range(K):
            pred_map = F.max_pool2d(pred_map, (2, 2))
            # pred_map = F.adaptive_max_pool2d(pred_map,)
            gt_map = F.max_pool2d(gt_map, (2, 2))
            sal_loss += self.loss_mse_fn(pred_map, gt_map)
        return sal_loss

    def build_custom_spp_loss(self, pred_map, gt_map, K=3):
        # print('build_sal_loss')
        # print('the pred_map shape is {0}'.format(pred_map.shape))
        # print('the gt_map shape is {0}'.format(gt_map.shape))

        sal_loss_1 = self.loss_mse_fn(pred_map, gt_map)
        sal_loss_2 = self.loss_mse_fn(F.max_pool2d(pred_map, (2, 2)), F.max_pool2d(gt_map, (2, 2)))
        sal_loss_3 = self.loss_mse_fn(F.max_pool2d(pred_map, (4, 4)), F.max_pool2d(gt_map, (4, 4)))

        sal_loss = sal_loss_1 + 0.5 * sal_loss_2 + 0.5 * sal_loss_3
        return sal_loss

    def build_block_loss(self, pred_map, gt_map, block_size=4):
        square_error_map = (pred_map - gt_map) ** 2
        # square_error_map = F.mse_loss(pred_map,gt_map)
        # square_error_map = (pred_map - gt_map) ** 2
        # print(square_error_map)
        # print(square_error_map)
        element_amount = reduce(lambda x, y: x * y, square_error_map.shape)
        square_error_map = square_error_map / element_amount
        block_square_error = build_block(square_error_map, block_size)
        block_gt_map = build_block(gt_map, block_size)
        block_loss = block_square_error / (block_gt_map + 1)
        loss = torch.sum(block_loss)
        return loss

    def get_sc_loss(self):
        return self.loss_s / self.batch_num, self.loss_h / self.batch_num, self.loss_class

    def forward(self, img, gt_map, gt_seg=None, gt_den_class=None, is_train=True):
        self.batch_num = img.shape[0]
        if self.model_name == 'Crowd_SC' or self.model_name == 'Crowd_SC_01' or self.model_name == 'Crowd_high' \
                or self.model_name == 'Crowd_SC_0325':
            density_map, density_class = self.CCN(img)
            self.loss_mse = self.build_loss(density_map.squeeze(), gt_map.squeeze())
            self.loss_class = torch.FloatTensor([0]).cuda()
            if gt_den_class is not None:
                self.loss_class = self.loss_class_fn(density_class, gt_den_class)
            return density_map
        elif self.model_name == 'Crowd_SC_05' or self.model_name == 'Crowd_SC_06':
            sparse_map, high_map, density_level = self.CCN(img)
            self.loss_s = self.build_loss(sparse_map.squeeze(), gt_map.squeeze())
            self.loss_h = self.build_loss(high_map.squeeze(), gt_map.squeeze())
            self.loss_mse = self.loss_s + self.loss_h
            self.loss_class = torch.FloatTensor([0]).cuda()
            if gt_den_class is not None:
                self.loss_class = self.loss_class_fn(density_level, gt_den_class)
            return sparse_map, high_map
        elif self.model_name == 'Crowd_Seg_UNet' or self.model_name == 'Crowd_Seg':
            den_map, seg_map = self.CCN(img)
            if cfg.use_sal and is_train:
                self.loss_mse = self.build_spp_loss(den_map, gt_map, K=3)
            elif cfg.use_sal_1 and is_train:
                self.loss_mse = self.build_custom_spp_loss(den_map, gt_map)
            else:
                self.loss_mse = self.build_loss(den_map.squeeze(), gt_map.squeeze())
            if gt_seg is not None:
                self.loss_seg = self.loss_bce_fn(seg_map.squeeze(), gt_seg.squeeze())
            return den_map, seg_map
        elif self.model_name == 'Crowd_Seg_Decoder':
            den_map, seg_map = self.CCN(img)
            # if cfg.use_sal and is_train:
            #     self.loss_mse = self.build_spp_loss(den_map, gt_map, K=3)
            # elif cfg.use_sal_1 and is_train:
            #     self.loss_mse = self.build_custom_spp_loss(den_map, gt_map)
            # else:
            #     self.loss_mse = self.build_loss(den_map.squeeze(), gt_map.squeeze())
            self.loss_mse = self.build_loss(den_map.squeeze(), gt_map.squeeze())
            if gt_seg is not None:
                self.loss_seg = self.build_seg_loss(gt_seg, seg_map)
            return den_map, seg_map[0]
        elif self.model_name == 'Crowd_Seg_Decoder':
            den_map, seg_map = self.CCN(img)
            # if cfg.use_sal and is_train:
            #     self.loss_mse = self.build_spp_loss(den_map, gt_map, K=3)
            # elif cfg.use_sal_1 and is_train:
            #     self.loss_mse = self.build_custom_spp_loss(den_map, gt_map)
            # else:
            #     self.loss_mse = self.build_loss(den_map.squeeze(), gt_map.squeeze())
            self.loss_mse = self.build_loss(den_map.squeeze(), gt_map.squeeze())
            if gt_seg is not None:
                self.loss_seg = self.build_seg_loss(gt_seg, seg_map)
            return den_map, seg_map[0]
        elif self.model_name == 'Crowd_MSPNet':
            den_maps, seg_maps = self.CCN(img)
            density_map = den_maps[-1]
            if is_train:
                self.loss_mse = self.build_msp_loss(gt_map, gt_seg, den_maps, seg_maps)
            else:
                self.loss_mse = self.loss_mse_fn(density_map.squeeze(), gt_map.squeeze())

            return density_map, seg_maps[-1]
        elif self.model_name == 'Crowd_MCNN':
            density_map, f_mps = self.CCN(img)
            if is_train:
                self.loss_d = self.calculate_f_cos(f_mps)
                self.loss_mse = self.build_loss(density_map.squeeze(), gt_map.squeeze())
            else:
                self.loss_mse = self.loss_mse_fn(density_map.squeeze(), gt_map.squeeze())

            return density_map
        else:
            # CSRNet et al baseline
            if cfg.SM:
                density_map = self.CCN(img, is_train)
            else:
                density_map = self.CCN(img)
            if cfg.use_sal and is_train:
                # self.loss_mse = self.build_spp_loss(density_map, gt_map, K=3)
                self.loss_mse = self.build_block_loss(density_map, gt_map, block_size=4)
                # print(self.loss_mse)
            elif cfg.use_sal_1 and is_train:
                self.loss_mse = self.build_custom_spp_loss(density_map, gt_map)
                # self.loss_mse = self.build_custom_spp_loss(density_map, gt_map)
            else:
                # print(density_map.shape)
                # print(gt_map.shape)
                self.loss_mse = self.build_loss(density_map.squeeze(), gt_map.squeeze())
            return density_map

    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)
        return loss_mse

    def infernece_forward(self, img):
        density_map = self.CCN(img)
        return density_map

    def test_forward(self, img, gt_map, gt_class):
        sparse_map, high_map, density_level = self.CCN(img)
        self.loss_s = self.build_loss(sparse_map.squeeze(), gt_map.squeeze())
        self.loss_h = self.build_loss(high_map.squeeze(), gt_map.squeeze())
        self.loss_mse = self.loss_s + self.loss_h
        self.loss_class = self.loss_class_fn(density_level, gt_class)
        return sparse_map, high_map, density_level

    def scale_test_forward(self, img, gt_map, scale_mask):
        # 1st prediction
        density_map_org = self.CCN(img)
        self.loss_mse = self.build_loss(density_map_org.squeeze(), gt_map.squeeze())
        density_map_org = density_map_org.data.cpu().numpy()[0, 0]

        print('gt is ', torch.sum(gt_map).data.cpu())
        print('org_pred density is ', np.sum(density_map_org))
        img_pic = restore_transform(torch.squeeze(img, dim=0).data.cpu())
        # print('the img_pic shape is ', img_pic.size)
        # img = img.data.cpu().numpy()[0]
        # img_pic = np.transpose(img, (1, 2, 0))
        # print('the image shape is ', img_pic.shape)
        # 2: get the crop area
        bboxs = get_crop_area(scale_mask.data.cpu(), 3)
        # print(bboxs)
        if bboxs is not None:
            #
            w_org = bboxs[0][2]
            h_org = bboxs[0][3]
            # crop_image
            pred_map_crop = density_map_org[bboxs[0][1]:bboxs[0][1] + h_org, bboxs[0][0]:bboxs[0][0] + w_org]
            # print('the crop density is ', np.sum(pred_map_crop))

            resized_imgs = resized_crop_image(img_pic, bboxs, scale_factor=1.5)

            img_tensor = img_transform(resized_imgs[0])
            # img_pic = np.transpose(resized_imgs[0], (2, 0, 1))

            crop_image_tensor = torch.unsqueeze(img_tensor, dim=0).cuda()
            # print('the crop image shape is ', crop_image_tensor.shape)
            # predicted density map of cropped area
            density_map_new = self.CCN(crop_image_tensor)
            density_map_new = resize_density(density_map_new.data.cpu().numpy()[0, 0], target_size=(w_org, h_org))
            print('the cc density is ', np.sum(density_map_new))
            density_map_org[bboxs[0][1]:bboxs[0][1] + h_org, bboxs[0][0]:bboxs[0][0] + w_org] = density_map_new

            density_map = density_map_org[np.newaxis, np.newaxis, :, :]
            print('the prediction density_map is ', np.sum(density_map))
        else:
            density_map = density_map_org[np.newaxis, np.newaxis, :, :]
        return density_map


def image_concat(divide_image):
    m, n, grid_h, grid_w = [divide_image.shape[0], divide_image.shape[1],  # 每行，每列的图像块数
                            divide_image.shape[2], divide_image.shape[3]]  # 每个图像块的尺寸

    restore_image = np.zeros([m * grid_h, n * grid_w])
    for i in range(m):
        for j in range(n):
            restore_image[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w] = divide_image[i, j, :]
    return restore_image


def resize_density(density, target_size):
    width, height = density.shape[1], density.shape[0]

    height_new = target_size[1]
    width_new = target_size[0]
    scale_factor = (height_new / height) * (width_new / width)
    density = cv2.resize(density, (width_new, height_new), interpolation=cv2.INTER_LINEAR) / scale_factor

    return density


def resized_crop_image(image, bboxs, scale_factor):
    crop_images_list = []
    for b in bboxs:
        x, y, w, h = b
        # image = Image.fromarray(image)
        # crop_image = image[y:y + h, x:x + w, :]
        crop_image = image.crop((x, y, x + w, y + h))
        crop_image = crop_image.resize((int(w * scale_factor), int(h * scale_factor)), Image.BILINEAR)
        # crop_image = cv2.resize(crop_image, (int(w * scale_factor), int(h * scale_factor)), cv2.INTER_CUBIC)
        crop_images_list.append(crop_image)
    return crop_images_list


def get_crop_area(scale_mask, scale_label):
    '''

    :param scale_mask:
    :param scale_label:
    :return:
    '''
    scale_mask_array = scale_mask.numpy()[0, 0]
    # print(scale_mask_array)
    # print(np.sum(scale_mask_array == 3))
    mask_array = np.zeros([scale_mask_array.shape[0], scale_mask_array.shape[1]], dtype=np.uint8)
    mask_array[scale_mask_array == scale_label] = 255
    # print('~~~~~~~~~~~~~~~~')
    # print(mask_array.shape)
    # print('~~~~~~~~~~~~~~~~')
    ret, thresh = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox_list = []
    # print(contours)
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        if w > 10:
            bbox_list.append([x, y, w, h])
            return bbox_list
        else:
            return None
    else:
        return None


def build_block(x, size):
    # x shape=(1, c, h, w)
    height = x.shape[2]
    width = x.shape[3]
    padding_height = math.ceil((math.ceil(height / size) * size - height) / 2)
    padding_width = math.ceil((math.ceil(width / size) * size - width) / 2)
    return F.avg_pool2d(x, size, stride=size, padding=(padding_height, padding_width),
                        count_include_pad=True) * size * size


class CrowdSegmentation(nn.Module):
    def __init__(self, gpus, model_name):
        super(CrowdSegmentation, self).__init__()
        if model_name == 'Crowd_MSB_Seg':  # Crowd_MSB
            from .Custom_Model.crowd_fpn import Crowd_MSB_Seg as net
            self.CCN = net(pretrained=True, bn=True)
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_seg = torch.FloatTensor([0]).cuda()
        self.loss_bce_fn = nn.BCELoss(reduction='mean').cuda()

    @property
    def loss(self):
        return self.loss_seg

    def get_loss(self):
        return self.loss_mse, self.loss_seg

    def forward(self, img, gt_map):
        seg_map = self.CCN(img)
        self.loss_seg = self.loss_bce_fn(seg_map.squeeze(), gt_map.squeeze())
        return seg_map

    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)
        return loss_mse

    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map
