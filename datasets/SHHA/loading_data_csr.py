from torchvision import transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from .SHHA import SHHA
from .setting import cfg_data
import torch
import random


# file objetc : create train_loader and test_loader


def get_min_size(batch):
    min_ht = cfg_data.TRAIN_SIZE[0]
    min_wd = cfg_data.TRAIN_SIZE[1]

    for i_sample in batch:

        _, ht, wd = i_sample.shape
        if ht < min_ht:
            min_ht = ht
        if wd < min_wd:
            min_wd = wd
    return min_ht, min_wd


def random_crop(img, den, dst_size):
    # dst_size: ht, wd
    _, ts_hd, ts_wd = img.shape

    x1 = random.randint(0, ts_wd - dst_size[1]) // cfg_data.LABEL_FACTOR * cfg_data.LABEL_FACTOR
    y1 = random.randint(0, ts_hd - dst_size[0]) // cfg_data.LABEL_FACTOR * cfg_data.LABEL_FACTOR
    x2 = x1 + dst_size[1]
    y2 = y1 + dst_size[0]

    label_x1 = x1 // cfg_data.LABEL_FACTOR
    label_y1 = y1 // cfg_data.LABEL_FACTOR
    label_x2 = x2 // cfg_data.LABEL_FACTOR
    label_y2 = y2 // cfg_data.LABEL_FACTOR

    return img[:, y1:y2, x1:x2], den[label_y1:label_y2, label_x1:label_x2]


def loading_data():
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA
    factor = cfg_data.LABEL_FACTOR

    #######################################
    ''' data transform'''
    main_transform_list = []
    if cfg_data.USE_FLIP:
        main_transform_list.append(own_transforms.RandomHorizontallyFlip())
    # if cfg_data.USE_CROP:
    #     main_transform_list.append(standard_transforms.RandomCrop())
    train_main_transform = own_transforms.Compose(main_transform_list)

    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.GTScaleDown(factor),
        own_transforms.LabelNormalize(log_para)
    ])
    ###########################################33
    train_set = SHHA(cfg_data.DATA_PATH + '/train_data/', 'train', main_transform=train_main_transform,
                     img_transform=img_transform, gt_transform=gt_transform)
    if cfg_data.TRAIN_BATCH_SIZE == 1:
        train_loader = DataLoader(train_set, batch_size=1, num_workers=1, shuffle=True, drop_last=False)
    else:
        raise ValueError('the batch number only 1')
    val_set = SHHA(cfg_data.DATA_PATH + '/test_data/', 'test', main_transform=None, img_transform=img_transform,
                   gt_transform=gt_transform)
    val_loader = DataLoader(val_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=False)

    return train_loader, val_loader, None
