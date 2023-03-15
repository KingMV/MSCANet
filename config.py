import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

# ------------------------------TRAIN------------------------
__C.SEED = 3035  # random seed,  for reproduction
__C.DATASET = 'SHHA'  # dataset selection: GCC, SHHA, SHHB, UCF50, QNRF, WE, Mall, UCSD
# __C.DATASET = 'SHHA'  # dataset selection: GCC, SHHA, SHHB, UCF50, QNRF, WE, Mall, UCSD
# __C.DATASET = 'SHHB'  # dataset selection: GCC, SHHA, SHHB, UCF50, QNRF, WE, Mall, UCSD
#
if __C.DATASET == 'UCF50':  # only for UCF50
    from datasets.UCF50.setting import cfg_data

    __C.VAL_INDEX = cfg_data.VAL_INDEX

if __C.DATASET == 'GCC':  # only for GCC
    from datasets.GCC.setting import cfg_data

    __C.VAL_MODE = cfg_data.VAL_MODE
# Crowd_MSB pretraine
__C.CC_NET = 'Crowd_CSRNet'  # net selection: MCNN, AlexNet, VGG, VGG_DECODER, Res50, CSRNet, SANet,VGG13_Custom,Crowd_FPN_Base,Crowd_UNet
# __C.SEGNET = 'Crowd_MSB_Seg'  Crowd_Seg_UNet Res101_SFCN Crowd_PSNet Crowd_MCNN
__C.add_seg_mask = False
__C.density_stride = 8
__C.batch_num = 1
__C.mask_input_image = False
__C.DEN_CLASS = False

__C.step_train = False

__C.PRE_GCC = False  # use the pretrained model on GCC dataset
__C.PRE_GCC_MODEL = 'path to model'  # path to model

__C.SEG_RESUME = False  # contine training

__C.RESUME = False  # contine training
__C.RESUME_PATH = './exp/04-15_11-09_SHHA_02_Crowd_Baseline_UNet_Seg_0.0001/latest_state.pth'  #
__C.RESUME_SEG_PATH = '../crowd_seg/exp/02-23_04-03_SHHA_01_Crowd_MSB_Seg_0.0001/all_ep_54_loss_0.25.pth'
__C.BL_LOSS = False
__C.GPU_ID = [1]  # sigle gpu: [0], [1] ...; multi gpus: [0,1]
__C.mst = False
__C.use_sal = False
__C.SM = False
__C.den_ds_ratio = 8
###############################################
__C.data_trans = 0

# learning rate settings
__C.LR = 1e-4  # learning rate
__C.LR_DECAY = 0.995  # decay rate
__C.LR_DECAY_START = -1  # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1  # decay frequency
__C.MAX_EPOCH = 1000

if __C.CC_NET == 'Crowd_MSB':
    __C.batch_num = 32
    __C.data_trans = 0
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.mst = True

if __C.CC_NET == 'Res101_SFCN':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.mst = True
if __C.CC_NET == 'Crowd_Baseline':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.mst = False

if __C.CC_NET == 'Crowd_VGG':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.mst = False

if __C.CC_NET == 'Crowd_VGG_NoBN':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.mst = False

if __C.CC_NET == 'Crowd_MRCM':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.mst = False

if __C.CC_NET == 'Crowd_MRCM_AB':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.mst = False

if __C.CC_NET == 'CSRNet':
    __C.batch_num = 32
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.mst = False

if __C.CC_NET == 'SANet':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.mst = False

if __C.CC_NET == 'Crowd_CSRNet':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = True
    __C.use_sal_1 = False
    __C.mst = False
    __C.add_seg_mask = False

if __C.CC_NET == 'Crowd_PSNet':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.mst = False
    __C.add_seg_mask = False
    __C.SM = True
if __C.CC_NET == 'Crowd_MCNN':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.mst = False
    __C.add_seg_mask = False

if __C.CC_NET == 'Crowd_SAM' or __C.CC_NET == 'Crowd_trident':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.mst = False
if __C.CC_NET == 'Crowd_DSAM' or __C.CC_NET == 'Crowd_DDSAM':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.mst = False
if __C.CC_NET == 'Crowd_SAM_UNet' or __C.CC_NET == 'Crowd_DSAM_UNet' or __C.CC_NET == 'Crowd_DDSAM_UNet':
    __C.batch_num = 6
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.mst = True

if __C.CC_NET == 'Crowd_UNet':
    __C.batch_num = 16
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.mst = False

if __C.CC_NET == 'Crowd_Baseline_UNet':
    __C.batch_num = 16
    __C.data_trans = 0
    __C.use_sal = True
    __C.BL_LOSS = False
    __C.mst = False
if __C.CC_NET == 'Crowd_Seg':
    __C.batch_num = 16
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.mst = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.add_seg_mask = True
if __C.CC_NET == 'Crowd_Seg_UNet' or __C.CC_NET == 'Crowd_Seg_Decoder' or __C.CC_NET == 'Crowd_MSPNet':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.mst = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.add_seg_mask = True
if __C.CC_NET == 'Crowd_Scale_Baseline':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.mst = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.add_seg_mask = False
    __C.add_scale_mask = True
if __C.CC_NET == 'Crowd_Scale_Baseline_01':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.mst = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.add_seg_mask = False
    __C.add_scale_mask = True

if __C.CC_NET == 'Crowd_Scale_Baseline_02':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.mst = False
    __C.use_sal = False
    __C.use_sal_1 = False
    __C.add_seg_mask = False
    __C.add_scale_mask = True

if __C.CC_NET == 'Crowd_Baseline_UNet_Seg':
    __C.batch_num = 12
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.use_sal = True
    __C.mst = False
    __C.add_seg_mask = True
if __C.CC_NET == 'Crowd_SC':
    __C.batch_num = 20
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.mst = False
    __C.DEN_CLASS = True
if __C.CC_NET == 'Crowd_Class':
    __C.batch_num = 16
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.mst = False
    __C.DEN_CLASS = True

# multi-task learning weights, no use for single model, such as MCNN, VGG, VGG_DECODER, Res50, CSRNet, and so on

__C.LAMBDA_1 = 1e-3  # SANet:0.001 CMTL 0.0001

# print
__C.PRINT_FREQ = 5

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
               + '_' + __C.DATASET \
               + '_' + __C.CC_NET \
               + '_' + str(__C.LR)

if __C.DATASET == 'UCF50':
    __C.EXP_NAME += '_' + str(__C.VAL_INDEX)

if __C.DATASET == 'GCC':
    __C.EXP_NAME += '_' + __C.VAL_MODE

__C.EXP_PATH = './exp'  # the path of logs, checkpoints, and current codes

# ------------------------------VAL------------------------
__C.VAL_DENSE_START = 400
__C.VAL_FREQ = 10  # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

# ------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1  # must be 1 for training images with the different sizes

# ================================================================================
# ================================================================================
# ================================================================================
