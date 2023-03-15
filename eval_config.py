import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

# ------------------------------TRAIN------------------------
__C.SEED = 3035  # random seed,  for reproduction
__C.DATASET = 'SHHA_02'  # dataset selection: GCC, SHHA, SHHB, UCF50, QNRF, WE, Mall, UCSD

if __C.DATASET == 'UCF50':  # only for UCF50
    from datasets.UCF50.setting import cfg_data

    __C.VAL_INDEX = cfg_data.VAL_INDEX

if __C.DATASET == 'GCC':  # only for GCC
    from datasets.GCC.setting import cfg_data

    __C.VAL_MODE = cfg_data.VAL_MODE
# Crowd_MSB pretrained Crowd_SC
__C.CCNET = 'Crowd_CSRNet'  # net selection: MCNN, AlexNet, VGG, VGG_DECODER, Res50, CSRNet, SANet,VGG13_Custom,Crowd_FPN_Base,Crowd_UNet
__C.SCALENET = ''
__C.add_seg_mask = False
__C.density_stride = 1
__C.batch_num = 1
__C.mask_input_image = False
__C.DEN_CLASS = False

__C.step_train = False

__C.PRE_GCC = False  # use the pretrained model on GCC dataset
__C.PRE_GCC_MODEL = 'path to model'  # path to model

__C.RESUME_SCALE = False  # contine training

# csrnet model path
# ../CSRNet_test/exp/02-14_18-51_SHHA01_CSRNet_1e-05/all_ep_417_mae_67.9_mse_111.6.pth
__C.RESUME_CC = True  # contine training
# __C.RESUME_CC_PATH = '../crowd_code/exp/02-14_08-10_SHHA_01_Crowd_MSB_0.0001/' \
#                      'all_ep_567_mae_57.80_mse_96.48_psnr_22.89_ssim_0.74.pth'
# __C.RESUME_CC_PATH = './exp/03-14_08-40_SHHA_02_Crowd_SC_0.0001/all_ep_484_mae_61.46_mse_107.36_psnr_23.07_ssim_0.78.pth'
# __C.RESUME_CC_PATH = './exp/03-28_15-11_SHHA_02_Crowd_Baseline_Seg_0.0001/all_ep_491_mae_5.54_mse_14.66_psnr_18.68_ssim_0.65.pth'
# __C.RESUME_CC_PATH = './exp/04-14_11-43_SHHA_02_Crowd_Baseline_UNet_0.0001/all_ep_145_mae_6.02_mse_15.29_psnr_19.79_ssim_0.64.pth'
# __C.RESUME_CC_PATH = './exp/04-15_11-09_SHHA_02_Crowd_Baseline_UNet_Seg_0.0001/all_ep_81_mae_6.01_mse_15.36_psnr_20.14_ssim_0.63.pth'
# __C.RESUME_CC_PATH = './exp/06-02_05-29_SHHA_02_CSRNet_0.0001/all_ep_288_mae_70.65_mse_116.52_psnr_20.71_ssim_0.65.pth'
__C.RESUME_CC_PATH = './exp/07-23_02-37_SHHA_02_Crowd_CSRNet_0.0001/all_ep_609_mae_67.03_mse_108.22_psnr_21.40_ssim_0.76.pth'
# __C.RESUME_CC_PATH = './exp/07-22_15-35_SHHA_02_Crowd_CSRNet_0.0001/all_ep_737_mae_58.93_mse_97.12_psnr_21.82_ssim_0.78.pth'
# __C.RESUME_CC_PATH = './exp/07-19_10-35_SHHA_02_Crowd_CSRNet_0.0001/all_ep_281_mae_66.31_mse_112.32_psnr_20.41_ssim_0.70.pth'
# __C.RESUME_CC_PATH = './exp/07-19_10-35_SHHA_02_Crowd_CSRNet_0.0001/all_ep_483_mae_58.55_mse_98.43_psnr_20.66_ssim_0.71.pth'
# __C.RESUME_CC_PATH = './exp/03-12_16-07_SHHA_02_Crowd_SC_0.0001/all_ep_451_mae_5.89_mse_15.22_psnr_17.91_ssim_0.59.pth'
# __C.RESUME_CC_PATH = './exp/03-23_16-35_SHHA_02_Crowd_SC_0.0001/all_ep_697_mae_5.95_mse_15.20_psnr_18.23_ssim_0.60.pth'
# __C.RESUME_CC_PATH = './exp/03-26_14-31_SHHA_02_Crowd_Baseline_02_0.0001/all_ep_425_mae_5.84_mse_14.82_psnr_20.04_ssim_0.67.pth'
__C.RESUME_Class_PATH = './exp/03-21_08-09_SHHA_02_Crowd_Class_0.0001/all_ep_84_acc_78.00.pth'
# __C.RESUME_CC_PATH = './exp/03-20_16-28_SHHA_02_Crowd_SC_06_1e-06/all_ep_11_mae_24.49_mse_40.65_psnr_20.42_ssim_0.72.pth'
__C.RESUME_SCALE_PATH = '../crowd_seg/exp/02-23_04-03_SHHA_01_Crowd_MSB_Seg_0.0001/all_ep_54_loss_0.25.pth'
__C.BL_LOSS = False
__C.GPU_ID = [3]  # sigle gpu: [0], [1] ...; multi gpus: [0,1]
__C.mst = False

###############################################
__C.data_trans = 0

if __C.CCNET == 'Crowd_MSB':
    __C.batch_num = 6
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.mst = False
if __C.CCNET == 'Crowd_SC':
    __C.batch_num = 1
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.mst = False
if __C.CCNET == 'Crowd_SC_02':
    __C.batch_num = 24
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.mst = False
if __C.CCNET == 'Crowd_SC_06':
    __C.batch_num = 16
    __C.data_trans = 0
    __C.BL_LOSS = False
    __C.mst = False
    __C.DEN_CLASS = True
# learning rate settings
__C.LR = 1e-4  # learning rate
__C.LR_DECAY = 0.995  # decay rate
__C.LR_DECAY_START = -1  # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1  # decay frequency
__C.MAX_EPOCH = 1000

# multi-task learning weights, no use for single model, such as MCNN, VGG, VGG_DECODER, Res50, CSRNet, and so on

__C.LAMBDA_1 = 1e-4  # SANet:0.001 CMTL 0.0001

# print
__C.PRINT_FREQ = 10

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
               + '_' + __C.DATASET \
               + '_' + __C.CCNET \
               + '_' + str(__C.LR)

if __C.DATASET == 'UCF50':
    __C.EXP_NAME += '_' + str(__C.VAL_INDEX)

if __C.DATASET == 'GCC':
    __C.EXP_NAME += '_' + __C.VAL_MODE

__C.EXP_PATH = './exp'  # the path of logs, checkpoints, and current codes

# ------------------------------VAL------------------------
__C.VAL_DENSE_START = 50
__C.VAL_FREQ = 10  # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

# ------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1  # must be 1 for training images with the different sizes

# ================================================================================
# ================================================================================
# ================================================================================
