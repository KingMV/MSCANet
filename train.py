import os
import numpy as np
import torch

from config import cfg

# ------------prepare enviroment------------
seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus) == 1:
    torch.cuda.set_device(gpus[0])
#     # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus[0])

torch.backends.cudnn.benchmark = True

# ------------prepare data loader------------
data_mode = cfg.DATASET
if data_mode is 'SHHA':
    from datasets.SHHA.loading_data import loading_data
    from datasets.SHHA.setting import cfg_data
elif data_mode is 'SHHA_01':
    from datasets.SHHA_01.setting import cfg_data
elif data_mode is 'SHHA_02':
    from datasets.SHHA_02.loading_data import loading_data
    from datasets.SHHA_02.setting import cfg_data
elif data_mode is 'SHHB':
    from datasets.SHHB.loading_data import loading_data
    from datasets.SHHB.setting import cfg_data
elif data_mode is 'QNRF':
    from datasets.QNRF.loading_data import loading_data
    from datasets.QNRF.setting import cfg_data
elif data_mode is 'UCF50':
    from datasets.UCF50.loading_data import loading_data, loading_data_custom
    from datasets.UCF50.setting import cfg_data
elif data_mode is 'WE':
    from datasets.WE.loading_data import loading_data
    from datasets.WE.setting import cfg_data
elif data_mode is 'GCC':
    from datasets.GCC.loading_data import loading_data
    from datasets.GCC.setting import cfg_data
elif data_mode is 'Mall':
    from datasets.Mall.loading_data import loading_data
    from datasets.Mall.setting import cfg_data
elif data_mode is 'UCSD':
    from datasets.UCSD.loading_data import loading_data
    from datasets.UCSD.setting import cfg_data

# ------------Start Training------------
pwd = os.path.split(os.path.realpath(__file__))[0]

cc_trainer = None
if cfg.CC_NET == 'Crowd_MRCM':
    from trainer import Trainer_CrowdCounter
    from datasets.SHHA_02.loading_data import loading_test_group_image, loading_data, loading_test_whole_image

    # train_loader_01, _, _ = loading_sc_train_data(mode='all', patch_mode='class_224')
    train_loader_01, _, _ = loading_data()
    # train_loader_02, _, _ = loading_sc_train_data(mode='all', patch_mode='class_224')
    _, test_loader, _ = loading_test_whole_image()
    # _, test_loader, _ = loading_test_group_image(group='all', patch_mode='class_224',
    #                                              bn_size=20)  # only evaluate hc mae and mse
    loading_data_list = [train_loader_01, None, test_loader]
    cc_trainer = Trainer_CrowdCounter(loading_data_list, cfg_data, pwd)
elif cfg.CC_NET == 'Crowd_MRCM_AB':
    from trainer import Trainer_CrowdCounter
    from datasets.SHHA_02.loading_data import loading_test_group_image, loading_data, loading_test_whole_image

    # train_loader_01, _, _ = loading_sc_train_data(mode='all', patch_mode='class_224')
    train_loader_01, _, _ = loading_data()
    # train_loader_02, _, _ = loading_sc_train_data(mode='all', patch_mode='class_224')
    _, test_loader, _ = loading_test_whole_image()
    # _, test_loader, _ = loading_test_group_image(group='all', patch_mode='class_224',
    #                                              bn_size=20)  # only evaluate hc mae and mse
    loading_data_list = [train_loader_01, None, test_loader]
    cc_trainer = Trainer_CrowdCounter(loading_data_list, cfg_data, pwd)

elif cfg.CC_NET == 'CSRNet':
    from trainer import Trainer_CrowdCounter
    from datasets.SHHA_02.loading_data import loading_data, loading_test_whole_image

    train_loader, test_loader, _ = loading_data()
    # train_loader, test_loader, _ = loading_data_custom()
    loading_data_list = [train_loader, None, test_loader]
    cc_trainer = Trainer_CrowdCounter(loading_data_list, cfg_data, pwd)

elif cfg.CC_NET == 'Crowd_CSRNet' or cfg.CC_NET == 'Crowd_UNet' or cfg.CC_NET == 'Crowd_SAM' \
        or cfg.CC_NET == 'Crowd_SAM_UNet' or cfg.CC_NET == 'Crowd_trident' or cfg.CC_NET == 'Crowd_DSAM' \
        or cfg.CC_NET == 'Crowd_DDSAM' or cfg.CC_NET == 'Crowd_DSAM_UNet' or cfg.CC_NET == 'Crowd_DDSAM_UNet' \
        or cfg.CC_NET == 'Crowd_MSB' or cfg.CC_NET == 'SANet' or cfg.CC_NET == 'Res101_SFCN' \
        or cfg.CC_NET == 'Crowd_PSNet' or cfg.CC_NET == 'Crowd_MCNN':
    from trainer import Trainer_CrowdCounter

    train_loader, test_loader, _ = loading_data()
    # train_loader, test_loader, _ = loading_data_custom()
    loading_data_list = [train_loader, None, test_loader]
    cc_trainer = Trainer_CrowdCounter(loading_data_list, cfg_data, pwd)

elif cfg.CC_NET == 'Crowd_Seg_UNet' or cfg.CC_NET == 'Crowd_Seg' or cfg.CC_NET == 'Crowd_Seg_Decoder' \
        or cfg.CC_NET == 'Crowd_MSPNet':
    from trainer import Trainer_CrowdCounter

    # from datasets.SHHA_02.loading_data import loading_data

    train_loader, test_loader, _ = loading_data()
    # train_loader, test_loader, _ = loading_data_custom()
    loading_data_list = [train_loader, None, test_loader]
    cc_trainer = Trainer_CrowdCounter(loading_data_list, cfg_data, pwd)

cc_trainer.forward()
