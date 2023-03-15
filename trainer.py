import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from models.CC import CrowdCounter, CrowdSegmentation, CrowdClass, Crowd_Seg_Counter, Crowd_Scale_Counter
from config import cfg
from misc.utils import *
import pdb
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr


class Trainer_CC():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.NET
        self.net_seg_name = cfg.SEGNET
        self.seg_net = CrowdSegmentation(cfg.GPU_ID, self.net_seg_name).cuda()
        self.crowd_net = CrowdCounter(cfg.GPU_ID, self.net_name).cuda()

        self.optimizer_1 = optim.Adam(self.seg_net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.optimizer_2 = optim.Adam(self.crowd_net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.scheduler_1 = StepLR(self.optimizer_1, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)
        self.scheduler_2 = StepLR(self.optimizer_2, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}

        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.epoch = 0
        self.i_tb = 0
        self.train_loader, self.val_loader, self.restore_transform = dataloader()
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler_1.step()
                if epoch > 100:
                    self.scheduler_2.step()
            # training    
            self.timer['train time'].tic()
            self.train(epoch)
            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHA_01', 'SHHB', 'QNRF', 'UCF50']:
                    if epoch > 100:
                        self.validate_V1()
                elif self.data_mode is 'WE':
                    self.validate_V2()
                elif self.data_mode is 'GCC':
                    self.validate_V3()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train(self, e):  # training for all datasets
        self.seg_net.train()
        self.crowd_net.train()

        for i, data in enumerate(self.train_loader):
            self.timer['iter time'].tic()
            seg_map = None
            if cfg.add_seg_mask:
                img, gt_map, seg_map = data
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                seg_map = Variable(seg_map).cuda()
            else:
                img, gt_map = data
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
            self.optimizer_1.zero_grad()
            self.optimizer_2.zero_grad()
            seg_loss = torch.FloatTensor([0]).cuda()
            den_loss = torch.FloatTensor([0]).cuda()

            # segnet
            seg_map = self.seg_net(img, seg_map)
            # masked _image
            if e < 100 and cfg.step_train:
                seg_loss = self.seg_net.loss
                seg_loss.backward()
                self.optimizer_1.step()
            else:
                img = seg_map * img
                # crowd_net
                pred_map = self.crowd_net(img, gt_map)
                den_loss = self.crowd_net.loss
                den_loss.backward()
                self.optimizer_2.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.timer['iter time'].toc(average=False)
                print('[ep %d][it %d][seg_loss %.4f den_loss %.4f][lr_1 %.4f lr_2 %.4f][%.2fs]' % \
                      (self.epoch + 1, i + 1, seg_loss.item(), den_loss.item(),
                       self.optimizer_1.param_groups[0]['lr'] * 10000, self.optimizer_2.param_groups[0]['lr'] * 10000,
                       self.timer['iter time'].diff))
                if e > 100:
                    print('        [cnt: gt: %.1f pred: %.2f]' % (
                        gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0].sum().data / self.cfg_data.LOG_PARA))

    def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50

        self.seg_net.eval()
        self.crowd_net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        ssims = AverageMeter()
        psnrs = AverageMeter()
        for vi, data in enumerate(self.val_loader):
            seg_map = None
            with torch.no_grad():
                if cfg.add_seg_mask:
                    img, gt_map, seg_map = data
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()
                    seg_map = Variable(seg_map).cuda()
                else:
                    img, gt_map = data
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()
                # segnet
                seg_map = self.seg_net.forward(img, seg_map)
                seg_loss = self.seg_net.loss
                # masked _image
                img = seg_map * img
                # crowd_net
                pred_map = self.crowd_net.forward(img, gt_map)
                den_loss = self.crowd_net.loss
                ####################################
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()
                loss = seg_loss + den_loss

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    pred_normalized_map = np.squeeze(pred_map[i_img] / np.max(pred_map[i_img] + 1e-20))
                    gt_normalized_map = np.squeeze(gt_map[i_img] / np.max(gt_map[i_img] + 1e-20))

                    s = ssim(gt_normalized_map, pred_normalized_map)
                    p = psnr(gt_normalized_map, pred_normalized_map)
                    ssims.update(s)
                    psnrs.update(p)
                    losses.update(loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        psnr_value = psnrs.avg
        ssim_value = ssims.avg

        self.train_record = update_model([self.seg_net, self.crowd_net], [self.optimizer_1, self.optimizer_2],
                                         [self.scheduler_1, self.scheduler_2], self.epoch,
                                         self.i_tb, self.exp_path,
                                         self.exp_name, [mae, mse, loss, psnr_value, ssim_value], self.train_record)

        print_summary(self.exp_name, [mae, mse, loss, psnr_value, ssim_value], self.train_record)
        # print('*' * 20)
        # print('[best] [model: %s] , [val_loss %.2f]' % (self.train_record['best_model_name'], \
        #                                                 self.train_record['best_loss']))
        # print('*' * 20)


class Trainer_Seg_CrowdCounter():
    def __init__(self, dataloader, cfg_data, pwd, iterative_train=False):
        self.iterative_train = iterative_train
        self.cfg_data = cfg_data
        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.CC_NET
        self.crowd_net = Crowd_Seg_Counter(cfg.GPU_ID, self.net_name).cuda()

        self.optimizer_crowd_net = optim.Adam(self.crowd_net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.scheduler_crowd_net = StepLR(self.optimizer_crowd_net, step_size=cfg.NUM_EPOCH_LR_DECAY,
                                          gamma=cfg.LR_DECAY)

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.epoch = 0
        self.i_tb = 0
        self.train_loader, self.train_loader_01, self.val_loader = dataloader
        if cfg.RESUME:
            print('load the pretrained model')
            latest_state = torch.load(cfg.RESUME_PATH)
            self.crowd_net.load_state_dict(latest_state['net'])
            self.optimizer_crowd_net.load_state_dict(latest_state['optimizer'])
            self.scheduler_crowd_net.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler_crowd_net.step()
            # training
            print('train class_224 dataset')
            self.timer['train time'].tic()
            # self.train(self.train_loader_01)
            self.train(self.train_loader)
            if self.iterative_train:
                if random.random() < 0.5:
                    self.train(self.train_loader_01)
            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHA_01', 'SHHA_02', 'SHHB', 'QNRF', 'UCF50']:
                    self.validate_V1()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train(self, data_loader):  # training for all datasets
        self.crowd_net.train()

        for i, data in enumerate(data_loader):
            self.timer['iter time'].tic()
            # img, gt_map, _, gt_seg_map = data
            img, gt_map, _, gt_seg_map = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            gt_seg_map = Variable(gt_seg_map).cuda()
            self.optimizer_crowd_net.zero_grad()

            pred_map, seg_map = self.crowd_net(img, gt_map, gt_seg_map)
            den_loss, seg_loss = self.crowd_net.get_loss()
            loss = den_loss + 1.0 * seg_loss
            loss.backward()

            self.optimizer_crowd_net.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.timer['iter time'].toc(average=False)
                print('[ep %d][it %d][ loss %.4f den_loss %.4f seg_loss %.4f][lr_1 %.4f ][%.2fs]' % \
                      (self.epoch + 1, i + 1, loss.item(), den_loss.item(), seg_loss.item(),
                       self.optimizer_crowd_net.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                    gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0].sum().data / self.cfg_data.LOG_PARA))

    def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50
        '''

        :return:
        '''
        self.crowd_net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        ssims = AverageMeter()
        psnrs = AverageMeter()
        for vi, data in enumerate(self.val_loader):
            with torch.no_grad():
                img, gt_map, _, gt_seg_map = data
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                gt_seg_map = Variable(gt_seg_map).cuda()
                # crowd_net
                pred_map, seg_map = self.crowd_net.forward(img, gt_map, gt_seg_map, is_train=False)
                den_loss, seg_loss = self.crowd_net.get_loss()
                ####################################
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    pred_normalized_map = np.squeeze(pred_map[i_img] / np.max(pred_map[i_img] + 1e-20))
                    gt_normalized_map = np.squeeze(gt_map[i_img] / np.max(gt_map[i_img] + 1e-20))

                    s = ssim(gt_normalized_map, pred_normalized_map)
                    p = psnr(gt_normalized_map, pred_normalized_map)
                    ssims.update(s)
                    psnrs.update(p)
                    losses.update(den_loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        psnr_value = psnrs.avg
        ssim_value = ssims.avg

        self.train_record = update_crowd_model(self.crowd_net, self.optimizer_crowd_net,
                                               self.scheduler_crowd_net, self.epoch,
                                               self.i_tb, self.exp_path,
                                               self.exp_name, [mae, mse, loss, psnr_value, ssim_value],
                                               self.train_record)

        print_summary(self.exp_name, [mae, mse, loss, psnr_value, ssim_value], self.train_record)

        print('the den_loss is {0:.4f} and the seg_loss is {1:.4f}'.format(den_loss.item(), seg_loss.item()))


class Trainer_Scale_CrowdCounter():
    def __init__(self, dataloader, cfg_data, pwd, iterative_train=False):
        self.iterative_train = iterative_train
        self.cfg_data = cfg_data
        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.CC_NET
        self.crowd_net = Crowd_Seg_Counter(cfg.GPU_ID, self.net_name).cuda()

        self.optimizer_crowd_net = optim.Adam(self.crowd_net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.scheduler_crowd_net = StepLR(self.optimizer_crowd_net, step_size=cfg.NUM_EPOCH_LR_DECAY,
                                          gamma=cfg.LR_DECAY)

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.epoch = 0
        self.i_tb = 0
        self.train_loader, self.train_loader_01, self.val_loader = dataloader
        if cfg.RESUME:
            print('load the pretrained model')
            latest_state = torch.load(cfg.RESUME_PATH)
            self.crowd_net.load_state_dict(latest_state['net'])
            self.optimizer_crowd_net.load_state_dict(latest_state['optimizer'])
            self.scheduler_crowd_net.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler_crowd_net.step()
            # training
            print('train class_224 dataset')
            self.timer['train time'].tic()
            # self.train(self.train_loader_01)
            self.train(self.train_loader)
            if self.iterative_train:
                if random.random() < 0.5:
                    self.train(self.train_loader_01)
            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHA_01', 'SHHA_02', 'SHHB', 'QNRF', 'UCF50']:
                    self.validate_V1()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train(self, data_loader):  # training for all datasets
        self.crowd_net.train()

        for i, data in enumerate(data_loader):
            self.timer['iter time'].tic()
            # img, gt_map, _, gt_seg_map = data
            img, gt_map, _, gt_seg_map = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            gt_seg_map = Variable(gt_seg_map).cuda()
            self.optimizer_crowd_net.zero_grad()

            pred_map, seg_map = self.crowd_net(img, gt_map, gt_seg_map)
            den_loss, seg_loss = self.crowd_net.get_loss()
            loss = den_loss + 1.0 * seg_loss
            loss.backward()

            self.optimizer_crowd_net.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.timer['iter time'].toc(average=False)
                print('[ep %d][it %d][ loss %.4f den_loss %.4f seg_loss %.4f][lr_1 %.4f ][%.2fs]' % \
                      (self.epoch + 1, i + 1, loss.item(), den_loss.item(), seg_loss.item(),
                       self.optimizer_crowd_net.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                    gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0].sum().data / self.cfg_data.LOG_PARA))

    def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50
        '''

        :return:
        '''
        self.crowd_net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        ssims = AverageMeter()
        psnrs = AverageMeter()
        for vi, data in enumerate(self.val_loader):
            with torch.no_grad():
                # img, gt_map, _, gt_seg_map = data
                img, gt_map = data
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                # gt_seg_map = Variable(gt_seg_map).cuda()
                # crowd_net
                pred_map, seg_map = self.crowd_net.forward(img, gt_map, None, is_train=False)
                den_loss, seg_loss = self.crowd_net.get_loss()
                ####################################
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    pred_normalized_map = np.squeeze(pred_map[i_img] / np.max(pred_map[i_img] + 1e-20))
                    gt_normalized_map = np.squeeze(gt_map[i_img] / np.max(gt_map[i_img] + 1e-20))

                    s = ssim(gt_normalized_map, pred_normalized_map)
                    p = psnr(gt_normalized_map, pred_normalized_map)
                    ssims.update(s)
                    psnrs.update(p)
                    losses.update(den_loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        psnr_value = psnrs.avg
        ssim_value = ssims.avg

        self.train_record = update_crowd_model(self.crowd_net, self.optimizer_crowd_net,
                                               self.scheduler_crowd_net, self.epoch,
                                               self.i_tb, self.exp_path,
                                               self.exp_name, [mae, mse, loss, psnr_value, ssim_value],
                                               self.train_record)

        print_summary(self.exp_name, [mae, mse, loss, psnr_value, ssim_value], self.train_record)

        print('the den_loss is {0:.4f} and the seg_loss is {1:.4f}'.format(den_loss.item(), seg_loss.item()))


class Trainer_Scale_01():
    def __init__(self, dataloader, cfg_data, pwd, iterative_train=False):
        self.iterative_train = iterative_train
        self.cfg_data = cfg_data
        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.CC_NET
        self.crowd_net = Crowd_Scale_Counter(cfg.GPU_ID, self.net_name).cuda()

        self.optimizer_crowd_net = optim.Adam(self.crowd_net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.scheduler_crowd_net = StepLR(self.optimizer_crowd_net, step_size=cfg.NUM_EPOCH_LR_DECAY,
                                          gamma=cfg.LR_DECAY)

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.epoch = 0
        self.i_tb = 0
        self.train_loader, self.train_loader_01, self.val_loader = dataloader
        if cfg.RESUME:
            print('load the pretrained model')
            latest_state = torch.load(cfg.RESUME_PATH)
            self.crowd_net.load_state_dict(latest_state['net'])
            self.optimizer_crowd_net.load_state_dict(latest_state['optimizer'])
            self.scheduler_crowd_net.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler_crowd_net.step()
            # training
            print('train class_224 dataset')
            self.timer['train time'].tic()
            # self.train(self.train_loader_01)
            self.train(self.train_loader)
            if self.iterative_train:
                if random.random() < 0.5:
                    self.train(self.train_loader_01)
            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHA_01', 'SHHA_02', 'SHHB', 'QNRF', 'UCF50']:
                    self.validate_V1()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train(self, data_loader):  # training for all datasets
        self.crowd_net.train()

        for i, data in enumerate(data_loader):
            self.timer['iter time'].tic()
            # img, gt_map, _, gt_seg_map = data
            img, gt_map, _, gt_seg_map = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            gt_seg_map = Variable(gt_seg_map).cuda()
            self.optimizer_crowd_net.zero_grad()

            pred_map, seg_map = self.crowd_net(img, gt_map, gt_seg_map)
            den_loss, seg_loss = self.crowd_net.get_loss()
            loss = den_loss + 1.0 * seg_loss
            loss.backward()

            self.optimizer_crowd_net.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.timer['iter time'].toc(average=False)
                print('[ep %d][it %d][ loss %.4f den_loss %.4f seg_loss %.4f][lr_1 %.4f ][%.2fs]' % \
                      (self.epoch + 1, i + 1, loss.item(), den_loss.item(), seg_loss.item(),
                       self.optimizer_crowd_net.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                    gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0].sum().data / self.cfg_data.LOG_PARA))

    def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50
        '''

        :return:
        '''
        self.crowd_net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        ssims = AverageMeter()
        psnrs = AverageMeter()
        for vi, data in enumerate(self.val_loader):
            with torch.no_grad():
                # img, gt_map, _, gt_seg_map = data
                img, gt_map = data
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                # gt_seg_map = Variable(gt_seg_map).cuda()
                # crowd_net
                pred_map, seg_map = self.crowd_net.forward(img, gt_map, None, is_train=False)
                den_loss, seg_loss = self.crowd_net.get_loss()
                ####################################
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    pred_normalized_map = np.squeeze(pred_map[i_img] / np.max(pred_map[i_img] + 1e-20))
                    gt_normalized_map = np.squeeze(gt_map[i_img] / np.max(gt_map[i_img] + 1e-20))

                    s = ssim(gt_normalized_map, pred_normalized_map)
                    p = psnr(gt_normalized_map, pred_normalized_map)
                    ssims.update(s)
                    psnrs.update(p)
                    losses.update(den_loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        psnr_value = psnrs.avg
        ssim_value = ssims.avg

        self.train_record = update_crowd_model(self.crowd_net, self.optimizer_crowd_net,
                                               self.scheduler_crowd_net, self.epoch,
                                               self.i_tb, self.exp_path,
                                               self.exp_name, [mae, mse, loss, psnr_value, ssim_value],
                                               self.train_record)

        print_summary(self.exp_name, [mae, mse, loss, psnr_value, ssim_value], self.train_record)

        print('the den_loss is {0:.4f} and the seg_loss is {1:.4f}'.format(den_loss.item(), seg_loss.item()))


class Trainer_CrowdCounter():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data
        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.CC_NET
        self.data_read_mode = 3
        if self.net_name == 'CSRNet' or self.net_name == 'Crowd_CSRNet' or self.net_name == 'Crowd_UNet' \
                or self.net_name == 'Crowd_SAM' or self.net_name == 'Crowd_SAM_UNet' or self.net_name == 'Crowd_DDSAM_UNet' \
                or self.net_name == 'Crowd_trident' or self.net_name == 'Crowd_DSAM' or self.net_name == 'Crowd_DDSAM' \
                or self.net_name == 'Crowd_DSAM_UNet' or self.net_name == 'Crowd_MSB' or self.net_name == 'SANet' \
                or self.net_name == 'Res101_SFCN' or self.net_name == 'Crowd_PSNet' or self.net_name == 'Crowd_MCNN':
            self.data_read_mode = 2
        elif self.net_name == 'Crowd_Seg_UNet' or self.net_name == 'Crowd_Seg' or self.net_name == 'Crowd_Seg_Decoder' \
                or self.net_name == 'Crowd_MSPNet':
            self.data_read_mode = 3

        self.crowd_net = CrowdCounter(cfg.GPU_ID, self.net_name).cuda()
        self.optimizer_crowd_net = optim.Adam(self.crowd_net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.scheduler_crowd_net = StepLR(self.optimizer_crowd_net, step_size=cfg.NUM_EPOCH_LR_DECAY,
                                          gamma=cfg.LR_DECAY)

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}

        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.epoch = 0
        self.i_tb = 0
        self.train_loader, self.train_loader_01, self.val_loader = dataloader
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler_crowd_net.step()
            # training
            self.timer['train time'].tic()
            self.train(self.train_loader)
            # print('train patch_224 dataset')
            # self.train(self.train_loader_01)
            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHA_01', 'SHHA_02', 'SHHB', 'QNRF', 'UCF50']:
                    self.validate_V1()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train(self, data_loader):  # training for all datasets
        self.crowd_net.train()
        losses = AverageMeter()
        for i, data in enumerate(data_loader):
            self.timer['iter time'].tic()
            seg_map = None
            if self.data_read_mode == 3:
                img, gt_map, seg_map = data
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                # if self.net_name == 'Crowd_Seg_Decoder':
                #     seg_map
                seg_map = Variable(seg_map).cuda()
            else:
                img, gt_map = data
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

            self.optimizer_crowd_net.zero_grad()
            if self.data_read_mode == 3:
                pred_map, _ = self.crowd_net(img, gt_map, seg_map)
            else:
                pred_map = self.crowd_net(img, gt_map, seg_map)
            # den_loss, seg_loss = self.crowd_net.get_loss()
            den_loss, seg_loss, d_loss = self.crowd_net.get_all_loss()
            if self.data_read_mode == 3:
                loss = den_loss + 0.01 * seg_loss
            else:
                loss = den_loss + 0.1 * d_loss
            loss.backward()
            losses.update(loss.item())
            self.optimizer_crowd_net.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.timer['iter time'].toc(average=False)
                if self.data_read_mode == 3:
                    print('[ep %d][it %d][ loss %.4f loss_1 %.4f,loss_2 %.4f][lr_1 %.4f ][%.2fs]' % \
                          (self.epoch + 1, i + 1, loss.item(), den_loss.item(), seg_loss.item(),
                           self.optimizer_crowd_net.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff))
                else:
                    print('[ep %d][it %d][ loss %.4f den_loss %.4f][lr_1 %.4f ][%.2fs]' % \
                          (self.epoch + 1, i + 1, loss.item(), den_loss.item(),
                           self.optimizer_crowd_net.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                    gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0].sum().data / self.cfg_data.LOG_PARA))
                print(d_loss.item())
        print('[ep {0}] the training loss:{1} '.format(self.epoch + 1, losses.avg))

    def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50
        '''

        :return:
        '''
        self.crowd_net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        patch_maes = AverageMeter()
        mses = AverageMeter()
        patch_mses = AverageMeter()
        ssims = AverageMeter()
        psnrs = AverageMeter()
        cos_d = AverageMeter()
        for vi, data in enumerate(self.val_loader):
            with torch.no_grad():
                # img, gt_map, _ = data
                if self.data_read_mode == 3:
                    img, gt_map, _ = data
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()
                else:
                    img, gt_map = data
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()
                # crowd_net
                if self.data_read_mode == 3:
                    pred_map, _ = self.crowd_net(img, gt_map, is_train=False)
                else:
                    pred_map = self.crowd_net(img, gt_map, is_train=False)

                loss, _, d_loss = self.crowd_net.get_all_loss()
                ####################################
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()
                cos_d.update(d_loss.item())
                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    pmae, pmse = caluate_game(np.squeeze(pred_map[i_img]), np.squeeze(gt_map[i_img]), L=4)

                    patch_maes.update(pmae)
                    patch_mses.update(pmse)

                    pred_normalized_map = np.squeeze(pred_map[i_img] / np.max(pred_map[i_img] + 1e-20))
                    gt_normalized_map = np.squeeze(gt_map[i_img] / np.max(gt_map[i_img] + 1e-20))

                    s = ssim(gt_normalized_map, pred_normalized_map)
                    p = psnr(gt_normalized_map, pred_normalized_map)
                    ssims.update(s)
                    psnrs.update(p)
                    losses.update(loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

        loss = losses.avg
        mae = maes.avg
        patch_mae = patch_maes.avg
        mse = np.sqrt(mses.avg)
        patch_mse = np.sqrt(patch_mses.avg)

        psnr_value = psnrs.avg
        ssim_value = ssims.avg

        self.train_record = update_crowd_model(self.crowd_net, self.optimizer_crowd_net,
                                               self.scheduler_crowd_net, self.epoch,
                                               self.i_tb, self.exp_path,
                                               self.exp_name, [mae, mse, loss, psnr_value, ssim_value],
                                               self.train_record)

        print_summary(self.exp_name, [mae, mse, loss, psnr_value, ssim_value], self.train_record)
        print('patch_mae:{0}\t patch_mse:{1}'.format(patch_mae, patch_mse))
        print('cos_d is {0}'.format(cos_d.avg))


class SC_CrowdCounter():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data
        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.CC_NET
        self.crowd_net = CrowdCounter(cfg.GPU_ID, self.net_name).cuda()
        if cfg.RESUME:
            print('load the pretrained model')
            latest_state = torch.load(cfg.RESUME_PATH, map_location=lambda storage, loc: storage.cuda(cfg.GPU_ID[0]))
            self.crowd_net.load_state_dict(latest_state)

        self.optimizer_crowd_net = optim.Adam(self.crowd_net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.scheduler_crowd_net = StepLR(self.optimizer_crowd_net, step_size=cfg.NUM_EPOCH_LR_DECAY,
                                          gamma=cfg.LR_DECAY)

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}

        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.epoch = 0
        self.i_tb = 0
        self.train_loader, _, self.val_loader = dataloader
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=False)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler_crowd_net.step()
            # training
            self.timer['train time'].tic()
            self.train(epoch)
            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHA_01', 'SHHA_02', 'SHHB', 'QNRF', 'UCF50']:
                    self.validate_V1()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train(self, e):  # training for all datasets
        self.crowd_net.train()

        for i, data in enumerate(self.train_loader):
            self.timer['iter time'].tic()
            img, gt_map, gt_class = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            gt_class = Variable(gt_class).cuda()

            self.optimizer_crowd_net.zero_grad()

            pred_map = self.crowd_net(img, gt_map, gt_class)
            den_loss, den_class_loss = self.crowd_net.get_loss()
            den_class_loss = 0.5 * den_class_loss
            loss = den_loss + den_class_loss
            loss.backward()

            self.optimizer_crowd_net.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.timer['iter time'].toc(average=False)
                print('[ep %d][it %d][ loss %.4f den_loss %.4f den_class_loss %.4f][lr_1 %.4f ][%.2fs]' % \
                      (self.epoch + 1, i + 1, loss.item(), den_loss.item(), den_class_loss.item(),
                       self.optimizer_crowd_net.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                    gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0].sum().data / self.cfg_data.LOG_PARA))

    def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50
        '''

        :return:
        '''
        self.crowd_net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        ssims = AverageMeter()
        psnrs = AverageMeter()
        for vi, data in enumerate(self.val_loader):
            gt_class = None
            with torch.no_grad():
                img, gt_map, _ = data
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                # crowd_net
                pred_map = self.crowd_net.forward(img, gt_map)
                loss, den_class_loss = self.crowd_net.get_loss()
                ####################################
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    pred_normalized_map = np.squeeze(pred_map[i_img] / np.max(pred_map[i_img] + 1e-20))
                    gt_normalized_map = np.squeeze(gt_map[i_img] / np.max(gt_map[i_img] + 1e-20))

                    s = ssim(gt_normalized_map, pred_normalized_map)
                    p = psnr(gt_normalized_map, pred_normalized_map)
                    ssims.update(s)
                    psnrs.update(p)
                    losses.update(loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        psnr_value = psnrs.avg
        ssim_value = ssims.avg

        self.train_record = update_crowd_model(self.crowd_net, self.optimizer_crowd_net,
                                               self.scheduler_crowd_net, self.epoch,
                                               self.i_tb, self.exp_path,
                                               self.exp_name, [mae, mse, loss, psnr_value, ssim_value],
                                               self.train_record)

        print_summary(self.exp_name, [mae, mse, loss, psnr_value, ssim_value], self.train_record)

        print('the den_class_loss is {0:.4f}'.format(den_class_loss.item()))


class SC_High_CrowdCounter():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data
        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.CC_NET
        self.crowd_net = CrowdCounter(cfg.GPU_ID, self.net_name).cuda()
        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH, map_location=lambda storage, loc: storage.cuda(cfg.GPU_ID[0]))
            self.crowd_net.load_state_dict(latest_state)
            print('load the model')
        self.optimizer_crowd_net = optim.Adam(self.crowd_net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.scheduler_crowd_net = StepLR(self.optimizer_crowd_net, step_size=cfg.NUM_EPOCH_LR_DECAY,
                                          gamma=cfg.LR_DECAY)

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}

        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.epoch = 0
        self.i_tb = 0
        self.train_loader, _, self.val_loader = dataloader
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler_crowd_net.step()
            # training
            self.timer['train time'].tic()
            self.train(epoch)
            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHA_01', 'SHHA_02', 'SHHB', 'QNRF', 'UCF50']:
                    self.validate_V1()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train(self, e):  # training for all datasets
        self.crowd_net.train()

        for i, data in enumerate(self.train_loader):
            self.timer['iter time'].tic()
            img, gt_map, gt_class = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            gt_class = Variable(gt_class).cuda()
            self.optimizer_crowd_net.zero_grad()

            pred_map = self.crowd_net(img, gt_map, gt_class)
            den_loss, den_class_loss = self.crowd_net.get_loss()
            loss = den_loss
            loss.backward()

            self.optimizer_crowd_net.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.timer['iter time'].toc(average=False)
                if cfg.DEN_CLASS:
                    print('[ep %d][it %d][ loss %.4f den_loss %.4f den_class_loss %.4f][lr_1 %.4f ][%.2fs]' % \
                          (self.epoch + 1, i + 1, loss.item(), den_loss.item(), den_class_loss.item(),
                           self.optimizer_crowd_net.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                    gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0].sum().data / self.cfg_data.LOG_PARA))

    def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50
        '''

        :return:
        '''
        self.crowd_net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        ssims = AverageMeter()
        psnrs = AverageMeter()
        for vi, data in enumerate(self.val_loader):
            gt_class = None
            with torch.no_grad():

                if cfg.DEN_CLASS:
                    img, gt_map, _ = data
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()
                    gt_class = None
                # crowd_net
                pred_map = self.crowd_net.forward(img, gt_map)
                loss, den_class_loss = self.crowd_net.get_loss()
                ####################################
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    pred_normalized_map = np.squeeze(pred_map[i_img] / np.max(pred_map[i_img] + 1e-20))
                    gt_normalized_map = np.squeeze(gt_map[i_img] / np.max(gt_map[i_img] + 1e-20))

                    s = ssim(gt_normalized_map, pred_normalized_map)
                    p = psnr(gt_normalized_map, pred_normalized_map)
                    ssims.update(s)
                    psnrs.update(p)
                    losses.update(loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        psnr_value = psnrs.avg
        ssim_value = ssims.avg

        self.train_record = update_crowd_model(self.crowd_net, self.optimizer_crowd_net,
                                               self.scheduler_crowd_net, self.epoch,
                                               self.i_tb, self.exp_path,
                                               self.exp_name, [mae, mse, loss, psnr_value, ssim_value],
                                               self.train_record)

        print_summary(self.exp_name, [mae, mse, loss, psnr_value, ssim_value], self.train_record)

        print('the den_class_loss is {0:.4f}'.format(den_class_loss.item()))


class SC02_CrowdCounter():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data
        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.CC_NET
        self.crowd_net = CrowdCounter(cfg.GPU_ID, self.net_name).cuda()
        self.optimizer_crowd_net = optim.Adam(self.crowd_net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.scheduler_crowd_net = StepLR(self.optimizer_crowd_net, step_size=cfg.NUM_EPOCH_LR_DECAY,
                                          gamma=cfg.LR_DECAY)

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}

        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.epoch = 0
        self.i_tb = 0
        self.train_sparse_loader, self.train_hc_loader, self.val_loader = dataloader
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)
        self.lamd_1 = 0.5
        self.lamd_2 = 1
        self.lamd_3 = 0.5

    def forward(self):
        T = 1
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler_crowd_net.step()
            # training
            self.timer['train time'].tic()
            print('train sparse')
            self.train_den_class()

            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHA_01', 'SHHA_02', 'SHHB', 'QNRF', 'UCF50']:
                    self.validate_V1()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train_den_class(self, train_high_density=False):
        self.crowd_net.train()
        # 训练所有正常尺寸图片 decoder采用unet架构
        data_loader = self.train_sparse_loader
        for i, data in enumerate(data_loader):
            self.timer['iter time'].tic()
            img, gt_map, gt_class = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            gt_class = Variable(gt_class).cuda()
            self.optimizer_crowd_net.zero_grad()
            s_map, h_map = self.crowd_net(img, gt_map, gt_class)
            s_loss, h_loss, c_loss = self.crowd_net.get_sc_loss()
            loss = self.lamd_1 * c_loss + self.lamd_2 * s_loss + self.lamd_3 * h_loss
            loss.backward()
            self.optimizer_crowd_net.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.timer['iter time'].toc(average=False)
                if cfg.DEN_CLASS:
                    print('[ep %d][it %d][ loss %.4f s_loss %.4f h_loss %.4f den_class_loss %.4f][lr_1 %.4f ][%.2fs]' % \
                          (self.epoch + 1, i + 1, loss.item(), s_loss.item(), h_loss.item(), c_loss.item(),
                           self.optimizer_crowd_net.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred_s: %.2f pred_h: %.2f]' % (
                    gt_map[0].sum().data / self.cfg_data.LOG_PARA, s_map[0].sum().data / self.cfg_data.LOG_PARA,
                    h_map[0].sum().data / self.cfg_data.LOG_PARA))

    def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50
        self.crowd_net.eval()
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        ssims = AverageMeter()
        psnrs = AverageMeter()
        for vi, data in enumerate(self.val_loader):
            gt_class = None
            with torch.no_grad():
                img, gt_map = data
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                # crowd_net
                # s_map, h_map, density_level = self.crowd_net.test_forward(img, gt_map)
                # 直接测试正常图的性能指标！
                pred_map, gt_map = self.crowd_net.whole_image_forward(img, gt_map)
                loss, _ = self.crowd_net.get_loss()

                ####################################
                # pred_map = pred_map.data.cpu().numpy()
                # gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    pred_normalized_map = np.squeeze(pred_map[i_img] / np.max(pred_map[i_img] + 1e-20))
                    gt_normalized_map = np.squeeze(gt_map[i_img] / np.max(gt_map[i_img] + 1e-20))

                    s = ssim(gt_normalized_map, pred_normalized_map)
                    p = psnr(gt_normalized_map, pred_normalized_map)
                    ssims.update(s)
                    psnrs.update(p)
                    losses.update(loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        psnr_value = psnrs.avg
        ssim_value = ssims.avg

        self.train_record = update_crowd_model(self.crowd_net, self.optimizer_crowd_net,
                                               self.scheduler_crowd_net, self.epoch,
                                               self.i_tb, self.exp_path,
                                               self.exp_name, [mae, mse, loss, psnr_value, ssim_value],
                                               self.train_record)

        print_summary(self.exp_name, [mae, mse, loss, psnr_value, ssim_value], self.train_record)


class Crowd_Classer():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data
        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.CC_NET
        self.crowd_net = CrowdClass(cfg.GPU_ID, self.net_name).cuda()
        self.optimizer_crowd_net = optim.Adam(self.crowd_net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.scheduler_crowd_net = StepLR(self.optimizer_crowd_net, step_size=cfg.NUM_EPOCH_LR_DECAY,
                                          gamma=cfg.LR_DECAY)

        self.train_record = {'best_acc': 0, 'best_model_name': ''}

        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.epoch = 0
        self.i_tb = 0
        self.train_sparse_loader, self.train_hc_loader, self.val_loader = dataloader
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler_crowd_net.step()
            # training
            self.timer['train time'].tic()
            print('train density class')
            self.train_den_class()

            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHA_01', 'SHHA_02', 'SHHB', 'QNRF', 'UCF50']:
                    self.validate_V1()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train_den_class(self, train_high_density=False):
        self.crowd_net.train()
        # 训练所有正常尺寸图片 decoder采用unet架构
        data_loader = self.train_sparse_loader
        total = 0
        correct = 0
        for i, data in enumerate(data_loader):
            self.timer['iter time'].tic()
            img, _, gt_class = data
            img = Variable(img).cuda()
            gt_class = Variable(gt_class).cuda()
            self.optimizer_crowd_net.zero_grad()
            den_pro = self.crowd_net(img, gt_class)
            den_level = torch.argmax(den_pro, dim=1)
            total += gt_class.size(0)
            correct += (gt_class == den_level).sum()
            loss = self.crowd_net.loss
            loss.backward()

            self.optimizer_crowd_net.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.timer['iter time'].toc(average=False)
                if cfg.DEN_CLASS:
                    print('[ep %d][it %d][ loss %.4f][lr_1 %.4f ][%.2fs]' % \
                          (self.epoch + 1, i + 1, loss.item(),
                           self.optimizer_crowd_net.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff))
        accuracy = 100.0 * correct / total
        print('[ep %d] [the train accuracy is %.2f]' % (self.epoch + 1, accuracy))

    def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50
        self.crowd_net.eval()
        total = 0
        correct = 0
        for vi, data in enumerate(self.val_loader):
            with torch.no_grad():
                img, _, gt_class = data
                img = Variable(img).cuda()
                gt_class = Variable(gt_class).cuda()

                pred_class = self.crowd_net(img, gt_class)
                pred_class = torch.argmax(pred_class, dim=1)
                total += gt_class.size(0)
                correct += (pred_class == gt_class).sum()
        accuracy = 100.0 * correct / total
        print('the accuracy is %.2f' % accuracy)
        self.train_record = update_class_model(self.crowd_net, self.optimizer_crowd_net,
                                               self.scheduler_crowd_net, self.epoch,
                                               self.i_tb, self.exp_path,
                                               self.exp_name, accuracy, self.train_record)


class SC02_high_CrowdCounter():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data
        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.CC_NET
        self.crowd_net = CrowdCounter(cfg.GPU_ID, self.net_name).cuda()
        self.optimizer_crowd_net = optim.Adam(self.crowd_net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.scheduler_crowd_net = StepLR(self.optimizer_crowd_net, step_size=cfg.NUM_EPOCH_LR_DECAY,
                                          gamma=cfg.LR_DECAY)

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}

        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.epoch = 0
        self.i_tb = 0
        self.train_sparse_loader, self.train_hc_loader, self.val_loader = dataloader
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)

    def forward(self):
        T = 1
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler_crowd_net.step()
            # training
            self.timer['train time'].tic()
            if epoch % T == 0:
                print('train sparse and high')
                self.train_den_class(train_high_density=False)

            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHA_01', 'SHHA_02', 'SHHB', 'QNRF', 'UCF50']:
                    self.validate_V1()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train_den_class(self, train_high_density=False):
        self.crowd_net.train()

        if train_high_density:
            data_loader = self.train_hc_loader
        else:
            data_loader = self.train_sparse_loader
        for i, data in enumerate(data_loader):
            self.timer['iter time'].tic()
            img, gt_map, gt_class = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            gt_class = Variable(gt_class).cuda()
            self.optimizer_crowd_net.zero_grad()

            s_map, h_map = self.crowd_net(img, gt_map, gt_class)
            s_loss, h_loss, c_loss = self.crowd_net.get_sc_loss()

            # process the loss
            pred_map = h_map + s_map
            den_loss = h_loss + s_loss
            den_class_loss = c_loss
            loss = den_loss + den_class_loss

            loss.backward()
            self.optimizer_crowd_net.step()
            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.timer['iter time'].toc(average=False)
                if cfg.DEN_CLASS:
                    print(
                        '[ep %d][it %d][ loss %.4f sparse_loss %.4f high_loss %.4f den_class_loss %.4f][lr_1 %.4f ][%.2fs]' % \
                        (self.epoch + 1, i + 1, loss.item(), s_loss.item(), h_loss.item(),
                         den_class_loss.item(), self.optimizer_crowd_net.param_groups[0]['lr'] * 10000,
                         self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                    gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0].sum().data / self.cfg_data.LOG_PARA))

    def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50
        '''

        :return:
        '''
        self.crowd_net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        ssims = AverageMeter()
        psnrs = AverageMeter()
        for vi, data in enumerate(self.val_loader):
            gt_class = None
            with torch.no_grad():
                if cfg.add_seg_mask:
                    img, gt_map, seg_map = data
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()
                    seg_map = Variable(seg_map).cuda()
                else:
                    if cfg.DEN_CLASS:
                        img, gt_map = data
                        img = Variable(img).cuda()
                        gt_map = Variable(gt_map).cuda()
                        gt_class = None
                    else:
                        img, gt_map = data
                        img = Variable(img).cuda()
                        gt_map = Variable(gt_map).cuda()
                # crowd_net
                s_map, h_map, density_level = self.crowd_net.test_forward(img, gt_map)
                s_loss, h_loss, c_loss = self.crowd_net.get_sc_loss()
                density_level = torch.argmax(density_level).item()
                if density_level == 3:
                    pred_map = h_map
                    loss = h_loss
                else:
                    pred_map = s_map
                    loss = s_loss
                den_class_loss = c_loss
                # loss, den_class_loss = self.crowd_net.get_loss()

                ####################################
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    pred_normalized_map = np.squeeze(pred_map[i_img] / np.max(pred_map[i_img] + 1e-20))
                    gt_normalized_map = np.squeeze(gt_map[i_img] / np.max(gt_map[i_img] + 1e-20))

                    s = ssim(gt_normalized_map, pred_normalized_map)
                    p = psnr(gt_normalized_map, pred_normalized_map)
                    ssims.update(s)
                    psnrs.update(p)
                    losses.update(loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        psnr_value = psnrs.avg
        ssim_value = ssims.avg

        self.train_record = update_crowd_model(self.crowd_net, self.optimizer_crowd_net,
                                               self.scheduler_crowd_net, self.epoch,
                                               self.i_tb, self.exp_path,
                                               self.exp_name, [mae, mse, loss, psnr_value, ssim_value],
                                               self.train_record)

        print_summary(self.exp_name, [mae, mse, loss, psnr_value, ssim_value], self.train_record)

        print('the den_class_loss is {0:.4f}'.format(den_class_loss.item()))


class SC04_CrowdCounter():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data
        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.CC_NET
        self.crowd_net = CrowdCounter(cfg.GPU_ID, self.net_name).cuda()
        self.optimizer_crowd_net = optim.Adam(self.crowd_net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.scheduler_crowd_net = StepLR(self.optimizer_crowd_net, step_size=cfg.NUM_EPOCH_LR_DECAY,
                                          gamma=cfg.LR_DECAY)
        self.epoch = 0
        self.i_tb = 0
        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.crowd_net.load_state_dict(latest_state['net'])
            self.optimizer_crowd_net.load_state_dict(latest_state['optimizer'])
            self.scheduler_crowd_net.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}

        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

        self.train_sparse_loader, self.train_hc_loader, self.val_loader = dataloader
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler_crowd_net.step()
            # training
            self.timer['train time'].tic()
            print('train sc04 all density and concat the output')
            self.train_den_class()

            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHA_01', 'SHHA_02', 'SHHB', 'QNRF', 'UCF50']:
                    self.validate_V1()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train_den_class(self, train_high_density=False):
        self.crowd_net.train()
        # 训练所有正常尺寸图片 decoder采用unet架构
        data_loader = self.train_sparse_loader
        for i, data in enumerate(data_loader):
            self.timer['iter time'].tic()
            img, gt_map, gt_class = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            gt_class = Variable(gt_class).cuda()
            self.optimizer_crowd_net.zero_grad()
            pred_map = self.crowd_net(img, gt_map, gt_class)
            den_loss, den_class_loss = self.crowd_net.get_loss()
            loss = den_loss + 0.5 * den_class_loss
            loss.backward()
            self.optimizer_crowd_net.step()
            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.timer['iter time'].toc(average=False)
                if cfg.DEN_CLASS:
                    print('[ep %d][it %d][ loss %.4f den_loss %.4f den_class_loss %.4f][lr_1 %.4f ][%.2fs]' % \
                          (self.epoch + 1, i + 1, loss.item(), den_loss.item(), den_class_loss.item(),
                           self.optimizer_crowd_net.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                    gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0].sum().data / self.cfg_data.LOG_PARA))

    def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50
        self.crowd_net.eval()
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        ssims = AverageMeter()
        psnrs = AverageMeter()
        for vi, data in enumerate(self.val_loader):
            gt_class = None
            with torch.no_grad():
                img, gt_map = data
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                # crowd_net
                # s_map, h_map, density_level = self.crowd_net.test_forward(img, gt_map)
                # 直接测试正常图的性能指标！
                pred_map, gt_map = self.crowd_net.csc_04_forward(img, gt_map)
                loss, _ = self.crowd_net.get_loss()

                ####################################
                # pred_map = pred_map.data.cpu().numpy()
                # gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    pred_normalized_map = np.squeeze(pred_map[i_img] / np.max(pred_map[i_img] + 1e-20))
                    gt_normalized_map = np.squeeze(gt_map[i_img] / np.max(gt_map[i_img] + 1e-20))

                    s = ssim(gt_normalized_map, pred_normalized_map)
                    p = psnr(gt_normalized_map, pred_normalized_map)
                    ssims.update(s)
                    psnrs.update(p)
                    losses.update(loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        psnr_value = psnrs.avg
        ssim_value = ssims.avg

        self.train_record = update_crowd_model(self.crowd_net, self.optimizer_crowd_net,
                                               self.scheduler_crowd_net, self.epoch,
                                               self.i_tb, self.exp_path,
                                               self.exp_name, [mae, mse, loss, psnr_value, ssim_value],
                                               self.train_record)

        print_summary(self.exp_name, [mae, mse, loss, psnr_value, ssim_value], self.train_record)


class SC05_CrowdCounter():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data
        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.CC_NET
        self.crowd_net = CrowdCounter(cfg.GPU_ID, self.net_name).cuda()
        self.optimizer_crowd_net = optim.Adam(self.crowd_net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.scheduler_crowd_net = StepLR(self.optimizer_crowd_net, step_size=cfg.NUM_EPOCH_LR_DECAY,
                                          gamma=cfg.LR_DECAY)
        self.epoch = 0
        self.i_tb = 0
        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.crowd_net.load_state_dict(latest_state['net'])
            self.optimizer_crowd_net.load_state_dict(latest_state['optimizer'])
            self.scheduler_crowd_net.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}

        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

        self.train_sparse_loader, self.train_hc_loader, self.val_loader = dataloader
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler_crowd_net.step()
            # training
            self.timer['train time'].tic()
            print('train sc05 all density and concat the output')
            self.train_den_class()

            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHA_01', 'SHHA_02', 'SHHB', 'QNRF', 'UCF50']:
                    self.validate_V1()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train_den_class(self, train_high_density=False):
        self.crowd_net.train()
        # 训练所有正常尺寸图片 decoder采用unet架构
        data_loader = self.train_sparse_loader
        for i, data in enumerate(data_loader):
            self.timer['iter time'].tic()
            img, gt_map, gt_class = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            gt_class = Variable(gt_class).cuda()
            self.optimizer_crowd_net.zero_grad()
            s_map, h_map = self.crowd_net(img, gt_map, gt_class)
            pred_map = [s_map, h_map]
            den_loss, den_class_loss = self.crowd_net.get_loss()
            loss = den_loss + den_class_loss
            loss.backward()
            self.optimizer_crowd_net.step()
            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.timer['iter time'].toc(average=False)
                if cfg.DEN_CLASS:
                    print('[ep %d][it %d][ loss %.4f den_loss %.4f den_class_loss %.4f][lr_1 %.4f ][%.2fs]' % \
                          (self.epoch + 1, i + 1, loss.item(), den_loss.item(), den_class_loss.item(),
                           self.optimizer_crowd_net.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred_s: %.2f  pred_h:%.2f]' % (
                    gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0].sum().data / self.cfg_data.LOG_PARA,
                    pred_map[1].sum().data / self.cfg_data.LOG_PARA))

    def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50
        '''

        :return:
        '''
        self.crowd_net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        ssims = AverageMeter()
        psnrs = AverageMeter()
        for vi, data in enumerate(self.val_loader):
            gt_class = None
            with torch.no_grad():
                if cfg.DEN_CLASS:
                    img, gt_map, gt_class = data
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()
                    gt_class = Variable(gt_class).cuda()
                # crowd_net
                s_map, h_map, density_level = self.crowd_net.test_forward(img, gt_map, gt_class)
                pred_map = s_map
                s_loss, h_loss, den_class_loss = self.crowd_net.get_sc_loss()
                loss = s_loss + h_loss + den_class_loss
                # loss, den_class_loss = self.crowd_net.get_loss()
                ####################################
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    pred_normalized_map = np.squeeze(pred_map[i_img] / np.max(pred_map[i_img] + 1e-20))
                    gt_normalized_map = np.squeeze(gt_map[i_img] / np.max(gt_map[i_img] + 1e-20))

                    s = ssim(gt_normalized_map, pred_normalized_map)
                    p = psnr(gt_normalized_map, pred_normalized_map)
                    ssims.update(s)
                    psnrs.update(p)
                    losses.update(loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        psnr_value = psnrs.avg
        ssim_value = ssims.avg

        self.train_record = update_crowd_model(self.crowd_net, self.optimizer_crowd_net,
                                               self.scheduler_crowd_net, self.epoch,
                                               self.i_tb, self.exp_path,
                                               self.exp_name, [mae, mse, loss, psnr_value, ssim_value],
                                               self.train_record)

        print_summary(self.exp_name, [mae, mse, loss, psnr_value, ssim_value], self.train_record)

        print('the den_class_loss is {0:.4f}'.format(den_class_loss.item()))
        print('the s_loss is {0:.4f}'.format(s_loss.item()))
        print('the h_loss is {0:.4f}'.format(h_loss.item()))


class High_branch_CrowdCounter():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data
        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.CC_NET
        self.crowd_net = CrowdCounter(cfg.GPU_ID, self.net_name).cuda()

        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH, map_location=lambda storage, loc: storage.cuda(cfg.GPU_ID[0]))
            self.crowd_net.load_state_dict(latest_state)

        # for name, para in self.crowd_net.named_parameters():
        #     # print(name)
        #     if 'CCN.hd_branch' in name or 'CCN.hd_pred' in name:
        #         # print(name)
        #         if 'bn' in name:
        #             para.requires_grad = False
        #         else:
        #             para.requires_grad = True
        #     else:
        #         para.requires_grad = False
        # only train high branch

        # for name, para in self.crowd_net.named_parameters():
        #     print(name, para.requires_grad)
        self.optimizer_crowd_net = optim.Adam(filter(lambda p: p.requires_grad, self.crowd_net.CCN.parameters()),
                                              lr=cfg.LR, weight_decay=1e-4)
        # self.optimizer_crowd_net = optim.Adam(self.crowd_net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.scheduler_crowd_net = StepLR(self.optimizer_crowd_net, step_size=cfg.NUM_EPOCH_LR_DECAY,
                                          gamma=cfg.LR_DECAY)
        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}

        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.epoch = 0
        self.i_tb = 0
        self.train_sparse_loader, self.train_hc_loader, self.val_loader = dataloader
        # print(s)
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=False)
        print('init finish')

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler_crowd_net.step()
            # training
            self.timer['train time'].tic()
            if epoch == 0:
                self.validate_V1()
            print('train high density')
            self.train_den_class()

            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHA_01', 'SHHA_02', 'SHHB', 'QNRF', 'UCF50']:
                    self.validate_V1()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train_den_class(self):
        self.crowd_net.train()

        # def fix_bn(m):
        #     classname = m.__class__.__name__
        #     if classname.find('BatchNorm') != -1:
        #         m.eval()

        # for module in self.crowd_net.children():
        #     print(module)
        #     class_name = module.__class__.__name__
        #     print(class_name)
        # self.crowd_net.apply(fix_bn)

        # 训练所有正常尺寸图片 decoder采用unet架构
        for i, data in enumerate(self.train_hc_loader):
            self.timer['iter time'].tic()
            img, gt_map, gt_class = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            gt_class = Variable(gt_class).cuda()
            self.optimizer_crowd_net.zero_grad()
            pred_map, den_class = self.crowd_net(img, gt_map, gt_class)
            den_loss, den_class_loss = self.crowd_net.get_loss()
            # den_loss, den_class_loss = self.crowd_net.get_sc_loss()
            loss = den_loss
            loss.backward()
            self.optimizer_crowd_net.step()
            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.timer['iter time'].toc(average=False)
                if cfg.DEN_CLASS:
                    print('[ep %d][it %d][ loss %.4f den_loss %.4f den_class_loss %.4f][lr_1 %.4f ][%.2fs]' % \
                          (self.epoch + 1, i + 1, loss.item(), den_loss.item(), den_class_loss.item(),
                           self.optimizer_crowd_net.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                    gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0].sum().data / self.cfg_data.LOG_PARA))

    def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50
        '''

        :return:
        '''
        self.crowd_net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        ssims = AverageMeter()
        psnrs = AverageMeter()
        k = 0
        for vi, data in enumerate(self.val_loader):
            gt_class = None
            with torch.no_grad():
                if cfg.DEN_CLASS:
                    img, gt_map, gt_class = data
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()
                    gt_class = Variable(gt_class).cuda()
                    # gt_class = None
                else:
                    img, gt_map = data
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()
                # crowd_net
                s_map, h_map, density_level = self.crowd_net.test_forward(img, gt_map, gt_class)
                density_level = torch.argmax(density_level, dim=1)
                s_loss, h_loss, den_class_loss = self.crowd_net.get_sc_loss()

                # print('the loss of density level is {0}'.format(den_class_loss.item()))
                if density_level == 3:
                    k = k + 1
                    # print(density_level)
                    pred_map = h_map
                    den_loss = h_loss
                else:
                    pred_map = s_map
                    den_loss = s_loss
                # den_loss, den_class_loss = self.crowd_net.get_sc_loss()
                loss = den_loss
                # loss, den_class_loss = self.crowd_net.get_loss()
                ####################################
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    pred_normalized_map = np.squeeze(pred_map[i_img] / np.max(pred_map[i_img] + 1e-20))
                    gt_normalized_map = np.squeeze(gt_map[i_img] / np.max(gt_map[i_img] + 1e-20))

                    s = ssim(gt_normalized_map, pred_normalized_map)
                    p = psnr(gt_normalized_map, pred_normalized_map)
                    ssims.update(s)
                    psnrs.update(p)
                    losses.update(loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        psnr_value = psnrs.avg
        ssim_value = ssims.avg

        self.train_record = update_crowd_model(self.crowd_net, self.optimizer_crowd_net,
                                               self.scheduler_crowd_net, self.epoch,
                                               self.i_tb, self.exp_path,
                                               self.exp_name, [mae, mse, loss, psnr_value, ssim_value],
                                               self.train_record)

        print_summary(self.exp_name, [mae, mse, loss, psnr_value, ssim_value], self.train_record)

        print('the den_class_loss is {0:.4f}'.format(den_class_loss.item()))
        print('the number of k=3 is {0}'.format(k))


class Two_stage_CrowdCounter():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data
        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.CC_NET
        self.crowd_net = CrowdCounter(cfg.GPU_ID, self.net_name).cuda()
        self.optimizer_crowd_net = optim.Adam(self.crowd_net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.scheduler_crowd_net = StepLR(self.optimizer_crowd_net, step_size=cfg.NUM_EPOCH_LR_DECAY,
                                          gamma=cfg.LR_DECAY)

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}

        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.epoch = 0
        self.i_tb = 0
        self.train_den_class_loader, self.val_loader, self.train_hc_loader = dataloader
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)

    def forward(self):
        T = 1
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler_crowd_net.step()
            # training
            self.timer['train time'].tic()

            if self.epoch % T == 0 and self.epoch > 50:
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('train big scale image')
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                self.train_big_scale()
            else:
                self.train_den_class()
            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHA_01', 'SHHA_02', 'SHHB', 'QNRF', 'UCF50']:
                    self.validate_V1()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train_den_class(self):
        self.crowd_net.train()
        for i, data in enumerate(self.train_den_class_loader):
            self.timer['iter time'].tic()
            img, gt_map, gt_class = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            gt_class = Variable(gt_class).cuda()
            self.optimizer_crowd_net.zero_grad()
            pred_map = self.crowd_net(img, gt_map, gt_class)
            loss = self.crowd_net.loss
            loss.backward()
            den_loss, den_class_loss = self.crowd_net.get_loss()
            self.optimizer_crowd_net.step()
            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.timer['iter time'].toc(average=False)
                if cfg.DEN_CLASS:
                    print('[ep %d][it %d][ loss %.4f den_loss %.4f den_class_loss %.4f][lr_1 %.4f ][%.2fs]' % \
                          (self.epoch + 1, i + 1, loss.item(), den_loss.item(), den_class_loss.item(),
                           self.optimizer_crowd_net.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff))
                else:
                    print('[ep %d][it %d][ den_loss %.4f][lr_1 %.4f ][%.2fs]' % \
                          (self.epoch + 1, i + 1, loss.item(),
                           self.optimizer_crowd_net.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                    gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0].sum().data / self.cfg_data.LOG_PARA))

    def train_big_scale(self):  # training for all datasets
        self.crowd_net.train()
        for i, data in enumerate(self.train_hc_loader):
            self.timer['iter time'].tic()
            img, gt_map, _ = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            self.optimizer_crowd_net.zero_grad()
            pred_map = self.crowd_net(img, gt_map)
            loss = self.crowd_net.loss
            loss.backward()
            den_loss, den_class_loss = self.crowd_net.get_loss()
            self.optimizer_crowd_net.step()
            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.timer['iter time'].toc(average=False)
                if cfg.DEN_CLASS:
                    print('[ep %d big_scale][it %d][ loss %.4f den_loss %.4f den_class_loss %.4f][lr_1 %.4f ][%.2fs]' % \
                          (self.epoch + 1, i + 1, loss.item(), den_loss.item(), den_class_loss.item(),
                           self.optimizer_crowd_net.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff))
                else:
                    print('[ep %d][it %d][ den_loss %.4f][lr_1 %.4f ][%.2fs]' % \
                          (self.epoch + 1, i + 1, loss.item(),
                           self.optimizer_crowd_net.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                    gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0].sum().data / self.cfg_data.LOG_PARA))

    def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50
        '''

        :return:
        '''
        self.crowd_net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        ssims = AverageMeter()
        psnrs = AverageMeter()
        for vi, data in enumerate(self.val_loader):
            gt_class = None
            with torch.no_grad():
                if cfg.add_seg_mask:
                    img, gt_map, seg_map = data
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()
                    seg_map = Variable(seg_map).cuda()
                else:
                    if cfg.DEN_CLASS:
                        img, gt_map = data
                        img = Variable(img).cuda()
                        gt_map = Variable(gt_map).cuda()
                        gt_class = None
                    else:
                        img, gt_map = data
                        img = Variable(img).cuda()
                        gt_map = Variable(gt_map).cuda()
                # crowd_net
                pred_map = self.crowd_net.forward(img, gt_map)
                loss, den_class_loss = self.crowd_net.get_loss()
                ####################################
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    pred_normalized_map = np.squeeze(pred_map[i_img] / np.max(pred_map[i_img] + 1e-20))
                    gt_normalized_map = np.squeeze(gt_map[i_img] / np.max(gt_map[i_img] + 1e-20))

                    s = ssim(gt_normalized_map, pred_normalized_map)
                    p = psnr(gt_normalized_map, pred_normalized_map)
                    ssims.update(s)
                    psnrs.update(p)
                    losses.update(loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        psnr_value = psnrs.avg
        ssim_value = ssims.avg

        self.train_record = update_crowd_model(self.crowd_net, self.optimizer_crowd_net,
                                               self.scheduler_crowd_net, self.epoch,
                                               self.i_tb, self.exp_path,
                                               self.exp_name, [mae, mse, loss, psnr_value, ssim_value],
                                               self.train_record)

        print_summary(self.exp_name, [mae, mse, loss, psnr_value, ssim_value], self.train_record)

        print('the den_class_loss is {0:.4f}'.format(den_class_loss.item()))
