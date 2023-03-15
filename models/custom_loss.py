import torch.nn as nn
import torch


class Custom_MSE_loss(nn.Module):
    def __init__(self, reduction='sum'):
        super(Custom_MSE_loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        loss = torch.abs(pred - target)
        loss = loss ** 2
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class Custom_Log_MSE_loss(nn.Module):
    def __init__(self, reduction='sum'):
        super(Custom_Log_MSE_loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        loss = torch.abs(pred - target)
        loss = loss ** 2
        loss = torch.log(1 + loss)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class Balanced_loss(nn.Module):
    def __init__(self, reduction='sum'):
        super(Balanced_loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        loss = torch.abs(pred - target)
        loss = loss ** 2
        # print('loss shape is {0}'.format(loss.size()))
        gt_cnt = torch.sum(torch.reshape(target, [target.size()[0], -1]), 1) + 1
        gt_cnt = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(gt_cnt, dim=-1), dim=-1), dim=-1)
        # print('gt_cnt shape is {0}'.format(gt_cnt.size()))
        # gt_cnt = torch.unsqueeze(torch.unsqueeze(gt_cnt, dim=-1), dim=-1)
        loss = loss / gt_cnt
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)
