import argparse
from torchvision import transforms
import torch
from models.CC import CrowdCounter
import scipy.io as sio
from PIL import Image, ImageOps
import numpy as np
import cv2
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import csv
import os
# './exp/07-19_10-35_SHHA_02_Crowd_CSRNet_0.0001/all_ep_483_mae_58.55_mse_98.43_psnr_20.66_ssim_0.71.pth'
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='../', type=str, help='path to dataset')
parser.add_argument('--net_name', default='Crowd_CSRNet', type=str, help='name of network')
parser.add_argument('--model_path', default=r'./checkpoint', type=str, help='path to save checkpoint')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')

args = parser.parse_args()
mean_std = ([0.413525998592, 0.378520160913, 0.371616870165], [0.284849464893, 0.277046442032, 0.281509846449])
# mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*mean_std)
])


class log_csv(object):
    def __init__(self, root, csv_name, field_names):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.path = os.path.join(self.root, '{0}.csv'.format(csv_name))
        # new file
        file = open(self.path, 'w', newline='', encoding='utf-8')
        self.field_names = field_names
        self.csv_obj = csv.DictWriter(file, fieldnames=self.field_names)
        self.csv_obj.writeheader()

    def write_csv(self, temp_dict):
        assert isinstance(temp_dict, dict) == True
        self.csv_obj.writerow(temp_dict)


def initialize_model(model_path, gpu, net_name):
    torch.cuda.set_device(gpu[0])
    torch.backends.cudnn.benchmark = True
    net = CrowdCounter(gpu, net_name)
    net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage.cuda(gpu[0])))
    net.cuda()

    return net


def inference_image(model, img):
    model.eval()
    img_tensor = img_transform(img)
    pred_mask = None
    with torch.no_grad():
        img_tensor = torch.unsqueeze(img_tensor, dim=0)
        img_tensor = img_tensor.cuda()
        predict_result = model.test_forward(img_tensor, True)
        if isinstance(predict_result, tuple) and len(predict_result) == 2:
            pred_map, pred_mask = predict_result
        else:
            pred_map = predict_result
        print('predict:{:.2f}'.format(pred_map.sum().item()))
    predict_map = pred_map.cpu().data.numpy()[0, 0, :, :]
    return predict_map, pred_mask


def vis_dir(root):
    import glob
    model = initialize_model(args.model_path, [args.gpu], args.net_name)
    image_files = glob.glob(os.path.join(root, '*.jpg'))
    exp_root = './eval/exp/{0}/shha'.format(args.net_name)
    csv_root = './eval/csv/{0}/shha'.format(args.net_name)
    os.makedirs(exp_root, exist_ok=True)
    os.makedirs(csv_root, exist_ok=True)
    log = log_csv(csv_root, 'shha02_07191015.csv', field_names=['image_name', 'pred_cnt', 'gt_cnt', 'psnr', 'ssim'])
    for f in image_files:
        temp_dict = {}
        print(f)
        img = Image.open(f)
        img_name = f.split('/')[-1].split('.jpg')[0]
        print(img_name)
        temp_dict['image_name'] = img_name
        if img.mode == 'L':
            img = img.convert('RGB')
        height, width = img.size[1], img.size[0]
        height_new = round(height / 16) * 16
        width_new = round(width / 16) * 16
        img = img.resize((width_new, height_new), Image.BILINEAR)
        pred_map, pred_mask = inference_image(model, img)
        temp_dict['gt_cnt'] = 0
        temp_dict['psnr'] = 0
        temp_dict['ssim'] = 0
        temp_dict['error'] = 0
        temp_dict['pred_cnt'] = np.sum(pred_map)
        log.write_csv(temp_dict)
        sio.savemat('{0}_pred_density.mat'.format(os.path.join(exp_root, img_name)), {'data': pred_map})
        if pred_mask is not None:
            pred_mask = pred_mask.cpu().data.numpy()
            sio.savemat('{0}_pred_mask.mat'.format(os.path.join(exp_root, img_name)), {'data': pred_mask})


if __name__ == '__main__':
    root_path = args.data_path
    vis_dir(root_path)
