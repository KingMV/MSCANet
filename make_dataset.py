import numpy as np
import scipy
# import scipy.io as io
import scipy.spatial
from scipy import io as sio
from scipy.ndimage.filters import gaussian_filter

import os
import glob
from matplotlib import pyplot as plt
import h5py
import PIL.Image as Image
from matplotlib import cm as CM
from PIL import Image
import cv2


# parameter setting


def gaussian_filter_density(img, points):
    w, h = img.size
    img_shape = (h, w)
    print("Shape of current image: ", img_shape, ". Totally need generate ", len(points), "gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)

    gt_count = len(points)

    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=4)
    print('generate density...')

    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) / 3 * 0.1
        else:
            sigma = np.average(np.array(img)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    print('gt_count:{0}\tdensity_count:{1}'.format(gt_count, np.sum(density)))
    return density


def get_gt_dots(mat_path, h, w, nh, nw):
    """
    Load Matlab file with ground truth labels and save it to numpy array.
    ** cliping is needed to prevent going out of the array
    """
    mat = sio.loadmat(mat_path)
    gt = mat["image_info"][0, 0][0, 0][0].astype(np.float32).round().astype(int)
    gt[:, 0] = gt[:, 0].clip(0, w - 1)
    gt[:, 1] = gt[:, 1].clip(0, h - 1)
    gt[:, 0] = (gt[:, 0] / w * nw).round().astype(int)
    gt[:, 1] = (gt[:, 1] / h * nh).round().astype(int)
    return gt


def generate_fixed_gaussian_kernel_density(img, points, kernel_size):
    gt = points
    count = 0
    w, h = img.size
    k = np.zeros((h, w), dtype=np.float32)
    for i in range(len(gt)):
        if int(gt[i][1]) < h and int(gt[i][1]) >= 0 and int(gt[i][0]) < w and int(gt[i][0]) >= 0:
            k[int(gt[i][1]), int(gt[i][0])] = 1
            count += 1
    k = gaussian_filter(k, kernel_size)
    print('Ignore {} wrong annotation.'.format(len(gt) - count))
    return k


def gen_gauss_kernels(kernel_size=15, sigma=4):
    kernel_shape = (kernel_size, kernel_size)
    kernel_center = (kernel_size // 2, kernel_size // 2)

    arr = np.zeros(kernel_shape).astype(float)
    arr[kernel_center] = 1

    arr = scipy.ndimage.filters.gaussian_filter(arr, sigma, mode='constant')
    kernel = arr / arr.sum()
    return kernel


def gaussian_fixed_filter_density(non_zero_points, map_h, map_w):
    """
    Fast gaussian filter implementation : using precomputed distances and kernels
    """

    gt_count = non_zero_points.shape[0]
    density_map = np.zeros((map_h, map_w), dtype=np.float32)

    for i in range(gt_count):
        point_y, point_x = non_zero_points[i]
        # print(point_x, point_y)
        kernel_size = 15 // 2
        kernel = gen_gauss_kernels(kernel_size * 2 + 1, 4)
        min_img_x = int(max(0, point_x - kernel_size))
        min_img_y = int(max(0, point_y - kernel_size))
        max_img_x = int(min(point_x + kernel_size + 1, map_h - 1))
        max_img_y = int(min(point_y + kernel_size + 1, map_w - 1))
        # print(min_img_x, min_img_y, max_img_x, max_img_y)
        kernel_x_min = int(kernel_size - point_x if point_x <= kernel_size else 0)
        kernel_y_min = int(kernel_size - point_y if point_y <= kernel_size else 0)
        kernel_x_max = int(kernel_x_min + max_img_x - min_img_x)
        kernel_y_max = int(kernel_y_min + max_img_y - min_img_y)
        # print(kernel_x_max, kernel_x_min, kernel_y_max, kernel_y_min)

        density_map[min_img_x:max_img_x, min_img_y:max_img_y] += kernel[kernel_x_min:kernel_x_max,
                                                                 kernel_y_min:kernel_y_max]
    return density_map


def show_cam_on_image(img, wait_second):
    # layer_name = 'back_encoder_mean_fp'
    layer_name = 'density'
    img = normalize_input(img)
    # print(np.min(mask))
    # print(np.max(mask))
    # print(mask)
    # print(mask.shape)
    heatmap = cv2.applyColorMap(np.uint8(img * 255), cv2.COLORMAP_JET)
    cam = heatmap

    # cam = cv2.addWeighted(heatmap, 0.6, img, 0.4,0)
    # cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imshow(layer_name, np.uint8(255 * cam))
    cv2.waitKey(wait_second)


def generate_foreground_mask(img, points, k_size):
    gt = points
    w, h = img.size
    k = np.zeros((h, w), dtype=np.float32)
    count = 0
    for i in range(len(gt)):
        if int(gt[i][1]) < h and int(gt[i][1]) >= 0 and int(gt[i][0]) < w and int(gt[i][0]) >= 0:
            k[int(gt[i][1]), int(gt[i][0])] = 1
            count += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    mask = cv2.dilate(k, kernel)

    return k, mask


def normalize_input(input):
    return (input - np.min(input)) / (np.max(input) - np.min(input) + 1e-20)
    # return (input - np.min(input)) / (np.max(input))


if __name__ == '__main__':
    # root = r'E:\Dataset\ShanghaiTech'
    root = r'../Crowd_Dataset_01/ShanghaiTech'
    # kernel_size = 15
    method_name = 'fixed'
    sigma = 15
    save_mn = '{0}_sigma_{1}'.format(method_name, sigma)
    # method = 'fixed_sigma_{0}'.format(kernel_size)  # adaptive
    Part = 'A'
    save_flag = True

    mod = 16

    part_train = os.path.join(root, 'part_{0}/train_data'.format(Part))
    part_test = os.path.join(root, 'part_{0}/test_data'.format(Part))

    part_train_image = os.path.join(root, 'part_{0}/train_data'.format(Part), 'images')
    part_train_save = os.path.join(part_train, '{0}'.format(save_mn))

    part_test_image = os.path.join(root, 'part_{0}/test_data'.format(Part), 'images')
    part_test_save = os.path.join(part_test, '{0}'.format(save_mn))

    os.makedirs(part_train_save, exist_ok=True)
    os.makedirs(part_test_save, exist_ok=True)

    # path_sets = [part_train_image, part_test_image]
    path_sets = [part_train_image, part_test_image]
    img_paths = []
    for path in path_sets:
        os.makedirs(os.path.join(path.replace('images', save_mn), 'images'), exist_ok=True)
        os.makedirs(os.path.join(path.replace('images', save_mn), 'dens'), exist_ok=True)
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    for img_path in img_paths:
        print(img_path)
        img = Image.open(img_path)
        w, h = img.size
        if max(w, h) > 1024:
            if w == max(w, h):
                nw, nh = 1024, round(h * 1024 / w / mod) * mod
            else:
                nh, nw = 1024, round(w * 1024 / h / mod) * mod
        else:
            nw, nh = round((w / mod)) * mod, round((h / mod)) * mod
        image_name = os.path.split(img_path)[-1]
        # print(os.path.dirname(img_path))
        new_image_path = os.path.join(os.path.dirname(img_path).replace('images', save_mn), 'images', image_name)
        print(new_image_path)
        if save_flag:
            img.resize((nw, nh), Image.BILINEAR).save(new_image_path)

        points = get_gt_dots(
            img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_'),
            h, w, nh, nw)
        # print(points.shape)
        if method_name == 'adaptive':
            k = gaussian_filter_density(img, points)
        elif method_name == 'fixed':
            # k = gaussian_fixed_filter_density(points, nh, nw)
            k = generate_fixed_gaussian_kernel_density(img, points, kernel_size=sigma)
            print('gt count:{0}\tgen_count{1}'.format(len(points), np.sum(k)))
        else:
            raise ValueError('error mode')
        #
        #     # show_cam_on_image(k, 100)
        point_map, mask = generate_foreground_mask(img, points, 15)
        if save_flag:
            sio.savemat(new_image_path.replace('.jpg', '.mat').replace('images', 'dens'), {'map': k, 'mask': mask,
                                                                                           'points_map': point_map})
