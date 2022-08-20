from torch.utils.data import Dataset
import imageio as io
import cv2
import torch
from scipy.ndimage import filters
import scipy.io as sio
import numpy as np
import os


def transform(x, y):
    if np.random.uniform() < 0.5:
        x = x[:, ::-1]
        y = y[:, ::-1]
    return x, y


def getLabel(vid_index, frame_index, img_size):
    fixdatafile = ('./fixdata/fixdata' + str(vid_index) + '.mat')
    data = sio.loadmat(fixdatafile)

    fix_x = data['fixdata'][frame_index - 1][0][:, 3]  # frame_index索引从1开始，所以减1；截取第三列数据
    fix_y = data['fixdata'][frame_index - 1][0][:, 2]
    mask = np.zeros((720, 1280), dtype='float32')

    for i in range(len(fix_x)):
        mask[fix_x[i], fix_y[i]] = 1

    mask = filters.gaussian_filter(mask, 40)
    mask = np.array(mask, dtype='float32')
    mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_CUBIC)
    mask = mask.astype('float32') / 255.0
    if mask.max() == 0:
        print(mask.max())
        # print mask.max()
        # print img_name
    else:
        mask = mask / mask.max()  # 归一化
    return mask


class ImageList(Dataset):
    def __init__(self, root, imgs, img_size=(320, 192), for_train=False):
        self.root = root
        self.imgs = imgs
        self.for_train = for_train
        self.img_size = img_size

    def __getitem__(self, index):
        img_name = self.imgs[index]
        vid_index = int(img_name[0:2])
        frame_index = int(img_name[3:9])

        image_name = os.path.join(self.root, img_name)
        img = io.imread(image_name)

        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32')/255.0

        mask = getLabel(vid_index, frame_index, self.img_size)

        if self.for_train:
            img, mask = transform(img, mask)

        img = img.transpose(2, 0, 1)
        mask = mask[None, ...]  # 加维度，无内容，例：[1, 2, 3] -> [[1, 2, 3]]
        img = np.ascontiguousarray(img)  # 放至连续存储空间

        mask = np.ascontiguousarray(mask)
        # print(torch.from_numpy(img))
        # print torch.from_numpy(img)
        # exit(0)
        return torch.from_numpy(img), torch.from_numpy(mask)

    def __len__(self):
        return len(self.imgs)


def get_bdda_label(root, mode, img_name, img_size):
    label_name = os.path.join(root, 'BDDA_fixation')
    label_name = os.path.join(label_name, mode)
    label_name = os.path.join(label_name, img_name)
    label = io.imread(label_name, as_gray=True)
    crop = label[72:648, :]
    crop = cv2.resize(crop, img_size, interpolation=cv2.INTER_CUBIC)
    if crop.max() == 0:
        print(crop.max())
    else:
        crop = crop / crop.max()
    return crop


class BDDA(Dataset):
    def __init__(self, root, imgs, mode='train', img_size=(320, 192), for_train=False):
        self.root = root
        self.mode = mode
        self.imgs = imgs
        self.for_train = for_train
        self.img_size = img_size

    def __getitem__(self, index):
        img_name = self.imgs[index]
        image_path = os.path.join(self.root, 'BDDA_frames')
        image_path = os.path.join(image_path, self.mode)
        image_path = os.path.join(image_path, img_name)
        img = io.imread(image_path)

        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32') / 255.0

        mask = get_bdda_label(self.root, self.mode, img_name, self.img_size)

        if self.for_train:
            img, mask = transform(img, mask)

        img = img.transpose(2, 0, 1)
        mask = mask[None, ...]
        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)
        return torch.from_numpy(img), torch.from_numpy(mask)

    def __len__(self):
        return len(self.imgs)
