##########################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
##########################################################################

import torch.utils.data as data
from data.transforms import cityscapeslabel
from PIL import Image
import os
import os.path
from sklearn.utils import shuffle
from depth_evaluation_utils import *
import numpy as np
from pdb import set_trace as st
from util import normalize_depth_for_display
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_depth_set(test_files, root):
    gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, root)
    return root, test_files,gt_files, gt_calib, im_sizes, im_files, cams


def make_dataset(dir, mode='train', sort=True, silent=True):
    root = [line.rstrip('\n') for line in open(dir+'/root.txt')][0]

    if mode=='train':
        image_files_path = dir + '/train.txt'
    else:
        image_files_path = dir + '/test.txt'
    images = read_text_lines(image_files_path)
    return load_depth_set(images,root)


def default_loader(path):
    return Image.open(path).convert('RGB')


class DepthDataset(data.Dataset):

    def __init__(self, root, transform=None,
        target_transform=None, return_paths=False,loader=default_loader,
        ignore_flag=False,mode = 'train'):
        root , images, gt_files, gt_calib, im_sizes, im_files, cams = make_dataset(root,mode='')
        self.root = root
        self.mode = mode
        self.images = images
        self.gt_files = gt_files
        self.gt_calib = gt_calib
        self.im_sizes = im_sizes
        self.im_files = im_files
        self.cams = cams
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.num_classes=1
        self.pixelmap=None

    def __getitem__(self, index):
        depth_map = generate_depth_map(self.gt_calib[index],
                                   self.gt_files[index],
                                   self.im_sizes[index],
                                   self.cams[index],
                                   False,
                                   True)
        if self.transform:
            image = self.transform(self.loader(self.root+'/'+self.images[index]))
        else:
            image = self.loader(self.root+'/'+self.images[index])

        depth_map = Image.fromarray(depth_map.astype(np.uint8))
        depth_map = self.target_transform(depth_map)
        return ( image, depth_map) 

    def __len__(self):
        return len(self.gt_files)

    def load_label_map(self):
        pass

    def to_rgb1(self,im):
        # I think this will be slow
        w, h = im.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = im
        ret[:, :, 1] = im
        ret[:, :, 2] = im
        return ret

    def label2image(self,label):
        return normalize_depth_for_display(label).astype(np.uint8)

if __name__=='__main__':
    depth = DepthDataset('/groups/jbhuang_lab/usr/sloke/data/kitti')
    print depth[0]
    
