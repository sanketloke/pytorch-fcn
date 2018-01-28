##########################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
##########################################################################

import torch.utils.data as data
from data.transforms.cityscapestransform import ToLabelTensor
from data.transforms import cityscapeslabel
from PIL import Image
import os
import os.path
from sklearn.utils import shuffle
from depth_evaluation_utils import *


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_depth_set(test_files, root):
    gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, args.kitti_dir)
    return gt_files, gt_calib, im_sizes, im_files, cams


def make_dataset(dir, mode='train', sort=True, silent=True):
    root = [line.rstrip('\n') for line in open(dir+'/root.txt')][0]

    if mode=='train':
        image_files_path = dir + 'train.txt'
    else:
        image_files_path = dir + 'test.txt'
    images = read_text_lines(image_files_path)
    return load_depth_set(images,root)


def default_loader(path):
    return Image.open(path).convert('RGB')


class DepthDataset(data.Dataset):

    def __init__(self, root, transform=None,
        target_transform=None, return_paths=False,loader=default_loader,
        ignore_flag=False,mode = 'train'):
        gt_files, gt_calib, im_sizes, im_files, cams = make_dataset(root,mode='')
        self.root = root
        self.mode = mode

        self.gt_files = gt_files
        self.gt_calib = gt_calib
        self.im_sizes = im_sizes
        self.im_files = im_files
        self.cams = cams

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        depth_map = generate_depth_map(self.gt_calib[index],
                                   self.gt_files[index],
                                   self.im_sizes[index],
                                   self.cams[index],
                                   False,
                                   True)
        return (self.loader(gt_files[index]), depth_map) 
        # if self.transform is not None:
        #     image = self.transform(image)
        # if self.target_transform is not None:
        #     label = self.target_transform(label)
        # if self.return_paths:
        #     return ((image, image_path), (label, label_path))
        # else:
        #     return (image, depth_map)

    def __len__(self):
        return len(self.gt_files)


if __name__=='__main__':
    depth = DepthDataset('/groups/jbhuang_lab/usr/sloke/data/kitti')
    from pdb import set_trace as st
    st()
