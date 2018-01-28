from __future__ import print_function
import numpy as np
import errno
import hashlib
import os
import sys
import tarfile

import torch.utils.data as data
from PIL import Image

from six.moves import urllib

#reference desimone
class VOCSegmentation(data.Dataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambigious'
    ]

    URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    FILE = "VOCtrainval_11-May-2012.tar"
    MD5 = '6cd6e144f989b92b3379bac3b3de84fd'
    BASE_DIR = 'VOCdevkit/VOC2012'

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,custom_type=None):
        self.root = root
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void' ]
        # train/val/test splits are pre-cut
        self.pixelmap,self.labels=self.get_labels_and_map()
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')

        if custom_type==None:
            _split_f = os.path.join(_splits_dir, 'train.txt')
        elif custom_type=='A':
            _split_f = os.path.join(_splits_dir, 'trainA.txt')
        elif custom_type=='B':
            _split_f = os.path.join(_splits_dir, 'trainB.txt')

        if not self.train:
            _split_f = os.path.join(_splits_dir, 'val.txt')
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_mask)
                self.images.append(_image)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.masks[index])

        if self.transform is not None:
            _img = self.transform(_img)
        # todo(bdd) : perhaps transformations should be applied differently to masks?
        if self.target_transform is not None:
            _target = self.target_transform(_target)
        return _img, _target

    def __len__(self):
        return len(self.images)

    def _check_integrity(self):
        _fpath = os.path.join(self.root, self.FILE)
        if not os.path.isfile(_fpath):
            print("{} does not exist".format(_fpath))
            return False
        _md5c = hashlib.md5(open(_fpath, 'rb').read()).hexdigest()
        if _md5c != self.MD5:
            print(" MD5({}) did not match MD5({}) expected for {}".format(
                _md5c, self.MD5, _fpath))
            return False
        return True


    def color_map(self,N=256, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap/255 if normalized else cmap
        return cmap

    def get_labels_and_map(self):
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        labels=self.labels
        nclasses = len(labels)
        cmap = self.color_map()
        pixelmap={}
        for i in range(nclasses):
            pixelmap[i]=cmap[i]
        return pixelmap,labels


    def label2image(self,label_matrix):
        def reverse_val(temp):
            class_val = self.pixelmap[temp]
            return tuple(class_val)
        z=np.vectorize(reverse_val)
        a=np.dstack(z(label_matrix))
        return a.astype(np.uint8)

    def load_label_map(self):
        pixelmap,labels= self.get_labels_and_map()
        self.pixelmap=pixelmap




#!/usr/bin/env python

import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
from pdb import set_trace as st

class VOCClassSegBase(data.Dataset):

    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform

        # VOC2011 and others are subset of VOC2012
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2012')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(
                dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                lbl_file = osp.join(
                    dataset_dir, 'SegmentationClass/%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl


class VOC2011ClassSeg(VOCClassSegBase):

    def __init__(self, root, split='train', transform=False):
        super(VOC2011ClassSeg, self).__init__(
            root, split=split, transform=transform)
        pkg_root = osp.join(osp.dirname(osp.realpath(__file__)), '..')
        imgsets_file = osp.join(
            pkg_root, 'ext/fcn.berkeleyvision.org',
            'data/pascal/seg11valid.txt')
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2012')
        for did in open(imgsets_file):
            did = did.strip()
            img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
            lbl_file = osp.join(dataset_dir, 'SegmentationClass/%s.png' % did)
            self.files['seg11valid'].append({'img': img_file, 'lbl': lbl_file})


class VOC2012ClassSeg(VOCClassSegBase):

    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA

    def __init__(self, root, split='train', transform=False):
        super(VOC2012ClassSeg, self).__init__(
            root, split=split, transform=transform)


class SBDClassSeg(VOCClassSegBase):

    # XXX: It must be renamed to benchmark.tar to be extracted.
    url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA

    def __init__(self, root, train=True, transform=False,target_transform=False,custom_type=None):
        root = '/home/sloke/data/datasets'
        self.root = root
        self._transform = transform
        dataset_dir = osp.join(self.root, 'VOC/benchmark_RELEASE/dataset')
        self.files = collections.defaultdict(list)
        if train:
            split = 'train'
        else:
            split = 'val'
        self.split = split
        self._target_transform = target_transform
        for split in ['train', 'val']:
            imgsets_file = osp.join(dataset_dir, '%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'img/%s.jpg' % did)
                lbl_file = osp.join(dataset_dir, 'cls/%s.mat' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        mat = scipy.io.loadmat(lbl_file)
        lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)
        lbl[lbl == 255] = -1
        img = Image.fromarray(img)
        lbl = Image.fromarray(lbl)
        if self._transform:
            img = self._transform(img)
#            return self.transform(img, lbl)
        if self._target_transform:
            lbl = self._target_transform(lbl)[0]
        return img, lbl

    def load_label_map(self):
        pass

def get_split(root,
                 train=True,
                 transform=None,
                 target_transform=None):
    p1 = VOCSegmentation(root,transform=transform, target_transform=target_transform)
    p2 = VOCSegmentation(root,transform=transform, target_transform=target_transform)
    p2.images = p1.images[len(p1.images)/2:]
    p1.images = p1.images[:len(p1.images)/2]
    return (p1,p2)
