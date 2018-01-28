from __future__ import print_function

import errno
import hashlib
import os
import sys
import tarfile
import numpy as np
import torch.utils.data as data
from PIL import Image
import scipy.io
from six.moves import urllib
import numpy as np
import utils

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
    BASE_DIR = 'VOCdevkit/VOC2010'

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None):
        self.root = root
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'Context')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.labels_400 = {label.replace(' ',''):idx for idx, label in np.genfromtxt(_voc_root + '/labels.txt', delimiter=':', dtype=None)}
        self.labels_59 = {label.replace(' ',''):idx for idx, label in np.genfromtxt(_voc_root + '/classes-59.txt', delimiter=':', dtype=None)}
        self.pixelmap,self.labels=self.get_labels_and_map()
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Main')
        _split_f = os.path.join(_splits_dir, 'train.txt')
        if not self.train:
            _split_f = os.path.join(_splits_dir, 'val.txt')

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".mat")
                assert os.path.isfile(_image)
                assert os.path.isfile(_mask)
                self.images.append(_image)
                self.masks.append(_mask)

    def load_label(self, path):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        The full 400 labels are translated to the 59 class task labels.
        """
        label_400 = scipy.io.loadmat(path)['LabelMap']
        label = np.zeros_like(label_400, dtype=np.uint8)
        for idx, l in enumerate(self.labels_59):
            idx_400 = self.labels_400[l]
            label[label_400 == idx_400] = self.labels_59[l]
        label = label[np.newaxis, ...]
        return label

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = self.load_label(self.masks[index])
        _target=Image.fromarray(_target[0], mode='L')
        _img,_target=utils.randomHorizontalFlip(_img,_target)
        if self.transform is not None:
            _img = self.transform(_img)
        # todo(bdd) : perhaps transformations should be applied differently to masks?
        if self.target_transform is not None:
            _target = self.target_transform(_target)
        return _img, _target
    def __len__(self):
        return len(self.images)


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
        labels =  [label.replace(' ','') for idx, label in np.genfromtxt(_voc_root + '/classes-59.txt', delimiter=':', dtype=None)]
        labels= labels
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