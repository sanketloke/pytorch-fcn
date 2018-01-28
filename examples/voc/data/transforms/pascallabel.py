"""
Python implementation of the color map function for the PASCAL VOC data set.
Official Matlab version can be found in the PASCAL VOC devkit
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
"""
import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt
from pdb import set_trace as st


def color_map(N=256, normalized=False):
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

def get_labels_and_map():
    labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
    nclasses = 22
    cmap = color_map()
    pixelmap={}
    for i in range(nclasses):
        pixelmap[i]=cmap[i]
    return pixelmap,labels

pixelmap,labels=get_labels_and_map()

def label2image(label_matrix):
    def reverse_val(temp):
        class_val = pixelmap[temp]
        return class_val
    z=np.vectorize(reverse_val)
    a=np.dstack(z(label_matrix))
    return a.astype(np.uint8)
