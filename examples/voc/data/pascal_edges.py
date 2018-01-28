
import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from detail import Detail
from pdb import set_trace as st
import time
from torchvision import transforms
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, sort=True, silent=True):
    images = os.path.join(dir, 'images')
    labels = os.path.join(dir, 'label_processed')
    imageList = []
    labelList = []
    q = os.listdir(images)
    q = sorted(q)
    for filename in os.listdir(images):
        if ( os.path.isfile(os.path.join(labels, filename))
            and is_image_file(os.path.join(labels, filename)) ) :
            imageList.append(os.path.join(images, filename))
            labelList.append(os.path.join(labels, filename))
        else:
            if not silent:
                raise ValueError('File Missing')
    return imageList, labelList


def default_loader(path):
    return Image.open(path).convert('RGB')

annFile='/home/sloke/repos/github/detail-api/dataset/trainval_merged.json' # annotations
imgDir='/home/sloke/repos/github/detail-api/dataset/VOCdevkit/VOC2010/JPEGImages' # jpeg images
phase='trainval'
BASE_DIR = 'VOCdevkit/VOC2010'

class PascalCustomDataset(data.Dataset):

    def load_label_map(self):
        pass

    def __init__(self, root=None ,transform=None, mode='train',
        target_transform=None, return_paths=False,loader=default_loader,
        ignore_flag=False, dataset_type='A'):
        if root:
            annFile= root+'/trainval_withkeypoints.json'
            imgDir= root+'/VOCdevkit/VOC2010/JPEGImages'
        details = Detail(annFile, imgDir, mode)
        self.details=details
        self.dtimgs = details.getImgs()
        images=[ imgDir+'/'+i['file_name'] for i in self.dtimgs]
        self.mode = mode
        self.images = images
        self.transform = transform
        self.target_transform = target_transform
        self.return_paths = return_paths
        self.loader = loader
        self.num_classes= 2
        self.pixelmap =['edge','non-edge']

    def __getitem__(self, index):
        image_path = self.images[index]
        img=self.dtimgs[index]
        mask=self.details.getBounds(img)
        label=mask
        label= Image.fromarray(label)
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        label[ label > 0 ] = 1
        return (image, label[0])

    def __len__(self):
        return len(self.images)

    def get_labels_and_map(self):
        pass

    def label2image(self,label_matrix):
    	label_matrix = label_matrix*255
        return self.details.showMask(label_matrix)

