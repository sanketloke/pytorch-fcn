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


class SegmentationDataset(data.Dataset):

    def __init__(self, root, transform=None,
        target_transform=None, return_paths=False,loader=default_loader,
        ignore_flag=False):
        images, labels = make_dataset(root)
        if len(images) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.mode = 'train'
        self.images = images
        self.labels = labels
        if len(self.images) != len(self.labels) and ignore_flag == False:
            raise ValueError("Inconsistent dataset")
        
        #images, labels = shuffle(images, labels)
        self.transform = transform
        self.target_transform = target_transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]
        image = self.loader(image_path)
        label = self.loader(label_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.return_paths:
            return ((image, image_path), (label, label_path))
        else:
            return (image, label)

    def __len__(self):
        return len(self.images)



class CityScapesDataset(SegmentationDataset):

    def load_label_map(self):
        self.label_map,_=cityscapeslabel.get_labels_and_map()
        self.to_label=ToLabelTensor(self.label_map)

    def label2image(self,label):
        """ Numpy Array of
        """
        return self.to_label.label2image(label)

def make_dataset_multi(roots):
    imageList=[]
    labelList=[]
    for root in roots:
        i,l=make_dataset(root)
        imageList += i
        labelList += l
    return imageList,labelList


class SegmentationDataset_Multi(data.Dataset):
    def __init__(self, roots, transform=None,
        target_transform=None, return_paths=False,loader=default_loader,
        ignore_flag=False):
        images, labels = make_dataset_multi(roots)
        if len(images) == 0:
            raise(RuntimeError("Found 0 images in: " + roots + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.roots = roots
        self.mode = 'train'
        self.images = images
        self.labels = labels
        if len(self.images) != len(self.labels) and ignore_flag == False:
            raise ValueError("Inconsistent dataset")
        images, labels = shuffle(images, labels)
        self.transform = transform
        self.target_transform = target_transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]
        image = self.loader(image_path)
        label = self.loader(label_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.return_paths:
            return ((image, image_path), (label, label_path))
        else:
            return (image, label)

    def __len__(self):
        return len(self.images)



class CityScapesDataset_Multi(SegmentationDataset_Multi):

    def load_label_map(self):
        self.label_map,_=cityscapeslabel.get_labels_and_map()

    def label2image(self,label):
        """ Numpy Array of
        """
        to_label=ToLabelTensor(self.label_map)
        return to_label.label2image(label)
