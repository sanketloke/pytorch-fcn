import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
from pdb import set_trace as st
from torch.utils import data
from torchvision import transforms
import PIL
import scipy.io
import os
import random
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
from scipy.misc import imread, imresize


def recursive_glob(rootdir='.', suffix=''):
    ''' Performs recursive glob with given suffix and rootdir '''
    return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

class ADE20KLoader(data.Dataset):

    def load_label_map(self):
        pass
        
    def __init__(self, root,train=True, is_transform=True, img_size=512,transform=None,target_transform=None,max_sample=-1):
        self.root = root
        self.root_img = root +'/data/ADEChallengeData2016/images'
        self.root_seg = root + '/data/ADEChallengeData2016/annotations'
        self.is_train=train
        self.load_label_info()
        if train:
            self.split='training'
            self.split_file=root+'/ADE20K_object150_train.txt'
        else:
            self.split='validation'
            self.split_file = root +'/ADE20K_object150_val.txt'
        self.is_transform = is_transform
        self.transform=transform
        self.target_transform=target_transform
        self.n_classes = 150
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])

        self.list_sample = [x.rstrip() for x in open(self.split_file, 'r')]

        if self.is_train:
            random.shuffle(self.list_sample)
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        num_sample = len(self.list_sample)

        self.imgSize=500
        self.flip=0
        self.segSize = 500

    def __len__(self):
        return len(self.list_sample)

    # def __getitem__(self, index):
    #     img_path = self.files[self.split][index].rstrip()
    #     lbl_path = img_path[:-4] + '_seg.png'
    #     img = PIL.Image.open(img_path).convert('RGB')
    #     lbl = PIL.Image.open(lbl_path).convert('RGB')
    #     if self.transform is not None:
    #         img=self.transform(img)
    #     lbl = np.array(lbl, dtype=np.uint8)
    #     st()
    #     lbl = self.encode_segmap(lbl)
    #     lbl=PIL.Image.fromarray(lbl,mode='L')
    #     if self.target_transform is not None:
    #         lbl=self.target_transform(lbl)
    #     # print np.unique(lbl.numpy())
    #     # print img.size()
    #     # print lbl.size()
    #     return img, lbl

    def _scale_and_crop(self, img, seg, cropSize, is_train):
        h, w = img.shape[0], img.shape[1]

        if is_train:
            # random scale
            scale = random.random() + 0.5     # 0.5-1.5
            scale = max(scale, 1. * cropSize / (min(h, w) - 1))
        else:
            # scale to crop size
            scale = 1. * cropSize / (min(h, w) - 1)
        img_scale = imresize(img, scale, interp='bilinear')
        seg_scale = imresize(seg, scale, interp='nearest')
        h_s, w_s = img_scale.shape[0], img_scale.shape[1]
        if is_train:
            # random crop
            x1 = random.randint(0, w_s - cropSize)
            y1 = random.randint(0, h_s - cropSize)
        else:
            # center crop
            x1 = (w_s - cropSize) // 2
            y1 = (h_s - cropSize) // 2

        img_crop = img_scale[y1: y1 + cropSize, x1: x1 + cropSize, :]
        seg_crop = seg_scale[y1: y1 + cropSize, x1: x1 + cropSize]
        return img_crop, seg_crop

    def __getitem__(self, index):
        img_basename = self.list_sample[index]
        path_img = os.path.join(self.root_img, img_basename)
        path_seg = os.path.join(self.root_seg,
                                img_basename.replace('.jpg', '.png'))

        assert os.path.exists(path_img), '[{}] does not exist'.format(path_img)
        assert os.path.exists(path_seg), '[{}] does not exist'.format(path_seg)

        # load image and label
        try:
            img = imread(path_img, mode='RGB')
            seg = imread(path_seg)
            assert(img.ndim == 3)
            assert(seg.ndim == 2)
            assert(img.shape[0] == seg.shape[0])
            assert(img.shape[1] == seg.shape[1])

            # random sacle and crop
            if self.imgSize > 0:
                img, seg = self._scale_and_crop(img, seg,
                                                self.imgSize, self.is_train)

            # image to float
            img = img.astype(np.float32) / 255.
            img = img.transpose((2, 0, 1))

            if self.segSize > 0:
                seg = imresize(seg, (self.segSize, self.segSize),
                               interp='nearest')

            # label to int from -1 to 149
            seg = seg.astype(np.int) - 1

            # to torch tensor
            image = torch.from_numpy(img)
            segmentation = torch.from_numpy(seg)
        except Exception as e:
            print('Failed loading image/segmentation [{}]: {}'
                  .format(path_img, e))
            # dummy data
            image = torch.zeros(3, self.imgSize, self.imgSize)
            segmentation = -1 * torch.ones(self.segSize, self.segSize).long()
            return image, segmentation

        # substract mean and devide by std
        image = self.img_transform(image)

        # flip augmentation
        if self.flip and random.choice([-1, 1]) > 0:
            inv_idx = torch.LongTensor(range(image.size(2)-1, -1, -1))
            image = image.index_select(2, inv_idx)
            inv_idx = torch.LongTensor(range(segmentation.size(1)-1, -1, -1))
            segmentation = segmentation.index_select(1, inv_idx)

        return image, segmentation

    def label2image(self,label_matrix):
        def reverse_val(temp):
            class_val = self.pixelmap[temp]
            return tuple(class_val)
        z=np.vectorize(reverse_val)
        a=np.dstack(z(label_matrix))
        return a.astype(np.uint8)

    def load_label_info(self):
        mat=scipy.io.loadmat(self.root+'/color150.mat')
        colormap=list(mat['colors'])
        colormap.append( np.array([0,0,0]) )
        self.label_info = [line.rstrip('\n') for line in open(self.root+'/label.txt')]
        labels=self.label_info
        self.label2color= {  labels[i]:tuple(c)  for i,c in enumerate(colormap)}
        self.labelmap = { i:[l]  for i,l in enumerate(labels) }
        self.pixelmap= {  i:tuple(c)  for i,c in enumerate(colormap)}
        self.pixelmap[255]=(10,10,10)

if __name__ == '__main__':
    local_path = 'ade20k/'
    dst = ADE20KLoader('/home/sanket/repositories/lifelong-learning/dataset/ADE20K/ade20k',)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            for j in range(4):
                plt.imshow(dst.decode_segmap(labels.numpy()[j]))
            plt.show()
