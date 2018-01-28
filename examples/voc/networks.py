import torch
import torch.nn as nn
from torchvision import models
import math
import copy
import numpy as np
from torch.nn import functional as F
import copy
import torch
import torch.nn as nn
import torchvision
from pdb import set_trace as st
import os.path as osp
import torchfcn

class FCN8sBase(nn.Module):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn8s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU',
            path=cls.pretrained_model,
            md5='dbd9bbb3829a3184913bccc74373afbb',
        )

    def __init__(self):
        super(FCN8sBase, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        self.feat1 = [ self.conv1_1, self.relu1_1, self.conv1_2, self.relu1_2,self.pool1]

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4
        self.feat2 = [ self.conv2_1, self.relu2_1, self.conv2_2, self.relu2_2,self.pool2]


        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8
        self.feat3 = [ self.conv3_1, self.relu3_1, self.conv3_2, self.relu3_2, self.conv3_3,self.relu3_3, self.pool3]


        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16
        self.feat4 = [ self.conv4_1, self.relu4_1, self.conv4_2, self.relu4_2,self.conv4_3,self.relu4_3,self.pool4]

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32
        self.feat5 = [ self.conv5_1, self.relu5_1, self.conv5_2, self.relu5_2,self.conv5_3,self.relu5_3,self.pool5]


        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        self.fconn = [ self.fc6, self.relu6 , self.drop6 ,self.fc7 , self.relu7 , self.drop7 ]

        self.task_parameters = {}
        self.tasks = {}
        self.score_fr = None
        self.score_pool3 = None
        self.score_pool4 = None
        self.upscore2 = None
        self.upscore8 = None
        self.upscore_pool4 = None
        # self.score_fr = nn.Conv2d(4096, n_class, 1)
        # self.score_pool3 = nn.Conv2d(256, n_class, 1)
        # self.score_pool4 = nn.Conv2d(512, n_class, 1)

        # self.upscore2 = nn.ConvTranspose2d(
        #     n_class, n_class, 4, stride=2, bias=False)
        # self.upscore8 = nn.ConvTranspose2d(
        #     n_class, n_class, 16, stride=8, bias=False)
        # self.upscore_pool4 = nn.ConvTranspose2d(
        #     n_class, n_class, 4, stride=2, bias=False)
        self._initialize_weights()
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        self.copy_params_from_vgg16(vgg16)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))


class FCN8s(FCN8sBase):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn8s-atonce_from_caffe.pth')


    def __init__(self, task_name,task_type ,task_classes, gpu_ids=[0], setting='normal',encoder='resnet50_dilated8',decoder='c5bilinear'):
        super(FCN8s, self).__init__()
        self.modify_model(task_name, task_type, task_classes)
        self.set_setting(setting)

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vblE1VUIxV1o2d2M',
            path=cls.pretrained_model,
            md5='bfed4437e941fef58932891217fe6464',
        )

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

    def forward_custom(self, x, tasks_custom,uncertainty_samples=None):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        scores = {}

        for t in tasks_custom:
            dic = self.task_parameters[t.name]
            score_fr = dic['score_fr']
            score_pool3 = dic['score_pool3']
            score_pool4 = dic['score_pool4']
            upscore2 = dic['upscore2']
            upscore8 = dic['upscore8']
            upscore_pool4 = dic['upscore_pool4']


            h = score_fr(h)
            h = upscore2(h)
            upscore2 = h  # 1/16

            h = score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
            h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
            score_pool4c = h  # 1/16

            h = upscore2 + score_pool4c  # 1/16
            h = upscore_pool4(h)
            upscore_pool4 = h  # 1/8

            h = score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
            h = h[:, :,
                  9:9 + upscore_pool4.size()[2],
                  9:9 + upscore_pool4.size()[3]]
            score_pool3c = h  # 1/8

            h = upscore_pool4 + score_pool3c  # 1/8

            h = upscore8(h)
            h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
            scores[t.name]= h

        return scores

    def set_setting(self, setting):
        if setting == 'feature_extraction':
            print "Feature Extractor "
            k = self.feat1 + self.feat2 + self.feat3 +  self.feat4 +  self.feat5 + self.fconn
            for i in k:
                for j in i.parameters():
                    j.requires_grad = False



    def modify_model(self, task_name, task_type, task_classes, previous_load=None):
        num_classes = task_classes
        if task_name not in self.tasks:
            self.tasks[task_name] = task_name
        if task_type == 'segment':
            if previous_load and previous_load in self.tasks.keys():
                decoder = copy.deepcopy(self.task_parameters[previous_load])
                self.task_parameters[task_name] = decoder
            else:
                decoder = self.build_decoder(task_classes)
                self.task_parameters[task_name] = decoder
        elif task_type == 'depth':
            if previous_load and previous_load in self.tasks.keys():
                decoder = copy.deepcopy(self.task_parameters[previous_load])
                self.task_parameters[task_name] = decoder
            else:
                decoder = self.build_decoder(1)
                self.task_parameters[task_name] = decoder

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def build_decoder(self, n_class):
        dic = {}
        dic['score_fr'] = nn.Conv2d(4096, n_class, 1)
        dic['score_pool3'] = nn.Conv2d(256, n_class, 1)
        dic['score_pool4']= nn.Conv2d(512, n_class, 1)
        dic['upscore2'] = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        dic['upscore8'] = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, bias=False)
        dic['upscore_pool4'] = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        for k,m in dic.iteritems():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
        return dic

    def assign_decoder(self,dic,flag=True):
        for k,v in dic.iteritems():
            if flag:
                setattr(self,k,dic[k])
            else:
                setattr(self,k,None)

    def set_mode(self, task, train_flag=False):
        if task.name in self.tasks.keys():
            self.mode = task.name
            self.mode_type= task.type
            self.assign_decoder(self.task_parameters[task.name])
        else:
            raise ValueError("Task Not Found")

    def unset_mode(self, task, train_flag=False):
        if task.name in self.tasks.keys():
            self.mode = task.name
            self.mode_type= task.type
            self.assign_decoder( self.task_parameters[task.name],flag=False)
        else:
            raise ValueError("Task Not Found")

    def load_cuda_tasks(self, tasks):
        for t in tasks:
            for k,v in  self.task_parameters[t.name]:
                v.cuda()

    def unload_cuda_tasks(self, tasks):
        for t in tasks:
            for k,v in  self.task_parameters[t.name]:
                v.cpu()
