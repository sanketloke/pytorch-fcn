import os
import torch
from data.segmentation import SegmentationDataset, SegmentationDataset_Multi, CityScapesDataset, CityScapesDataset_Multi
from pdb import set_trace as st
import pandas as pd
import numpy as np
from data import pascal, pascalcontext, ade20k, segmentation, kitti_depth, pascal_edges
import numbers
import types
import collections
import math
import random
from PIL import Image, ImageOps
from collections import OrderedDict
from datetime import datetime
from data.transforms.cityscapestransform import ToLabelTensor
from data.transforms import cityscapeslabel

def generate_loader(train_data, batch_size, serial_batches, nThreads):
    return torch.utils.data.DataLoader(train_data,
                batch_size=batch_size,
                shuffle=not serial_batches,
                num_workers=int(nThreads))

class Task:

    def __init__(self, name, root_dir, num_classes, type, opt, task_type='cityscapes', transform=None, target_transform=None, pixelmap=None, custom_batch_size=None, custom_type=None):
        self.name = name
        self.root_dir = root_dir
        self.num_classes = num_classes
        self.type = type
        self.task_type = task_type
        if pixelmap is not None:
            self.pixelmap = pixelmap
        if transform is not None:
            self.transform = transform
        if target_transform is not None:
            self.target_transform = target_transform
        if custom_batch_size is not None:
            self.custom_batch_size = custom_batch_size
        else:
            self.custom_batch_size = opt.batchSize
        self.custom_type = custom_type
        self.__load__(opt)
        self.ignore_index = -1

    def __load__(self, opt):
        if self.type == 'segment':
            if self.task_type == 'cityscapes':
                self.train_data = CityScapesDataset(
                    self.root_dir + '/train/cityscapes', transform=self.transform, target_transform=self.target_transform, return_paths=True)
                self.test_data = CityScapesDataset(
                    root=self.root_dir + '/test/cityscapes', transform=self.transform, target_transform=self.target_transform, return_paths=True)
            elif self.task_type == 'pascalcontext':
                self.train_data = pascalcontext.VOCSegmentation(
                    self.root_dir + '/train/cityscapes', transform=self.transform, target_transform=self.target_transform)
                self.test_data = pascalcontext.VOCSegmentation(
                    self.root_dir + '/test/cityscapes', train=False, transform=self.transform, target_transform=self.target_transform)
                _, labels = self.train_data.get_labels_and_map()
                self.pixelmap = {i: label for i, label in enumerate(labels)}

            elif self.task_type == 'pascalvoc12':
                self.train_data = pascal.SBDClassSeg(
                    self.root_dir, transform=self.transform, target_transform=self.target_transform, custom_type=self.custom_type)
                self.test_data = pascal.SBDClassSeg(
                    self.root_dir, train=False, transform=self.transform, target_transform=self.target_transform, custom_type=self.custom_type)
                self.pixelmap = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                                 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
                self.ignore_index = 20

            elif self.task_type == 'ade20k':
                self.train_data = ade20k.ADE20KLoader(
                    self.root_dir, transform=self.transform, target_transform=self.target_transform)
                self.test_data = ade20k.ADE20KLoader(
                    self.root_dir, train=False, transform=self.transform, target_transform=self.target_transform)
                self.pixelmap = None

            elif self.task_type == 'gtav':
                self.train_data = SegmentationDataset(root=os.path.join(
                    self.root_dir, 'train', self.task_dir), transform=self.transform, target_transform=self.target_transform, return_paths=True)
                self.test_data = SegmentationDataset(root=os.path.join(
                    self.root_dir, 'test', self.task_dir), transform=self.transform, target_transform=self.target_transform, return_paths=True)

            elif self.task_type == 'pascal_custom':
                self.train_data = pascal_sequential.PascalCustomDataset(
                    root=self.root_dir, dataset_type=self.custom_type, transform=self.transform, target_transform=self.target_transform, return_paths=True)
                self.num_classes = self.train_data.num_classes
                self.pixelmap = self.train_data.pixelmap
                self.test_data = pascal_sequential.PascalCustomDataset(
                    root=self.root_dir, dataset_type=self.custom_type, mode='val', transform=self.transform, target_transform=self.target_transform, return_paths=True)

            elif self.task_type == 'edges':
                self.train_data = pascal_edges.PascalCustomDataset(
                    root=self.root_dir, dataset_type=self.custom_type, transform=self.transform, target_transform=self.target_transform, return_paths=True)
                self.num_classes = self.train_data.num_classes
                self.pixelmap = self.train_data.pixelmap
                self.test_data = pascal_edges.PascalCustomDataset(
                    root=self.root_dir, dataset_type=self.custom_type, mode='val', transform=self.transform, target_transform=self.target_transform, return_paths=True)

            else:
                raise NotImplementedError()
        else:

            if self.task_type == 'kitti':
                self.train_data = kitti_depth.DepthDataset(
                    root=self.root_dir, transform=self.transform, target_transform=self.target_transform)
                self.num_classes = self.train_data.num_classes
                self.pixelmap = self.train_data.pixelmap
                self.test_data = kitti_depth.DepthDataset(
                    root=self.root_dir, mode='val', transform=self.transform, target_transform=self.target_transform)

        self.train_data_loader = generate_loader(
            self.train_data, self.custom_batch_size, opt.serial_batches, opt.nThreads)
        self.test_data_loader = generate_loader(
            self.test_data, self.custom_batch_size, opt.serial_batches, opt.nThreads)

        if not os.path.exists(opt.checkpoints_dir + '/' + opt.name + '/' + self.name + '/plots'):
            try:
                os.makedirs(opt.checkpoints_dir + '/' +
                            opt.name + '/' + self.name)
            except:
                print 'Exists'
            try:
                os.makedirs(opt.checkpoints_dir + '/' +
                            opt.name + '/' + self.name + '/plots')
            except:
                print 'Exists'
        self.plot_dir = opt.checkpoints_dir + '/' + \
            opt.name + '/' + self.name + '/plots'


class Loss(object):

    def __init__(self, task):
        self.task = task
        self.mean_pixel_acc = 0
        self.mean_class_acc = 0
        self.mean_class_iou = 0
        classes = self.task.num_classes
        if self.task.name == 'pascalanimal' or self.task.name == 'pascalobject':
            classes = len(self.task.train_data_loader.dataset.categories) + 1
        self.per_class_acc = [0] * classes
        self.per_class_iou = [0] * classes
        self.loss = 0
        self.mpa = []
        self.mca = []
        self.mci = []
        self.pac = {}
        self.pci = {}
        for i in range(classes):
            self.pac[i] = []
            self.pci[i] = []
        self.pixelmap = task.pixelmap
        self.count = 0

    def aggregate(self, mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou, loss):
        self.count += 1
        count = self.count
        self.mpa.append(mean_pixel_acc)
        self.mca.append(mean_class_acc)
        self.mci.append(mean_class_iou)
        for i in range(len(per_class_acc)):
            self.pac[i].append(per_class_acc[i])
            self.pci[i].append(per_class_iou[i])
        self.mean_pixel_acc = np.nanmean(self.mpa)
        self.mean_class_acc = np.nanmean(self.mca)
        self.mean_class_iou = np.nanmean(self.mci)
        self.loss += (loss - self.loss) / count
        for i in range(len(per_class_acc)):
            self.per_class_acc[i] = np.nanmean(self.pac[i])
            self.per_class_iou[i] = np.nanmean(self.pci[i])

    def display(self, long_disp=True):
        if not long_disp:
            print 'Task[{}] , Loss:[{}] MeanPixelAcc:[{}] MeanClassAcc:[{}] MeanClassIOU:[{}]'.format(self.task.name, self.loss, self.mean_pixel_acc, self.mean_class_acc, self.mean_class_iou)
        else:
            print 'Task[{}] , Loss:[{}] MeanPixelAcc:[{}] MeanClassAcc:[{}] MeanClassIOU:[{}]'.format(self.task.name, self.loss, self.mean_pixel_acc, self.mean_class_acc, self.mean_class_iou)
            try:
                for i in range(len(self.per_class_acc)):
                    print 'Class {}'.format(self.pixelmap[i])
                    print 'Pixel Class Accuracy:{}'.format(self.per_class_acc[i])
                    print 'Pixel Class IoU:{}'.format(self.per_class_iou[i])
            except:
                print 'Error Occured'

    def get_scalars(self):
        return OrderedDict([('Mean Pixel Accuracy', self.mean_pixel_acc),
                            ('Mean Class Accuracy', self.mean_class_acc),
                            ('Mean Class IOU', self.mean_class_iou),
                            ('Loss', self.loss)
                            ])

def log_results(task_matrix, tasks):
    mean_pixel_acc = np.zeros((len(tasks), len(tasks),))
    mean_class_acc = np.zeros((len(tasks), len(tasks),))
    mean_class_iou = np.zeros((len(tasks), len(tasks),))

    for i in range(len(tasks)):
        print 'Task[{}] '.format(tasks[i].name)
        print 'Num of Classes: [{}]'.format(tasks[i].num_classes)
        print 'Root Directory Name: [{}]'.format(tasks[i].root_dir)

    for i in range(len(tasks)):
        for j in range(len(tasks)):
            mean_pixel_acc[j][i] = task_matrix[i][j].mean_pixel_acc

    for i in range(len(tasks)):
        for j in range(len(tasks)):
            mean_class_acc[j][i] = task_matrix[i][j].mean_class_acc

    for i in range(len(tasks)):
        for j in range(len(tasks)):
            mean_class_iou[j][i] = task_matrix[i][j].mean_class_iou

    mean_pixel_acc_task = np.zeros(len(tasks))
    mean_class_iou_task = np.zeros(len(tasks))
    mean_class_acc_task = np.zeros(len(tasks))

    var_pixel_acc_task = np.zeros(len(tasks))
    var_class_iou_task = np.zeros(len(tasks))
    var_class_acc_task = np.zeros(len(tasks))

    std_pixel_acc_task = np.zeros(len(tasks))
    std_class_acc_task = np.zeros(len(tasks))
    std_class_iou_task = np.zeros(len(tasks))

    for i in range(len(tasks)):
        mean_pixel_acc_task[i] = np.mean(mean_pixel_acc[:, i])
        var_pixel_acc_task[i] = np.var(mean_pixel_acc[:, i])
        std_pixel_acc_task[i] = np.std(mean_pixel_acc[:, i])

        mean_class_acc_task[i] = np.mean(mean_class_acc[:, i])
        var_class_acc_task[i] = np.var(mean_class_acc[:, i])
        std_class_acc_task[i] = np.std(mean_class_acc[:, i])

        mean_class_iou_task[i] = np.mean(mean_class_iou[:, i])
        var_class_iou_task[i] = np.var(mean_class_iou[:, i])
        std_class_iou_task[i] = np.std(mean_class_iou[:, i])

    print pd.DataFrame(mean_pixel_acc).to_latex()
    print
    print
    print pd.DataFrame(mean_class_acc).to_latex()
    print
    print
    print pd.DataFrame(mean_class_iou).to_latex()
    print
    print
    print 'Mean,Variance,Standard Deviation of Pixel Accuracy'
    print
    print pd.DataFrame(mean_pixel_acc_task).to_latex()
    print
    print pd.DataFrame(var_pixel_acc_task).to_latex()
    print
    print pd.DataFrame(std_pixel_acc_task).to_latex()
    print
    print
    print 'Mean,Variance,Standard Deviation of Class Accuracy'
    print
    print pd.DataFrame(mean_class_acc_task).to_latex()
    print
    print pd.DataFrame(var_class_acc_task).to_latex()
    print
    print pd.DataFrame(std_class_acc_task).to_latex()
    print
    print
    print 'Mean,Variance,Standard Deviation of Class IOU'
    print
    print pd.DataFrame(mean_class_iou_task).to_latex()
    print
    print pd.DataFrame(var_class_iou_task).to_latex()
    print
    print pd.DataFrame(std_class_iou_task).to_latex()

# LWF Utility Classes / Functions

class TaskCache(object):

    def __init__(self, checkpoints_dir, tasks, opt):
        self.checkpoints_dir = checkpoints_dir
        self.tasks = tasks
        self.cache = {}
        for task in tasks:
            self.cache[task.name] = []
            cachedir = os.path.join(checkpoints_dir, opt.name, task.name)
            if not os.path.exists(cachedir):
                os.makedirs(cachedir)
            if not os.path.exists(cachedir + '/cache'):
                os.makedirs(cachedir + '/cache')
        self.buffer_images = []
        self.buffer_scores = []
        self.current_index = 0
        self.bufferSize = 100
        self.job_name = opt.name

    def clear(self):
        for task in self.tasks:
            cachedir = os.path.join(
                self.checkpoints_dir, self.job_name, task.name, 'cache')
            for i in os.listdir(cachedir):
                file = os.path.join(cachedir, i)
                if os.path.exists(file):
                    os.remove(file)
        for task in self.tasks:
            self.cache[task.name] = []

    def reset_index(self):
        self.current_index = 0
        self.buffer_images = []
        self.buffer_scores = []

    def store_done(self, last_index, task):
        file_index = int(last_index / self.bufferSize)
        image_path = os.path.join(
            self.cachedir, "image_" + str(file_index) + ".pth")
        score_path = os.path.join(
            self.cachedir, "score_" + str(file_index) + ".pth")
        torch.save(self.buffer_images, image_path)
        torch.save(self.buffer_scores, score_path)
        self.cache[task.name].append((image_path, score_path))
        self.buffer_images = []
        self.buffer_scores = []
        self.current_index = 0

    def store(self, image, score, task, index, opt):
        self.cachedir = os.path.join(
            self.checkpoints_dir, opt.name, task.name, 'cache')
        if index % self.bufferSize == 0 and index != 0:
            file_index = int(index / self.bufferSize)
            image_path = os.path.join(
                self.cachedir, "image_" + str(file_index - 1) + ".pth")
            score_path = os.path.join(
                self.cachedir, "score_" + str(file_index - 1) + ".pth")
            torch.save(self.buffer_images, image_path)
            torch.save(self.buffer_scores, score_path)
            self.buffer_images = []
            self.buffer_scores = []
            self.cache[task.name].append((image_path, score_path))
        self.buffer_images.append(image)
        self.buffer_scores.append(score)
        self.current_index += 1

    def get(self, task_name, index):
        file_index = int(index / self.bufferSize)
        pointer = index % self.bufferSize
        if index < self.current_index:
            print 'Bad Move!!'
        try:
            if index % self.bufferSize == 0:
                self.buffer_images = torch.load(
                    self.cache[task_name][file_index][0])
                self.buffer_scores = torch.load(
                    self.cache[task_name][file_index][1])
                self.current_index = index
                return self.buffer_images[pointer], self.buffer_scores[pointer]
            else:
                return self.buffer_images[pointer], self.buffer_scores[pointer]
        except:
            print self.cache
            print "BAD!!!"

def store_responses(model, curr_task, tasks, task_cache, opt, limit_updates=None):
    print 'Storing Responses to Cache'
    task_cache.clear()
    for t in tasks:
        model.set_mode(t)
        model = model.cuda()
        for i, (images, _) in enumerate(curr_task.train_data_loader):
            if limit_updates > 0 and i == limit_updates:
                break
            if type(images) == torch.FloatTensor:
                images = (images,)
            scores = model.forward(torch.autograd.Variable(
                images[0], volatile=True).cuda())
            task_cache.store(images[0].cpu(), scores.data.cpu(), t, i, opt)
            print i
        model = model.cpu()
        model.unset_mode(t)
        task_cache.store_done(i, t)
    print 'Done'
