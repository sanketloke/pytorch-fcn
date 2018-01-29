
from data.segmentation import SegmentationDataset,SegmentationDataset_Multi,CityScapesDataset,CityScapesDataset_Multi
from data import pascal,pascalcontext,ade20k,segmentation
import os
import numpy as np
from data.custom_transforms import Scale , EliminationTransform, ClampTransform
import torchvision.transforms as transforms
import PIL
import torch
from pdb import set_trace as st
from sets import Set
from data.pascal import get_split
from networks import FCN8s
import torch.nn as nn
import time
import options
from utils.visualizer import Visualizer
from misc import Task,log_results,TaskCache,store_responses
from data.transforms.cityscapeslabel import get_labels_and_map
from core import train,multi_train,test,lwf_train,uncertainty_train,ewf_train
from util import unique,load_pretrained
import networks
import evaluation
opt = options.BaseOptions().parse()
visualizer = Visualizer(opt)



tasks=[]
transform = transforms.Compose([
                                      Scale( (opt.loadSize,opt.loadSize) ),
                                       transforms.ToTensor()
                                    ])
target_transform = transforms.Compose([
    Scale((opt.loadSize,opt.loadSize),interpolation=PIL.Image.NEAREST),
    transforms.ToTensor()
    #,EliminationTransform()
    #                                   ,ClampTransform(20)
                                    ])
t1 = Task('pascalvocA',opt.data_dir+'/pascalvoc',21,'segment',opt,'pascalvoc12',transform=transform,target_transform=target_transform,custom_type='A')
t2 = Task('pascalvocB',opt.data_dir+'/pascalvoc',21,'segment',opt,'pascalvoc12',transform=transform,target_transform=target_transform,custom_type='B')

tasks.append(t1)
tasks.append(t2)


prev_tasks=[]
task_loss_after_each_train=[]
if opt.pretrained:
    model=load_pretrained(opt.load_pretrained)
    start_index= opt.pretrained
    test_results=test(tasks,model,opt,visualizer=visualizer,limit_updates=opt.limit_test_updates)
    # for i in range(0,start_index):
    #    test_results=test(tasks,model,opt)
    #    task_loss_after_each_train.append(test_results)
    #    prev_tasks.append(tasks[i])
else:
    t= tasks[0]
    model=  networks.__dict__[opt.model](t.name,t.type,t.num_classes,encoder=opt.encoder,decoder=opt.decoder,setting=opt.model_setting)
    for t in tasks:
        model.modify_model(t.name,t.type,t.num_classes)
    start_index=0

if opt.model_setting=='feature_extraction' or opt.model_setting=='finetuning':
    for task_index in range(start_index,len(tasks)):
        task=tasks[task_index]
        train(task,model,task.train_data_loader,opt.num_epochs,opt,visualizer=visualizer,in_parallel=opt.in_parallel,limit_updates=opt.limit_updates)
        previous_task=task
        if opt.same_task:
            for s_task_index in range( task_index+1,len(tasks)):
                s_task=tasks[s_task_index]
                model.modify_model(s_task,previous_load=task)
        test_results=test(tasks,model,opt,visualizer=visualizer,limit_updates=opt.limit_test_updates)
        task_loss_after_each_train.append(test_results)
elif opt.model_setting=='multitask':
    multi_train(tasks,model,opt.num_epochs,opt,visualizer=visualizer,limit_updates=opt.limit_updates)
    test_results=test(tasks,model,opt,limit_updates=opt.limit_test_updates)
    for t in tasks:
      task_loss_after_each_train.append(test_results)
elif opt.model_setting=='soft_distill':
    print 'soft distill here'
    distill_loss = evaluation.__dict__[opt.distill_loss]
    task_cache=TaskCache(opt.checkpoints_dir,tasks,opt)
    for task_index in range(start_index,len(tasks)):
        task=tasks[task_index]
        if task_index==0:
            model.set_mode(task)
            lwf_train(task,model,task.train_data_loader,opt.num_epochs,opt,visualizer=visualizer,limit_updates=opt.limit_updates,in_parallel=opt.in_parallel,distill_loss=distill_loss)
            model.unset_mode(task)
            previous_task=task
        else:
            task_cache.reset_index()
            store_responses(model,task,prev_tasks,task_cache,opt,limit_updates=opt.limit_updates)
            model.set_mode(task)
            lwf_train(task,model,task.train_data_loader,opt.num_epochs,opt, task_cache=task_cache,visualizer=visualizer,prev_tasks=prev_tasks,limit_updates=opt.limit_updates,in_parallel=opt.in_parallel,distill_loss=distill_loss)
            model.unset_mode(task)
            previous_task=task
        if opt.same_task:
            for s_task_index in range( task_index+1,len(tasks)):
                s_task=tasks[s_task_index]
                model.modify_model(s_task,previous_load=task)
        test_results=test(tasks,model,opt,limit_updates=opt.limit_test_updates)
        task_loss_after_each_train.append(test_results)
        prev_tasks.append(task)
elif opt.model_setting=='uncertainty_distill':
    print 'uncertainty_distill'
    if start_index>0:
        for k in range(start_index):
            prev_tasks.append(tasks[k])
    distill_loss = evaluation.__dict__[opt.distill_loss]
    uncertainty_func = evaluation.__dict__[opt.uncertainty_func]
    task_cache=TaskCache(opt.checkpoints_dir,tasks,opt)
    for task_index in range(start_index,len(tasks)):
        task=tasks[task_index]
        if task_index==0:
            model.set_mode(task)
            uncertainty_train(task,model,task.train_data_loader,opt.num_epochs,opt,visualizer=visualizer,limit_updates=opt.limit_updates,in_parallel=opt.in_parallel,distill_loss=distill_loss,uncertainty_func=uncertainty_func)
            model.unset_mode(task)
            previous_task=task
        else:
            task_cache.reset_index()
            store_responses(model,task,prev_tasks,task_cache,opt,limit_updates=opt.limit_updates)
            model.set_mode(task)
            uncertainty_train(task,model,task.train_data_loader,opt.num_epochs,opt, task_cache=task_cache,visualizer=visualizer,prev_tasks=prev_tasks,limit_updates=opt.limit_updates,in_parallel=opt.in_parallel,distill_loss=distill_loss,uncertainty_func=uncertainty_func)
            model.unset_mode(task)
            previous_task=task
        if opt.same_task:
            for s_task_index in range( task_index+1,len(tasks)):
                s_task=tasks[s_task_index]
                model.modify_model(s_task,previous_load=task)
        test_results=test(tasks,model,opt,limit_updates=opt.limit_test_updates)
        task_loss_after_each_train.append(test_results)
        prev_tasks.append(task)
elif opt.model_setting=='ewc_train':
    print 'ewc_train'
    for task_index in range(start_index,len(tasks)):
        task=tasks[task_index]
        if task_index==0:
            model.set_mode(task)
            ewf_train(task,model,task.train_data_loader,opt.num_epochs,opt,visualizer=visualizer,limit_updates=opt.limit_updates,in_parallel=opt.in_parallel)
            model.unset_mode(task)
            previous_task=task
        else:
            model.set_mode(task)
            ewf_train(task,model,task.train_data_loader,opt.num_epochs,opt,visualizer=visualizer,prev_tasks=prev_tasks,limit_updates=opt.limit_updates,in_parallel=opt.in_parallel)
            model.unset_mode(task)
            previous_task=task
        if opt.same_task:
            for s_task_index in range( task_index+1,len(tasks)):
                s_task=tasks[s_task_index]
                model.modify_model(s_task,previous_load=task)
        test_results=test(tasks,model,opt,limit_updates=opt.limit_test_updates)
        task_loss_after_each_train.append(test_results)
        prev_tasks.append(task)

log_results(task_loss_after_each_train,tasks)
