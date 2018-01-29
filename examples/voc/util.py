from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from pdb import set_trace as st
def unique(tensor1d):
    t, idx = np.unique(tensor1d.numpy(), return_inverse=True)
    return torch.from_numpy(t), torch.from_numpy(idx)


def load_pretrained(file_name):
    return torch.load(file_name)
# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap='plasma'):
    # convert to disparity
    depth = 1./(depth + 1e-6)
    if normalizer is not None:
        depth = depth/normalizer
    else:
        depth = depth/(np.percentile(depth, pc) + 1e-6)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]
    depth = depth*255
    return depth

def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def add_custom_settings(opt,custom_settings_file):
    import json
    custom_settings =None
    with open(custom_settings_file) as json_data:   
        json_data=json_data.read()
        print(json_data)
        custom_settings = json.loads(json_data)
    for i in custom_settings:
        t=custom_settings[i]
        if t=='True' or t=='False':
            t= (t=='True')
        opt.__dict__[i]=t
    return opt

# def store_visualizations(visualize_dir,results,tasks):
#     #st()
#     mean_pixel_acc=np.zeros( (len(tasks),len(tasks),) )
#     mean_class_acc=np.zeros( (len(tasks),len(tasks),) )
#     mean_class_iou=np.zeros( (len(tasks),len(tasks),) )
#     for i in range(len(tasks)):
#         print 'Task[{}] '.format(tasks[i].name)
#         print 'Num of Classes: [{}]'.format(tasks[i].num_classes)
#         print  'Root Directory Name: [{}]'.format(tasks[i].root_dir)

#     for i in range(len(tasks)):
#         for j in range(len(tasks)):
#             mean_pixel_acc[j][i]=task_matrix[i][j].mean_pixel_acc

#     for i in range(len(tasks)):
#         for j in range(len(tasks)):
#             mean_class_acc[j][i]=task_matrix[i][j].mean_class_acc

#     for i in range(len(tasks)):
#         for j in range(len(tasks)):
#             mean_class_iou[j][i]=task_matrix[i][j].mean_class_iou

#     name='Grouped Cityscapes'
#     visualize_matrix('MeanClassIOU of '+name,grouped,task_grouped)
#     visualize_matrix('DifferencesIOUMap(Pretrained) of '+name,subtract(grouped,pretrainedgrouped,task_grouped),task_grouped)
#     visualize_matrix('DifferencesIOUMap(Multitask) of '+name,subtract(grouped,multitaskgrouped,task_grouped),task_grouped)
#     statistics(name,grouped)

#     name='Individual Cityscapes'
#     visualize_matrix('MeanClassIOU of '+name,individual,tasks)
#     visualize_matrix('DifferencesIOUMap(Pretrained) of '+name,subtract(individual,pretrained,tasks),tasks)
#     visualize_matrix('DifferencesIOUMap(Multitask) of '+name,subtract(individual,multitask,tasks),tasks)
#     statistics(name,individual)


def subtract(a,b,tasks):
    k=np.zeros_like(a)
    for i in range(len(tasks)):
        k[:,i]=a[:,i]-b
    return k

def visualize_matrix(name,matrix,tasks):
    # Plot it out
    fig, ax = plt.subplots()
    fig.canvas.set_window_title(name)
    heatmap = ax.pcolor(matrix, cmap=plt.cm.Blues, alpha=1)

    ##################################################
    ## FORMAT ##
    ##################################################

    fig = plt.gcf()
    fig.suptitle(name, fontsize=20)
    fig.set_size_inches(8,11)
    fig.colorbar(heatmap)
    # turn off the frame
    ax.set_frame_on(False)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(matrix.shape[0])+0.5, minor=False)
    ax.set_xticks(np.arange(matrix.shape[1])+0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # Set the labels

    # label source:https://en.wikipedia.org/wiki/Basketball_statistics

    # note I could have used nba_sort.columns but made "labels" instead
    ax.set_xticklabels(tasks, minor=False) 
    ax.set_yticklabels(tasks, minor=False)

    # rotate the 
    plt.xticks(rotation=90)

    ax.grid(False)

    # Turn off all the ticks
    ax = plt.gca()

    for t in ax.xaxis.get_major_ticks(): 
        t.tick1On = False 
        t.tick2On = False 
    for t in ax.yaxis.get_major_ticks(): 
        t.tick1On = False 
        t.tick2On = False  
    plt.show()

def statistics(name,matrix):
    mean_statistics= [ np.mean(matrix[:,i]) for i in range(len(matrix))]
    variance_statistics= [ np.var(matrix[:,i]) for i in range(len(matrix))]
    plt.figure(1)
    fig=plt.gcf()
    fig.suptitle('Mean and Variance plots of '+name, fontsize=20)
    plt.subplot(211)
    plt.plot(mean_statistics)
    plt.subplot(212)
    plt.plot(variance_statistics)
    plt.show()

