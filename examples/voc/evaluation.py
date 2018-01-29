
# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
import torch
from torch.nn import functional as F
from pdb import set_trace as st
"""
    Loss Functions:
"""
def cross_entropy2d(inputV, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = inputV.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(inputV)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target.long(), weight=weight, size_average=False)
    if size_average:
        # print mask.data.sum()
        loss /= mask.data.sum()
    return loss

def compute_l2_loss(inputV ,target ,mask=None ,weight=None):
    if mask!= None:
        target = target * mask
        inputV = inputV * mask
    l2_loss = torch.nn.MSELoss()
    return l2_loss(inputV, target)


def compute_ce_loss(inputV, target, mask=None, weight=None):
    sigmoid = torch.nn.Sigmoid()
    return torch.nn.BCELoss()(sigmoid(inputV), sigmoid(target))
    #return torch.nn.BCEWithLogitsLoss()(inputV,target)

def seg_loss(inputV, target, weight=None, size_average=True):
    return cross_entropy2d(inputV, target, weight=weight, size_average=size_average)


# def compute_hard_distill_loss(student, teacher, mask=None, T=1, distillation_constant=1.1):
#     if mask!=None:
#         target = target * mask
#         inputV = inputV * mask
#     return cross_entropy2d(student, teacher) * distillation_constant


"""
    Evaluation methods:
    Segmentation
    &
    Depth Estimation

"""


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def overall_scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))
    return (acc, acc_cls, fwavacc, mean_iu, cls_iu)


def compute_depth_errors(gt, pred):
    gt = gt.view(-1).numpy()
    pred = pred.view(-1).numpy()
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


"""
    Uncertainty over all pixels
    Input:
        S x B x C x H x W (S=samples, C = Classes, HxW = dimensions of the images)
    Output:
        B x C x H x W
"""


def global_uncertainty(inputT):
    uncertainty_batch = []
    for i in range(inputT.size()[1]):
        scores_p = [inputT[j][i].numpy() for j in range(inputT.size()[0])]
        scores_p = np.array(scores_p)
        uncertainty = np.var(scores_p, axis=0)
        uncertainty = np.transpose(uncertainty, (1, 2, 0))
        average_unc = np.mean(uncertainty, axis=2)
        min_average_unc = np.min(average_unc)
        max_average_unc = np.max(average_unc)
        average_unc = average_unc / max_average_unc
        uncertainty_batch.append(average_unc)
    uncertainty = torch.from_numpy(np.array(uncertainty_batch))
    return uncertainty.unsqueeze(1)

def global_uncertainty_fast(inputT):
    uncertainty_batch = []
    for i in range(inputT.size()[1]):
        scores_p = [inputT[j][i] for j in range(inputT.size()[0])]
        uncertainty = torch.var(torch.stack(scores_p), 0)
        average_unc = torch.transpose(torch.transpose(uncertainty,1,2),0,2)
        average_unc = torch.mean(average_unc,2)
        max_average_unc = torch.max(average_unc)
        average_unc = average_unc / max_average_unc
        uncertainty_batch.append(average_unc)
    uncertainty= torch.stack(uncertainty_batch)
    return uncertainty.unsqueeze(1)

"""
    Uncertainty over all pixels
    Input:
        S x B  x H x W (S=samples, C = Classes, HxW = dimensions of the images)
    Output:
        B x H x W
"""


def global_uncertainty_depth(inputT):
    uncertainty_batch = []
    for i in range(inputT.size()[1]):
        scores_p = inputT[:,i].numpy()
        uncertainty = np.var(scores_p, axis=0)
        max_average_unc = np.max(uncertainty)
        uncertainty = uncertainty / max_average_unc
        uncertainty_batch.append(uncertainty)
    return torch.from_numpy(np.array(uncertainty_batch))


"""
    Class-wise uncertainty
    Input:
        B x S x C x H x W (S=samples, C = Classes, HxW = dimensions of the images)
    Output:
        B x C x H x W
"""


def class_uncertainty(inputT):
    uncertainty_batch = []
    for i in range(inputT.size()[1]):
        scores_p = [inputT[j][i].cpu().numpy() for j in range(inputT.size()[0])]
        scores_p = np.array(scores_p)
        class_unct = []
        for j in range(inputT.size()[2]):
            uncertainty = np.var(scores_p[:,j], axis=0)
            max_average_unc = np.max(uncertainty)
            uncertainty = uncertainty / max_average_unc
            class_unct.append(uncertainty)
        uncertainty_batch.append(np.array(class_unct))
    return torch.from_numpy(np.array(uncertainty_batch))


"""
    Calculate entropy for each pixel
    Input:
        B x C x H x W (S=samples, C = Classes, HxW = dimensions of the images)
    Output:
        B x H x W
"""


def entropy_uncertainty(inputT):
    scores_p = inputT.cuda()
    scores_p = scores_p[0]
    p = torch.nn.Softmax2d()(torch.autograd.Variable(scores_p).cuda()).data
    uncertainty = torch.mean(-p * F.log_softmax(torch.autograd.Variable(scores_p).cuda()).data,1)
    max_average_unc = torch.max(uncertainty)
    average_unc = uncertainty / max_average_unc
    return average_unc.unsqueeze(1)
"""
    Max-value based uncertainty.
    Find the max value among all the classes.
    The max value normalized between 0-1 becomes the uncertainty score for the pixel
    Input:
        S x C x H x W (S=samples, C = Classes, HxW = dimensions of the images)
    Output:
        C x H x W
"""
def max_uncertainty(inputT):
    inputT=inputT[0]
    s = inputT.size()
    inputT = torch.nn.Softmax2d()(torch.autograd.Variable(inputT)).data
    uncertainty = torch.zeros(s[0],s[2],s[3])
    for b in range(s[0]):
        for i in range(s[2]):
            for j in range(s[3]):
                uncertainty[b,i,j] = torch.max(inputT[b,:,i,j])
        uncertainty[b] = uncertainty[b] / torch.max(uncertainty[b])
    return 1- uncertainty.unsqueeze(1)

if __name__ == "__main__":
    input = torch.randn(3,10, 500, 500).random_(0,5)
    out = torch.randn(3, 500, 500).random_(0,5)
    print overall_scores(torch.autograd.Variable(input).data.numpy().argmax(0),torch.autograd.Variable(out).data.int().numpy(),10)
