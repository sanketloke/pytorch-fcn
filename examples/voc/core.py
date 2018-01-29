import torch
import torch.optim
from tqdm import *
from misc import *
from torch import nn
import copy
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from evaluation import overall_scores, cross_entropy2d
from util import normalize_depth_for_display
from torch.nn import functional as F
plt.switch_backend('agg')
from distutils.version import LooseVersion


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def to_rgb1(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def console_log(task_name,epoch,num_epochs,iterations,total_iterations,loss):
    print('Task[%s]   Epoch [%d/%d], Iter [%f/%f] Loss: %.4f'
          % (task_name.name, epoch + 1, num_epochs, iterations + 1, total_iterations, loss))

def train(task, model, data_loader, num_epochs, opt, in_parallel=False, visualizer=None, limit_updates=0):
    """ Trains model on given task
    Keyword arguments:
    task: Task Object
    model: Model to be trained
    num_epochs: How many epochs to train?
    opt: Options Argument File
    in_parallel: run on multiple gpus?

    Returns None
    """
    model.set_mode(task)
    model = model.cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters(
    )), lr=opt.lr, momentum=0.99, weight_decay=0.0005)
    visualize = opt.visualize
    web_visualize = opt.web_visualize
    record_parameters = opt.record_parameters

    # Initialize Criterion
    if task.type == 'depth':
        criterion = nn.MSELoss()
    else:
        if task.task_type == 'edges':
            criterion = nn.NLLLoss2d(ignore_index=task.ignore_index, weight=torch.cuda.FloatTensor([0.1, 0.9]))
        else:
            criterion = cross_entropy2d # nn.NLLLoss2d(ignore_index=task.ignore_index)

    if in_parallel == 1:
        model_p = torch.nn.DataParallel(model)

    # Load Dataset
    dataset = data_loader.dataset
    dataset.load_label_map()

    # Loss Meters
    if opt.validation_setting == 1:
        validation_losses = []
    else:
        validation_losses = {}
        for t in model.tasks:
            validation_losses[t] = []
    validation_losses = []

    # Counter Variable
    updates = 0


    for epoch in range(num_epochs):
        iterations = 0
        for i, (images, labels) in tqdm(enumerate(data_loader)):
            model.zero_grad()

            # Verify if tensor
            if type(images) == list:
                img_batch = images[0]
                label_batch = labels[0]
            else:
                img_batch = images
                label_batch = labels

            # Ready Input
            input = torch.autograd.Variable(img_batch)
            target = torch.autograd.Variable(label_batch.float())
            input = input.cuda()
            target = target.cuda()

            optimizer.zero_grad()
            if in_parallel:
                scores = model_p.forward(input)
            else:
                scores = model.forward(input)

            if task.type == 'depth':
                scores = F.softmax(scores)
                loss = criterion(scores, target)
            else:
                loss = criterion(scores, target.long())
            loss.backward()

            optimizer.step()

            updates += 1
            if limit_updates > 0 and limit_updates <= updates:
                break



            if (iterations + 1) % opt.display_freq == 0:
                print('Task[%s]   Epoch [%d/%d], Iter [%f/%f] Loss: %.4f '
                      % (task.name, epoch + 1, num_epochs, iterations + 1, len(data_loader), loss.data[0]))
                if task.type == 'segment':
                    if task.task_type == 'edges':
                        label = target.cpu().data[0].numpy()
                        label[label > 0] = 1
                    else:
                        label = target.cpu().data[0].int(
                        ).numpy().astype(np.uint8)

                    predicted = scores[0].cpu().data.numpy().argmax(
                        0).astype(np.uint8)
                    #label = dataset.label2image(label)
                    #predicted = dataset.label2image(predicted)

                    #visuals = OrderedDict([('Image', tensor2im(img_batch)),
                    #                       ('Truth', label),
                    #                       ('Predicted', predicted)
                    #                       ])

                    #if web_visualize and visualizer:
                    #    visualizer.display_current_results(visuals, i)
                if task.type == 'depth':
                    label = target.cpu().data[0].numpy()[0]
                    predicted = scores[0].cpu().data.numpy()[0]
                    label = dataset.label2image(label)
                    predicted = dataset.label2image(predicted)

                    visuals = OrderedDict([('Image', tensor2im(img_batch)),
                                           ('Truth', label),
                                           ('Predicted', predicted)
                                           ])

                    if web_visualize and visualizer:
                        visualizer.display_current_results(visuals, i)
            iterations += 1

        if epoch % opt.save_freq == 0:
            try:
                # Avoiding the serialization bug here
                m = model.mode
                model.mode = None
                torch.save(model, opt.checkpoints_dir + "/" + opt.name + "/" +
                           model.__class__.__name__ + task.name + "_" + str(epoch) + ".pth")
                model.mode = m
            except:
                print "Error Occured"
        if opt.validation_freq != 0 and epoch % opt.validation_freq == 0:
            if opt.validation_setting == 1:  # if validation setting==1 indicates validate on only the current one , else validate on all
                validation_losses = validate(model, task.test_data_loader, plot_dir=task.plot_dir,
                                             validation_losses=validation_losses, task=task, samples=10)
            else:
                tasks = [model.tasks[t] for t in model.tasks]
                validation_losses = test(
                    tasks, model, opt, plot_dir=task.plot_dir, validation_losses=validation_losses)
                model.set_mode(task)
                model = model.cuda()

        if limit_updates > 0 and limit_updates <= updates:
            break

    model = model.cpu()
    # Avoiding the serialization bug here
    m = model.mode
    model.mode = None
    torch.save(model, opt.checkpoints_dir + "/" + opt.name + "/" +
               model.__class__.__name__ + task.name + "_final.pth")
    model.mode = m
    model.unset_mode(task)
    return model

def multi_train(tasks, model, num_epochs, opt, in_parallel=False, visualizer=None, mode='sequential', limit_updates=0):
    """ Trains model on given tasks

    Keyword arguments:
    task: Task Object
    model: Model to be trained
    num_epochs: How many epochs to train?
    opt: Options Argument File
    in_parallel: run on multiple gpus?

    Returns None
    """

    optimizers={}
    for task in model.task_parameters.keys():
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()) + list(model.task_parameters[task].parameters()), lr=opt.lr, momentum=0.9, weight_decay=0.0002)
        optimizers[task] = optimizer

    visualize = opt.visualize
    web_visualize = opt.web_visualize
    record_parameters = opt.record_parameters

    if in_parallel == 1:
        model_p = torch.nn.DataParallel(model)

    validation_losses = []

    updates = 0
    for epoch in range(num_epochs):

        for task in tasks:
            if task.type == 'depth':
                criterion = nn.MSELoss()
            else:
                if task.task_type == 'edges':
                    criterion = nn.NLLLoss2d(
                        ignore_index=task.ignore_index, weight=torch.cuda.FloatTensor([0.1, 0.9]))
                else:
                    criterion = nn.NLLLoss2d(ignore_index=task.ignore_index)
            optimizer = optimizers[task.name]
            data_loader = task.train_data_loader
            dataset = task.train_data_loader.dataset
            dataset.load_label_map()
            model.set_mode(task)
            model = model.cuda()
            iterations = 0
            #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=0.9, weight_decay=0.0002)
            for i, (images, labels) in tqdm(enumerate(task.train_data_loader)):

                if type(images) == list:
                    img_batch = images[0]
                    label_batch = labels[0]
                else:
                    img_batch = images
                    label_batch = labels
                input = torch.autograd.Variable(img_batch)
                target = torch.autograd.Variable(label_batch.float())
                input = input.cuda()
                target = target.cuda()

                optimizer.zero_grad()
                if in_parallel:
                    scores = model_p.forward(input)
                else:
                    scores = model.forward(input)

                if task.type == 'depth':
                    loss = criterion(scores, target)
                else:
                    if task.type == 'edges':
                        loss = criterion(F.log_softmax(
                            scores), target.long(), weight=[0.1, 0.9])
                    else:
                        loss = criterion(F.log_softmax(scores), target.long())

                loss.backward()

                optimizer.step()
                updates += 1
                if limit_updates > 0 and limit_updates <= updates:
                    break
                iterations += 1
                if (iterations + 1) % opt.display_freq == 0:
                    print('Task[%s]   Epoch [%d/%d], Iter [%f/%f] Loss: %.4f'
                          % (task.name, epoch + 1, num_epochs, iterations + 1, len(data_loader), loss.data[0]))
                    if task.type == 'classify':
                        logs = 0
                    if task.type == 'segment':
                        label = target.cpu().data[0].int(
                        ).numpy().astype(np.uint8)
                        predicted = scores[0].cpu().data.numpy().argmax(
                            0).astype(np.uint8)
                        label = dataset.label2image(label)
                        predicted = dataset.label2image(predicted)

                        visuals = OrderedDict([('Image', tensor2im(img_batch)),
                                               ('Truth', label),
                                               ('Predicted', predicted)
                                               ])
                        if web_visualize and visualizer:
                            visualizer.display_current_results(visuals, i)
                iterations += 1
            if limit_updates > 0 and limit_updates <= updates:
                break

            if epoch % opt.save_freq == 0:
                try:
                    torch.save(model, opt.checkpoints_dir + "/" + opt.name + "/" +
                               model.__class__.__name__ + task.name + "_" + str(epoch) + ".pth")
                except:
                    print "Error Occured"
            if opt.validation_freq != 0 and epoch % opt.validation_freq == 0:
                if opt.validation_setting == 1:  # if validation setting==1 indicates validate on only the current one , else validate on all
                    validation_losses = validate(
                        model, task.test_data_loader, plot_dir=task.plot_dir, validation_losses=validation_losses, task=task, samples=10)
                else:
                    tasks = [model.tasks[t] for t in model.tasks]
                    validation_losses = test(
                        tasks, model, opt, plot_dir=task.plot_dir, validation_losses=validation_losses)
                    model.set_mode(task)
                    model = model.cuda()

            model = model.cpu()
            model.unset_mode(task)
        if limit_updates > 0 and limit_updates <= updates:
            break
     # Avoiding the serialization bug here
    m = model.mode
    model.mode = None
    torch.save(model, opt.checkpoints_dir + "/" + opt.name + "/" +
               model.__class__.__name__ + task.name + "_final.pth")
    model.mode = m

    return model


def lwf_train(task, model, data_loader, num_epochs, opt, task_cache=None, prev_tasks=None, in_parallel=False, visualizer=None, limit_updates=0, distill_loss=None):
    """ Trains model on given task

    Keyword arguments:
    task: Task Object
    model: Model to be trained
    num_epochs: How many epochs to train?
    opt: Options Argument File
    in_parallel: run on multiple gpus?

    Returns None
    """
    model = model.cuda()



    visualize = opt.visualize
    web_visualize = opt.web_visualize
    record_parameters = opt.record_parameters

    if task.type == 'depth':
        criterion = nn.MSELoss()
    else:
        if task.task_type == 'edges':
            criterion = nn.NLLLoss2d(
                ignore_index=task.ignore_index, weight=torch.cuda.FloatTensor([0.1, 0.9]))
        else:
            criterion = nn.NLLLoss2d(ignore_index=task.ignore_index)

    if in_parallel == 1:
        model_p = torch.nn.DataParallel(model)
    dataset = data_loader.dataset
    dataset.load_label_map()


    if opt.validation_setting == 1:
        validation_losses = []
    else:
        validation_losses = {}
        for t in model.tasks:
            validation_losses[t] = []

    validation_losses = []

    if task_cache is not None:
        model.load_cuda_tasks(prev_tasks)
        task_cache.reset_index()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters(
    )), lr=opt.lr, momentum=0.9, weight_decay=0.0002)
    updates = 0
    for epoch in range(num_epochs):
        iterations = 0
        for i, (images, labels) in tqdm(enumerate(data_loader)):
            if type(images) == list:
                # Tensor # TODO Ensure the tensors are sent by the dataset
                # instead of tuple
                img_batch = images[0]
                label_batch = labels[0]  # Tensor
            else:
                img_batch = images
                label_batch = labels

            input = torch.autograd.Variable(img_batch).cuda()
            target = torch.autograd.Variable(label_batch.float()).cuda()
            input = input.cuda()
            target = target.cuda()

            optimizer.zero_grad()
            if in_parallel:
                scores = model_p.forward(input)
            else:
                scores = model.forward(input)
            if task.type == 'depth':
                loss = criterion(scores, target)
            else:
                loss = criterion(F.log_softmax(scores), target.long())
            losses = []
            if task_cache is not None:
                prev_tasks_out = model.forward_custom(input, prev_tasks)
                for t in prev_tasks:
                    image_old, score_old = task_cache.get(t.name, i)
                    print input.data.cpu().equal(image_old)
                    losses.append(distill_loss(prev_tasks_out[t.name], torch.autograd.Variable(score_old.cuda()),
                                               1.5, 0.3))
            for l in losses:
                loss += l

            loss.backward()
            optimizer.step()
            updates += 1
            if limit_updates > 0 and limit_updates <= updates:
                break

            if (iterations + 1) % opt.display_freq == 0:
                print('Task[%s]   Epoch [%d/%d], Iter [%f/%f] Loss: %.4f'
                    % (task.name, epoch + 1, num_epochs, iterations + 1, len(data_loader), loss.data[0]))
                if task.type == 'segment':
                    label = target.cpu().data[0].int().numpy().astype(np.uint8)
                    predicted = scores[0].cpu().data.numpy().argmax(0).astype(np.uint8)
                    label = dataset.label2image(label)
                    predicted = dataset.label2image(predicted)
                    visuals = OrderedDict([('Image', tensor2im(img_batch)),
                                           ('Truth', label),
                                           ('Predicted', predicted)
                                           ])
                    if web_visualize and visualizer:
                        visualizer.display_current_results(visuals, i)
            iterations += 1

        if epoch % opt.save_freq == 0:
            try:
                # Avoiding serialization bug here.
                m = model.mode
                model.mode = None
                torch.save(model, opt.checkpoints_dir + "/" + opt.name + "/" +
                           model.__class__.__name__ + task.name + "_" + str(epoch) + ".pth")
                model.mode = m
            except:
                print "Error Occured"

        if opt.validation_freq != 0 and epoch % opt.validation_freq == 0:
            if opt.validation_setting == 1:  # if validation setting==1 indicates validate on only the current one , else validate on all
                validation_losses = validate(model, task.test_data_loader, plot_dir=task.plot_dir,
                                             validation_losses=validation_losses, task=task, samples=10)
            else:
                tasks = [model.tasks[t] for t in model.tasks]
                validation_losses = test(
                    tasks, model, opt, plot_dir=task.plot_dir, validation_losses=validation_losses)
                model.set_mode(task)
                model = model.cuda()
        if limit_updates > 0 and limit_updates <= updates:
            break

    if task_cache is not None:
        model.unload_cuda_tasks(prev_tasks)

    # Avoiding the serialization bug here
    m = model.mode
    model.mode = None
    torch.save(model, opt.checkpoints_dir + "/" + opt.name + "/" +
               model.__class__.__name__ + task.name + "_final.pth")
    model.mode = m

    model = model.cpu()
    return model


def distill_custom_loss(input, target):
    sigmoid = nn.Sigmoid()
    return torch.nn.BCELoss()(sigmoid(input), sigmoid(target))

def uncertainty_train(task, model, data_loader, num_epochs, opt, task_cache=None, prev_tasks=None, in_parallel=False, visualizer=None, limit_updates=0, distill_loss=None, uncertainty_func=None, uncertainty_samples=10):
    """ Trains model on given task

    Keyword arguments:
    task: Task Object
    model: Model to be trained
    num_epochs: How many epochs to train?
    opt: Options Argument File
    in_parallel: run on multiple gpus?

    Returns None
    """
    model = model.cuda()
    #optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters()) , lr=opt.lr, betas=(opt.betas1,0.999))
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters(
    #)), lr=opt.lr, momentum=0.9, weight_decay=0.0002)
    # logger=task.logger
    visualize = opt.visualize
    web_visualize = opt.web_visualize
    record_parameters = opt.record_parameters

    if task.type == 'depth':
        criterion = distill_custom_loss  # nn.MSELoss()
    else:
        if task.task_type == 'edges':
            criterion = nn.NLLLoss2d(
                ignore_index=task.ignore_index, weight=torch.cuda.FloatTensor([0.1, 0.9]))
        else:
            criterion = nn.NLLLoss2d(ignore_index=task.ignore_index)

    if in_parallel == 1:
        model_p = torch.nn.DataParallel(model)
    dataset = data_loader.dataset
    dataset.load_label_map()

    if opt.validation_setting == 1:
        validation_losses = []
    else:
        validation_losses = {}
        for t in model.tasks:
            validation_losses[t] = []
    model.train()
    validation_losses = []
    if task_cache is not None:
        model.load_cuda_tasks(prev_tasks)
        task_cache.reset_index()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()) + list(model.task_parameters[task.name].parameters()), lr=opt.lr, momentum=0.9, weight_decay=0.0002)
    updates = 0
    for epoch in range(num_epochs):
        iterations = 0
        for i, (images, labels) in tqdm(enumerate(data_loader)):
            if type(images) == list:
                # Tensor # TODO Ensure the tensors are sent by the dataset
                # instead of tuple
                img_batch = images[0]
                label_batch = labels[0]  # Tensor
            else:
                img_batch = images
                label_batch = labels
            input = torch.autograd.Variable(img_batch)
            target = torch.autograd.Variable(label_batch.float())
            input = input.cuda()
            target = target.cuda()
            score_dict = {}
            if prev_tasks and len(prev_tasks) > 0:
                for j in prev_tasks:
                    score_dict[j.name] = []
                inpt2 = torch.autograd.Variable(input.data, volatile=True).cuda()
                for _ in range(uncertainty_samples):
                    if in_parallel:
                        scores= model.forward_custom(inpt2,prev_tasks)
                    else:
                        scores= model.forward_custom(inpt2,prev_tasks)
                    for t in scores.keys():
                        score_dict[t].append(scores[t].cpu())
                auc = None
                for t in score_dict:
                    score_lst = score_dict[t]
                    average_unc = uncertainty_func(
                        torch.stack(score_lst).data.cpu())
                    u2 = average_unc
                    if auc == None:
                        auc = average_unc.cpu()
                    weight_matrix = 1 - u2
                    weight_matrix = torch.autograd.Variable(
                        weight_matrix.cuda())
                    score_dict[t] = weight_matrix
                    # size = score_dict[t].size()
                    # if len(size)<4:
                    #     weight_matrix.data.resize_(size[0],1,size[1],size[2])
                    #     weight_matrix.data = weight_matrix.data.repeat(1,int(score_lst[0].size()[1]),1,1)
            optimizer.zero_grad()
            if in_parallel:
                scores = model_p.forward(input)
            else:
                scores = model.forward(input)
            if task.type == 'depth':
                loss = criterion(scores, target)
            else:
                loss = criterion(F.log_softmax(scores), target.long())

            losses = []
            inpt2 = None
            score_lst = None
            if task_cache is not None:
                prev_tasks_out = model.forward_custom(input, prev_tasks)
                for t in prev_tasks:
                    image_old, score_old = task_cache.get(t.name, i)
                    losses.append(distill_loss(prev_tasks_out[t.name] * score_dict[
                                  t.name], torch.autograd.Variable(score_old.cuda()) * score_dict[t.name]))
            for l in losses:
                loss += l
            loss.backward()
            score_dict = {}
            optimizer.step()
            updates += 1
            if limit_updates > 0 and limit_updates <= updates:
                break
            if (iterations + 1) % opt.display_freq == 0:
                print('Task[%s]   Epoch [%d/%d], Iter [%f/%f] Loss: %.4f'
                      % (task.name, epoch + 1, num_epochs, iterations + 1, len(data_loader), loss.data[0]))
                if task.type == 'classify':
                    logs = 0
                if task.type == 'segment':
                    label = target.cpu().data[0].int().numpy().astype(np.uint8)
                    predicted = scores[0].cpu().data.numpy().argmax(
                        0).astype(np.uint8)
                    label = dataset.label2image(label)
                    predicted = dataset.label2image(predicted)
                    if prev_tasks and len(prev_tasks) > 0:
                        average_unc = auc[0]
                        average_unc = np.mean(average_unc.numpy(), axis=0)
                        average_unc = average_unc * 255.0
                        average_unc = np.array(
                            Image.fromarray(average_unc).convert('RGB'))
                        visuals = OrderedDict([('Image', tensor2im(
                            img_batch)), ('Truth', label), ('Predicted', predicted), ('Uncertainty', average_unc)])
                    else:
                        visuals = OrderedDict([('Image', tensor2im(img_batch)),
                                               ('Truth', label),
                                               ('Predicted', predicted)
                                               ])
                    if web_visualize and visualizer:
                        visualizer.display_current_results(visuals, i)
            iterations += 1
        if epoch % opt.save_freq == 0:
            try:
                # Avoiding serialization bug here.
                m = model.mode
                model.mode = None
                torch.save(model, opt.checkpoints_dir + "/" + opt.name + "/" +
                           model.__class__.__name__ + task.name + "_" + str(epoch) + ".pth")
                model.mode = m
            except:
                print "Error Occured"
        if opt.validation_freq != 0 and epoch % opt.validation_freq == 0:
            if opt.validation_setting == 1:  # if validation setting==1 indicates validate on only the current one , else validate on all
                validation_losses = validate(model, task.test_data_loader, plot_dir=task.plot_dir,
                                             validation_losses=validation_losses, task=task, samples=10)
            else:
                tasks = [model.tasks[t] for t in model.tasks]
                validation_losses = test(
                    tasks, model, opt, plot_dir=task.plot_dir, validation_losses=validation_losses)
                model.set_mode(task)
                model = model.cuda()

        if limit_updates > 0 and limit_updates <= updates:
            break

    if task_cache is not None:
        model.unload_cuda_tasks(prev_tasks)

    # Avoiding the serialization bug here
    m = model.mode
    model.mode = None
    torch.save(model, opt.checkpoints_dir + "/" + opt.name + "/" +
               model.__class__.__name__ + task.name + "_final.pth")
    model.mode = m
    model.train()
    model = model.cpu()
    return model


def validate(model, data_loader, validation_losses, plot_dir, task, samples):
    print 'Validating'
    iou = []
    datasetLength = len(data_loader.dataset)
    indices = random.sample(range(datasetLength), samples)
    for index in indices:
        images, labels = data_loader.dataset[index]
        if type(images) == list or type(images) == tuple:
            # Tensor # TODO Ensure the tensors are sent by the dataset instead
            # of tuple
            img_batch = images[0]
            label_batch = labels[0]  # Tensor
        else:
            img_batch = images
            label_batch = labels
        img_batch = img_batch.unsqueeze(0)
        input = torch.autograd.Variable(img_batch, volatile=True)
        target = torch.autograd.Variable(label_batch.float())
        input = input.cuda()
        target = target.cuda()
        scores = model.forward(input)
        if task.type == 'segment':
            score_img_numpy = scores.cpu().data[0].numpy().argmax(0)
            target_img_numpy = target.cpu().data.int().numpy()
            if np.min(target_img_numpy) < 0:
                target_img_numpy[target_img_numpy < 0] = 150
            mean_pixel_acc, mean_class_acc, fwavacc, mean_class_iou, per_class_iou = overall_scores(
                score_img_numpy, target_img_numpy, task.num_classes)  # ERROR might occur over here
            iou.append(mean_class_iou)
    validation_losses.append(np.mean(iou))
    plt.plot(validation_losses)
    plt.savefig(plot_dir + '/validationplot.png')
    return validation_losses


def test(tasks, model, opt, visualizer=None, plot_dir=None, validation_losses=None, limit_updates=0, uncertainty_visualize=None, uncertainty_func=None):
    """ Test model over set of tasks

    Arguments:
    tasks: Set of tasks
    model: Model

    Returns None
    """
    # Temporary
    print "Testing on all tasks"
    k = opt.batchSize

    opt.batchSize = 1
    task_losses = []
    web_visualize = True
    # Record in losses object. Used later visualization
    for task in tasks:
        task_losses.append(Loss(task))
    updates = 0
    for i, task in enumerate(tasks):
        print 'Testing on ' + task.name
        model.set_mode(task)
        model = model.cuda()
        data_loader = task.test_data_loader
        dataset = data_loader.dataset
        dataset.load_label_map()
        print 'Task:' + task.name

        if task.type == 'depth':
            criterion = nn.MSELoss()
        else:
            criterion = nn.NLLLoss2d(ignore_index=task.ignore_index)

        iou = []

        model = model.eval()
        for j, (images, labels) in tqdm(enumerate(data_loader)):
            if type(images) == list:
                # Tensor # TODO Ensure the tensors are sent by the dataset
                # instead of tuple
                img = images[0]
                label = labels[0]  # Tensor
            else:
                img = images
                label = labels
            input = torch.autograd.Variable(img, volatile=True)
            target = torch.autograd.Variable(label.float())
            input = input.cuda()
            target = target.cuda()

            scores = model.forward(input)
            if uncertainty_visualize:
                score_dict = {}
                inpt2 = torch.autograd.Variable(
                    input.data, volatile=True).cuda()
                prev_tasks = [tasks[0]]
                for j in prev_tasks:
                    score_dict[j.name] = []
                model.load_cuda_tasks(prev_tasks)
                score_lst = model.forward_custom(
                    inpt2, prev_tasks, uncertainty_samples=10)
                model.unload_cuda_tasks(prev_tasks)
                model.set_mode(task)
                model.cuda()
                for sc in score_lst:
                    for t in sc.keys():
                        score_dict[t].append(sc[t])

            if task.type == 'depth':
                loss = criterion(scores, target)
            else:
                if task.type == 'edges':
                    loss = criterion(F.log_softmax(scores), target.long())
                else:
                    loss = criterion(F.log_softmax(scores), target.long())
            print updates
            updates += 1
            if task.type == 'segment':
                for j in range(input.size()[0]):
                    score_img_numpy = scores.cpu().data[j].numpy().argmax(0)
                    target_img_numpy = target.cpu().data[j].int().numpy()
                    if np.min(target_img_numpy) < 0:
                        target_img_numpy[target_img_numpy < 0] = 150
                    mean_pixel_acc, mean_class_acc, fwavacc, mean_class_iou, per_class_iou = overall_scores(
                        score_img_numpy, target_img_numpy, task.num_classes)

                    task_losses[i].aggregate(
                        mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_iou, per_class_iou, loss.data[0])
                    iou.append(mean_class_iou)
            elif task.type == 'classify':
                task_losses[i].aggregate(scores.numpy(), target.numpy())
            if task.type == 'segment':

                label = target.cpu().data[0].int().numpy().astype(np.uint8)
                predicted = scores[0].cpu().data.numpy().argmax(
                    0).astype(np.uint8)
                label = dataset.label2image(label)
                predicted = dataset.label2image(predicted)

                if uncertainty_visualize:
                    predicted2 = score_lst[
                        0].cpu().data.numpy().argmax(0).astype(np.uint8)
                    from evaluation import global_uncertainty_fast
                    score_lst = score_dict[tasks[1].name]
                    average_unc = global_uncertainty_fast(
                        torch.stack(score_lst).data.cpu())
                    average_unc = average_unc.cpu()
                    average_unc = np.mean(average_unc.numpy(), axis=0)
                    average_unc = average_unc * 255.0
                    average_unc = average_unc[0]
                    average_unc1 = np.array(
                        Image.fromarray(average_unc).convert('RGB'))
                    average_unc = entropy_uncertainty(
                        torch.stack(score_lst).data.cpu())
                    average_unc = average_unc.cpu()
                    average_unc = np.mean(average_unc.numpy(), axis=0)
                    average_unc = average_unc * 255.0
                    average_unc = average_unc[0]
                    average_unc2 = np.array(
                        Image.fromarray(average_unc).convert('RGB'))
                    visuals = OrderedDict([('Image', tensor2im(img)), ('Truth', label), ('Predicted (Current Task)', predicted), (
                        'Uncertainty', average_unc1), ('Uncertainty', average_unc2), ('Predicted (Old Task)', predicted2)])
                else:
                    visuals = OrderedDict([('Image', tensor2im(img)),
                                           ('Truth', label),
                                           ('Predicted', predicted)
                                           ])

                if web_visualize and visualizer:
                    visualizer.display_current_results(visuals, updates)
            if limit_updates > 0 and limit_updates <= updates:
                print "Reached Updates Limit"
                break
        print 'Changing task'
        print updates
        if validation_losses:
            validation_losses[task.name].append(np.mean(iou))
        task_losses[i].display()
        metrics = task_losses[i].get_scalars()
        model.train()
        model = model.cpu()
        model.unset_mode(task)
        print 'Task' + task.name + ' Done!!'

    opt.batchSize = k
    if validation_losses:
        for i in validation_losses:
            plt.plot(i)
        plt.savefig(plot_dir + '/validationplot.png')
    return task_losses


def ewf_train(task, model, data_loader, num_epochs, opt, prev_tasks=None, in_parallel=False, visualizer=None, limit_updates=0):
    """ Trains model on given task

    Keyword arguments:
    task: Task Object
    model: Model to be trained
    num_epochs: How many epochs to train?
    opt: Options Argument File
    in_parallel: run on multiple gpus?

    Returns None
    """
    model = model.cuda()
    #optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters()) , lr=opt.lr, betas=(opt.betas1,0.999))
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters(
    )), lr=opt.lr, momentum=0.9, weight_decay=0.0002)
    # logger=task.logger
    visualize = opt.visualize
    web_visualize = opt.web_visualize
    record_parameters = opt.record_parameters

    if task.type == 'depth':
        criterion = distill_custom_loss  # nn.MSELoss()
    else:
        if task.task_type == 'edges':
            criterion = nn.NLLLoss2d(
                ignore_index=task.ignore_index, weight=torch.cuda.FloatTensor([0.1, 0.9]))
        else:
            criterion = nn.NLLLoss2d(ignore_index=task.ignore_index)

    if in_parallel == 1:
        model_p = torch.nn.DataParallel(model)
    dataset = data_loader.dataset
    dataset.load_label_map()

    if opt.validation_setting == 1:
        validation_losses = []
    else:
        validation_losses = {}
        for t in model.tasks:
            validation_losses[t] = []

    validation_losses = []
    updates = 0

    optimizer.zero_grad()
    model.compute_fisher(prev_tasks,criterion)
    optimizer.step()
    for epoch in range(num_epochs):
        iterations = 0
        for i, (images, labels) in tqdm(enumerate(data_loader)):
            if type(images) == list:
                img_batch = images[0]
                label_batch = labels[0]  # Tensor
            else:
                img_batch = images
                label_batch = labels
            input = torch.autograd.Variable(img_batch)
            target = torch.autograd.Variable(label_batch.float())
            input = input.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            if in_parallel:
                scores, loss = model_p.forward_custom(
                    input, y=target, loss=criterion, prev_tasks=prev_tasks)
            else:
                scores, loss = model.forward_custom(
                    input, y=target, loss=criterion, prev_tasks=prev_tasks)
            optimizer.step()
            updates += 1
            if limit_updates > 0 and limit_updates <= updates:
                break
            if (iterations + 1) % opt.display_freq == 0:
                print('Task[%s]   Epoch [%d/%d], Iter [%f/%f] Loss: %.4f'
                      % (task.name, epoch + 1, num_epochs, iterations + 1, len(data_loader), loss.data[0]))

                if task.type == 'segment':
                    label = target.cpu().data[0].int().numpy().astype(np.uint8)
                    predicted = scores[0].cpu().data.numpy().argmax(
                        0).astype(np.uint8)
                    label = dataset.label2image(label)
                    predicted = dataset.label2image(predicted)
                    visuals = OrderedDict([('Image', tensor2im(img_batch)),
                                           ('Truth', label),
                                           ('Predicted', predicted)
                                           ])
                    if web_visualize and visualizer:
                        visualizer.display_current_results(visuals, i)
            iterations += 1
        if epoch % opt.save_freq == 0:
            try:
                # Avoiding serialization bug here.
                m = model.mode
                model.mode = None
                torch.save(model, opt.checkpoints_dir + "/" + opt.name + "/" +
                           model.__class__.__name__ + task.name + "_" + str(epoch) + ".pth")
                model.mode = m
            except:
                print "Error Occured"
        if opt.validation_freq != 0 and epoch % opt.validation_freq == 0:
            if opt.validation_setting == 1:  # if validation setting==1 indicates validate on only the current one , else validate on all
                validation_losses = validate(model, task.test_data_loader, plot_dir=task.plot_dir,
                                             validation_losses=validation_losses, task=task, samples=10)
            else:
                tasks = [model.tasks[t] for t in model.tasks]
                validation_losses = test(
                    tasks, model, opt, plot_dir=task.plot_dir, validation_losses=validation_losses)
                model.set_mode(task)
                model = model.cuda()

        if limit_updates > 0 and limit_updates <= updates:
            break

    # Avoiding the serialization bug here
    m = model.mode
    model.mode = None
    torch.save(model, opt.checkpoints_dir + "/" + opt.name + "/" +
               model.__class__.__name__ + task.name + "_final.pth")
    model.mode = m

    model = model.cpu()
    return model
