import argparse
import os
import util

class BaseOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            '--batchSize', type=int, default=4, help='input batch size')
        self.parser.add_argument('--loadSize',type=int, default=500, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=960, help='then crop to this size')
        self.parser.add_argument('--dataroot', type=str, default='/home/sloke/repos/lifelong-learning/lifelong-learning/dataset',
                                 help='path to images folder')
        self.parser.add_argument(
            '--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2')
        self.parser.add_argument('--name', type=str, default='testBeta1',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--model', type=str, default='FCN8s',
                                 help='chooses which model to use.')
        self.parser.add_argument(
            '--nThreads', default=20, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str,
                                 default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_false',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')

        self.parser.add_argument('--display_freq', type=int, default=1,
                                 help='frequency of displaying loss in an epoch')
        self.parser.add_argument('--lr', type=float, default=1.0e-10,
                                 help='Learning Rate')
        self.parser.add_argument('--betas1', type=float, default=0.9,
                                 help='Beta for ADAM')

        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--visualize', type=int, default=1, help='Visualize Results')
        self.parser.add_argument('--web_visualize', type=int, default=1, help='Web Visualize Results')
        self.parser.add_argument('--record_parameters', type=int, default=1, help='Record Parameters')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--save_freq', type=int, default=20, help='Save Frequency Epochs')
        self.parser.add_argument('--validation_freq', type=int, default=20, help='validation_freq')
        self.parser.add_argument('--validation_setting', type=int, default=1, help='Save Frequency Epochs')

        self.parser.add_argument('--num_epochs', type=int, default=25, help='Epochs')

        self.parser.add_argument('--pretrained', type=int, default=0, help='if pretrained')
        self.parser.add_argument('--load_pretrained', type=str,default='prt.th', help='Path to pretrained Model')
        self.parser.add_argument('--in_parallel', type=int,default=0, help='If parallel GPUs?')
        self.parser.add_argument('--limit_updates', type=int,default=0, help='Limit Backprops for each task')
        self.parser.add_argument('--limit_test_updates', type=int,default=0, help='Limit Backprops for each task')
        self.parser.add_argument('--custom_settings',type=str,default='settings/cityscapes_grouped.json', help='Path to custom settings')
        self.parser.add_argument('--distill_loss',type=str,default='compute_ce_loss', help='Distillation Loss settings')
        self.parser.add_argument('--model_setting',type=str,default='normal', help='Model  settings')
        self.parser.add_argument('--encoder',type=str,default='resnet50_dilated8', help='Model  settings')
        self.parser.add_argument('--decoder',type=str,default='c5bilinear', help='Model  settings')
        self.parser.add_argument('--same_task',type=int,default=0, help='Model  settings')
        self.parser.add_argument('--uncertainty_func',type=str,default='global_uncertainty_fast', help='Model  settings')
        self.parser.add_argument('--data_dir',type=str,default='/groups/jbhuang_lab/usr/sloke/data', help='Model  settings')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
