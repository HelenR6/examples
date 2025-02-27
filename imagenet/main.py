#!/usr/bin/env python3
import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from multiprocessing import set_start_method
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
#import torchvision.models as models
import sys
sys.path.insert(1,'/content/gdrive/MyDrive/Resnet/vision/torchvision')
import models
from batch_transforms import batch_transforms
#import batch_transforms

 # sys.path.append('/vision/torchvision/models')
 #import models as models
 #import models
 #import vision.torchvision as torchvision
 # import os
 # os.chdir('/scratch/helenr6/vision/torchvision')
 # print("Current working directory: {0}".format(os.getcwd()))
 #import models
 #from helenr6.vision.torchvision import models
 #from vision.torchvision import models
 #import torchvision.models as models 
 #import vision.torchvision.models as models
from torch.utils.data import Dataset 
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                        save_as_images, convert_to_images,display_in_terminal)
import logging
import torch.nn.functional as F
import urllib
from PIL import Image
from torchvision import transforms
from numpy import linalg as LA
import numpy as np
from scipy.stats import truncnorm
from torch.autograd import Variable
import imageio
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

#resnet=models.resnet50()

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                     help='number of data loading workers (default: 4)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
# parser.add_argument('--epochs', default=90, type=int, metavar='N',
#                     help='number of total epochs to run')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=256, type=int,
#                     metavar='N',
#                     help='mini-batch size (default: 256), this is the total '
#                          'batch size of all GPUs on the current node when '
#                          'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-t','--temp', default=4, type=int, metavar='T',
                    help='temperature of distillation')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distcributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
best_acc1 = 0

def main():
    args = parser.parse_args()
    print("Current working directory: {0}".format(os.getcwd()))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        print(ngpus_per_node)
        main_worker(args.gpu, ngpus_per_node, args)
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        #torch.cuda.set_device(args.gpu)
        torch.cuda.set_device(args.gpu)
        # model = model.cuda(args.gpu)
        model = model.cuda()
    else:
         # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            print("dataparallel enabled")
            model = torch.nn.DataParallel(model).cuda()
            GAN_model = BigGAN.from_pretrained('biggan-deep-256')
            GAN_model = torch.nn.DataParallel(GAN_model).cuda()
            # batch_transforms=torch.nn.DataParallel(batch_transforms).cuda()
            #b_transforms=torch.nn.DataParallel(batch_transforms).cuda()
    # define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    def criterion(z,target_class_vector,z_hat,class_hat,T):
        loss_fn_class=nn.KLDivLoss(size_average=False)(F.log_softmax(class_hat/T,dim=1),F.softmax(class_vector/T,dim=1))* (T * T) 
        #loss_fn_class=nn.KLDivLoss(size_average=False)(F.log_softmax(class_hat/T,dim=1),target_class_vector)* (T * T) 
        loss_fn_z=nn.MSELoss()(z,z_hat)
        total_loss=loss_fn_class+loss_fn_z
        return total_loss,loss_fn_class,loss_fn_z
        #return torch.norm(class_vector-class_hat) +torch.norm(z-z_hat)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                
    #                             weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True
    # # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)
    truncation=0.5
    image_list=[]
    z_np = truncated_normal((args.batch_size, 128), low=-2, high=2)
    z=Variable(torch.from_numpy(z_np), requires_grad=True).type('torch.FloatTensor').cuda(args.gpu).detach()
    input_one_hot=np.eye(1000)[np.random.choice(1000,args.batch_size)]
    noise = np.random.normal(0, 0.01, input_one_hot.shape)
    input_np= input_one_hot+noise
    input_tensor=torch.from_numpy(input_np)
    class_vector=Variable(input_tensor, requires_grad=True).type('torch.FloatTensor').cuda(args.gpu).detach()
    image_list=GAN_model(z,class_vector,truncation)
    #image_list=GAN_model(z,class_vector_target,truncation)
    image_list=image_list.detach()
    val_loader = GANDataset(z,image_list,class_vector)
    
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        image_list=[]
        m = nn.Softmax(dim=1)
        z_np = truncated_normal((args.batch_size, 128), low=-2, high=2)
        z=Variable(torch.from_numpy(z_np), requires_grad=True).type('torch.FloatTensor').cuda(args.gpu).detach()
        input_one_hot=np.eye(1000)[np.random.choice(1000,args.batch_size)]
        noise = np.random.normal(0, 0.01, input_one_hot.shape)
        input_np= input_one_hot+noise
        input_tensor=torch.from_numpy(input_np)
        class_vector=Variable(input_tensor, requires_grad=True).type('torch.FloatTensor').cuda(args.gpu).detach()
        image_list=GAN_model(z,class_vector,truncation)
        #image_list=GAN_model(z,class_vector_target,truncation)
        image_list=image_list.detach()
        train_dataset = GANDataset(z,image_list,class_vector)
        train(train_dataset, model, criterion, optimizer, epoch, args)
        # # evaluate on validation set
        accuracy = validate(val_loader, model, criterion, args)
        writer.add_scalar('accuracy',accuracy,epoch)
        
        # # remember best acc@1 and save checkpoint
        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)
        is_best=True
        best_acc1=0
        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #         and args.rank % ngpus_per_node == 0):
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer' : optimizer.state_dict(),
        #         'T': args.temp(),
        #     }, is_best)
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    loss_list=[]
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()
    end = time.time()
    normalize=batch_transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    transform_batch = transforms.Compose([
        batch_transforms.RandomCrop(224),
        batch_transforms.RandomHorizontalFlip(),
        batch_transforms.ToTensor(),
        normalize
        ]
        )
    #for i, (images, target) in enumerate(train_loader):
    for i, (target_z,target_class_vector,images) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target_z = target_z.cuda(args.gpu, non_blocking=True).detach()
            target_class_vector = target_class_vector.cuda(args.gpu, non_blocking=True).detach()
        # compute output
        #output = model(images)
        #images=images.unsqueeze(0)
        input_to_resnet=transform_batch(images.unsqueeze(0).cuda(args.gpu, non_blocking=True))
        #input_to_resnet=F.interpolate(images,(224,224))
        z,class_vector= model(input_to_resnet)
        if i==0 and epoch==0:
            first_target_class_vector=target_class_vector.tolist()
            m = nn.Softmax(dim=1)
            #class_vector_temp=m(class_vector)
            # first_class_vector=class_vector_temp.squeeze().tolist()
            first_class_vector=class_vector.squeeze().tolist()
            fig = plt.figure()
            difference = []
            plt.plot(first_target_class_vector,"g")
            plt.plot(first_class_vector,"r")
            fig.savefig('difference.png')
        
        T=args.temp
        target_z_squeeze=target_z.unsqueeze(0)
        target_class_vector_squeeze=target_class_vector.unsqueeze(0)
        loss,loss_fn_class,loss_fn_z = criterion(target_z_squeeze,target_class_vector_squeeze,z,class_vector,T)
        # # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        #acc1, acc5 = accuracy(class_vector, target_class_vector_squeeze, topk=(1, 5))
        
        writer.add_scalar('total_loss',loss,i)
        writer.add_scalar('loss_fn_class',loss_fn_class,i)
        writer.add_scalar('loss_fn_z',loss_fn_z,i)
        
        
        
        losses.update(loss.item(), images.size(0))
        # top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))
        loss_list.append(loss.item())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # if epoch%50==0:
        #     for n, p in model.named_parameters():
        #         writer.add_histogram(f'grads/{n}', p.grad.data, i)
        #         writer.add_histogram(f'weights/{n}', p.data, i)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
    #plt.plot(loss_list)
    print(loss_list)

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # switch to evaluate mode
    model.eval()
    accuracy_value=0
    with torch.no_grad():
        end = time.time()
        normalize=batch_transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transform_batch = transforms.Compose([
          batch_transforms.RandomCrop(224),
          batch_transforms.RandomHorizontalFlip(),
          batch_transforms.ToTensor(),
          normalize
        ]
        )
        #for i, (images, target) in enumerate(val_loader):
        correct=0
        for i, (target_z,target_class_vector,images) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target_z = target_z.cuda(args.gpu, non_blocking=True).detach()
                target_class_vector = target_class_vector.cuda(args.gpu, non_blocking=True).detach()
            val_input_to_resnet=transform_batch(images.unsqueeze(0).cuda(args.gpu, non_blocking=True))
            # compute output
            output_z,output_class = model(val_input_to_resnet)
            T=args.temp
            target_z_squeeze=target_z.unsqueeze(0)
            target_class_vector_squeeze=target_class_vector.unsqueeze(0)
            loss,loss_fn_class,loss_fn_z = criterion(target_z_squeeze,target_class_vector_squeeze,output_z,output_class,T)
            
            # print out the output class annd target class
            print(f"output_class: {torch.argmax(output_class)}")
            print(f"target_class: {torch.argmax(target_class_vector_squeeze)}")
            # check if the output class match up with target class
            if(torch.argmax(output_class)==torch.argmax(target_class_vector_squeeze)):
              print("correct!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
              correct=correct+1
            # record the loss
            losses.update(loss.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        # measure accuracy 
        accuracy_value=correct/(args.batch_size)
        print(' * accuracy_values {accuracy_value:.3f} '
              .format(accuracy_value=accuracy_value))
    #return top1.avg
    return accuracy_value
# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')
def truncated_normal(size, low=0, high=1):
    values = truncnorm.rvs(low, high, size=size).astype(np.float32)
    return values
class GANDataset(Dataset):
    def __init__(self,z_list,image_list,class_list):
        self.z_list=z_list
        self.image_list=image_list
        self.class_list=class_list
    def __len__(self):
        return len(self.z_list)
    def __getitem__(self,index):
        z=self.z_list[index]
        class_vector=self.class_list[index]
        image=self.image_list[index]
        return(z,class_vector,image)
        #return(z,image)
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res

if __name__ == '__main__':
    set_start_method('spawn')
    main()
