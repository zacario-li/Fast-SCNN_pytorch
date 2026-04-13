'''
train scripts

author: zacario li
date: 2020-04-02
'''


import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.init as initer

from tensorboardX import SummaryWriter
import sys
from utils import dataset, transform, common
from models import fastscnn
from loss import diceloss


cv2.ocl.setUseOpenCL(True)
cv2.setNumThreads(32)


def get_args():
    parser = argparse.ArgumentParser(description='Fast-SCNN Training')
    parser.add_argument('--data-root', type=str, default='voc2012',
                        help='path to dataset root directory')
    parser.add_argument('--train-list', type=str, default=None,
                        help='path to train list file (default: <data-root>/train.txt)')
    parser.add_argument('--val-list', type=str, default=None,
                        help='path to val list file (default: <data-root>/val.txt)')
    parser.add_argument('--num-classes', type=int, default=21,
                        help='number of segmentation classes (default: 21 for VOC2012)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='total training epochs')
    parser.add_argument('--base-lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=160,
                        help='training batch size')
    parser.add_argument('--val-batch-size', type=int, default=4,
                        help='validation batch size')
    parser.add_argument('--input-h', type=int, default=320,
                        help='input image height')
    parser.add_argument('--input-w', type=int, default=320,
                        help='input image width')
    parser.add_argument('--num-workers', type=int, default=32,
                        help='number of data loading workers')
    parser.add_argument('--save-dir', type=str, default='save',
                        help='directory for saving checkpoints')
    parser.add_argument('--save-freq', type=int, default=20,
                        help='save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume training from')
    parser.add_argument('--log-dir', type=str, default='runs',
                        help='TensorBoard log directory')
    parser.add_argument('--multigpu', action='store_true',
                        help='use nn.DataParallel for multi-GPU training')
    parser.add_argument('--dist', action='store_true',
                        help='use DistributedDataParallel for multi-GPU training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training (set by torch.distributed.launch)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='SGD weight decay')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='weight for auxiliary loss')
    args = parser.parse_args()

    if args.train_list is None:
        args.train_list = os.path.join(args.data_root, 'train.txt')
    if args.val_list is None:
        args.val_list = os.path.join(args.data_root, 'val.txt')

    return args


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr

def weightsInit(model):
    for m in model.modules():
        if isinstance(m, (nn.modules.conv._ConvNd)):
            initer.kaiming_normal_(m.weight)
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, (nn.modules.batchnorm._BatchNorm)):
            initer.normal_(m.weight, 1.0, 0.02)
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            initer.kaiming_normal_(m.weight)
            if m.bias is not None:
                initer.constant_(m.bias, 0)

def getMeanStd():
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    return mean, std

def prepareDataset(args, mean, std):
    inputHW = [args.input_h, args.input_w]
    trans = transform.Compose([
        transform.RandScale([0.5,2]),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop(inputHW,crop_type='rand',padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean,std=std)
    ])
    valTrans = transform.Compose([
        transform.Crop(inputHW, crop_type='center', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    trainData = dataset.SemData(split='train', data_root=args.data_root,
                                data_list=args.train_list, transform=trans)
    train_sampler = None
    if args.dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainData)

    trainDataLoader = torch.utils.data.DataLoader(
        trainData,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    valData = dataset.SemData(split='val', data_root=args.data_root,
                              data_list=args.val_list, transform=valTrans)
    valDataLoader = torch.utils.data.DataLoader(
        valData,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    return trainDataLoader, valDataLoader, train_sampler

def subTrain(model, optimizer, criterion, dataLoader, currentepoch, maxIter, device, args):
    model.train()

    intersectionMeter = common.AverageMeter()
    unionMeter = common.AverageMeter()
    targetMeter = common.AverageMeter()
    lossMeter = common.AverageMeter()

    for i, (x, y) in enumerate(dataLoader):
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        mainLoss = criterion(out[0], y)
        auxLoss = criterion(out[1], y)

        loss = args.aux_weight * auxLoss + mainLoss
        lossMeter.update(loss.item(), x.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        curIter = currentepoch * len(dataLoader) + i + 1
        newLr = poly_learning_rate(args.base_lr, curIter, maxIter)
        optimizer.param_groups[0]['lr'] = newLr

        result = out[0].max(1)[1]
        intersection, union, target = common.intersectionAndUnion(result, y, args.num_classes, 255)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersectionMeter.update(intersection), unionMeter.update(union), targetMeter.update(target)

    IoU = intersectionMeter.sum/(unionMeter.sum + 1e-10)
    mIoU = IoU.mean()
    accuracy = intersectionMeter.sum/(targetMeter.sum + 1e-10)
    mAcc = accuracy.mean()

    print(f'[{currentepoch}/{args.epochs}] loss:',lossMeter.avg)

    return lossMeter.avg, mIoU, mAcc

def subVal(model, criterion, dataLoader, device, args):
    model.eval()

    intersectionMeter = common.AverageMeter()
    unionMeter = common.AverageMeter()
    targetMeter = common.AverageMeter()
    lossMeter = common.AverageMeter()

    with torch.no_grad():
        for i, (x, y) in enumerate(dataLoader):
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            mainLoss = criterion(out[0], y)

            lossMeter.update(mainLoss.item(), x.shape[0])

            result = out[0].max(1)[1]
            intersection, union, target = common.intersectionAndUnion(result, y, args.num_classes, 255)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersectionMeter.update(intersection), unionMeter.update(union), targetMeter.update(target)

    IoU = intersectionMeter.sum/(unionMeter.sum + 1e-10)
    mIoU = IoU.mean()
    accuracy = intersectionMeter.sum/(targetMeter.sum + 1e-10)
    mAcc = accuracy.mean()

    print(f'val loss:',lossMeter.avg)
    for i in range(args.num_classes):
        print('Class_'+str(i)+' IoU:',IoU[i],' acc:',accuracy[i])

    return lossMeter.avg, mIoU, mAcc

def train():
    args = get_args()

    if args.dist:
        dist.init_process_group(backend='nccl')
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = fastscnn.FastSCNN(args.num_classes, True)
    numParams = sum(torch.numel(p) for p in model.parameters())
    print(f'Total paramers: {numParams}')
    model = model.to(device)

    start_epoch = 1
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print(f'=> loading checkpoint: {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f'=> resumed from epoch {checkpoint["epoch"]}')
        else:
            print(f'=> no checkpoint found at: {args.resume}')
            weightsInit(model)
    else:
        weightsInit(model)

    if args.dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    elif args.multigpu and torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs with DataParallel')
        model = torch.nn.DataParallel(model)

    criterion = diceloss.DiceLoss()
    mean, std = getMeanStd()
    trainDataLoader, valDataLoader, train_sampler = prepareDataset(args, mean, std)

    maxIter = args.epochs * len(trainDataLoader)

    is_main = (not args.dist) or (args.local_rank == 0)
    writer = None
    if is_main:
        writer = SummaryWriter(log_dir=args.log_dir)
        os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        if args.dist and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_mIoU, train_mAcc = subTrain(
            model, optimizer, criterion, trainDataLoader, epoch, maxIter, device, args)

        val_loss, val_mIoU, val_mAcc = subVal(
            model, criterion, valDataLoader, device, args)

        if writer is not None:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/mIoU', train_mIoU, epoch)
            writer.add_scalar('train/mAcc', train_mAcc, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/mIoU', val_mIoU, epoch)
            writer.add_scalar('val/mAcc', val_mAcc, epoch)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        if is_main and (epoch % args.save_freq) == 0:
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            filename = os.path.join(args.save_dir, f'train_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                'args': vars(args)
            }, filename)

    if writer is not None:
        writer.close()

    if args.dist:
        dist.destroy_process_group()


if __name__ == '__main__':
    train()
