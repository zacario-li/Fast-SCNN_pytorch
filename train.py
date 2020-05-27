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

numClasses = 21
dataRoot = 'voc2012'
trainList = 'voc2012/train.txt'
valList = 'voc2012/val.txt'
globalEpoch = 2000
baseLr = 0.01
inputHW = [320, 320]

cv2.ocl.setUseOpenCL(True)
cv2.setNumThreads(32)


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

def prepareDataset(rootpath, trainlist, vallist, mean, std):
    # prepare dataset transform before training
    trans = transform.Compose([
        transform.RandScale([0.5,2]),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop(inputHW,crop_type='rand',padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean,std=std)
    ])
    # val transform
    valTrans = transform.Compose([
        transform.Crop(inputHW, crop_type='center', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    # training data
    trainData = dataset.SemData(split='train', data_root=rootpath, data_list=trainlist, transform=trans)
    trainDataLoader = torch.utils.data.DataLoader(trainData,
                                                batch_size=160,
                                                shuffle=True,
                                                num_workers=32,
                                                pin_memory=True,
                                                drop_last=True)

    # val data
    valData = dataset.SemData(split='val', data_root=rootpath, data_list=vallist, transform=valTrans)
    valDataLoader = torch.utils.data.DataLoader(valData,
                                                batch_size=4,
                                                shuffle=False,
                                                num_workers=4,
                                                pin_memory=True)
    # return datasets
    return trainDataLoader, valDataLoader

def subTrain(model, optimizer, criterion, dataLoader, currentepoch, maxIter, device):
    # set to train mode to enable dropout and bn
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

        # whole loss
        loss = 0.4*auxLoss + mainLoss
        lossMeter.update(loss.item(), x.shape[0])
        #print('loss is:', loss.item())

        # step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ajust lr
        curIter = currentepoch * len(dataLoader) + i + 1
        newLr = poly_learning_rate(baseLr, curIter, maxIter)
        optimizer.param_groups[0]['lr'] = newLr

        # compute IoU/accuracy
        result = out[0].max(1)[1]
        intersection, union, target = common.intersectionAndUnionGPU(result, y, numClasses, 255)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        ## update meter
        intersectionMeter.update(intersection), unionMeter.update(union), targetMeter.update(target)
    
    # after every epoch, print the log    
    IoU = intersectionMeter.sum/(unionMeter.sum + 1e-10)
    accuracy = intersectionMeter.sum/(targetMeter.sum + 1e-10)
    
    print(f'[{currentepoch}/{globalEpoch}] loss:',lossMeter.avg)
    '''
    for i in range(numClasses):
        print(f'training Class_{i} IoU: {IoU[i]}, acc: {accuracy[i]}')
    '''

def subVal(model, criterion, dataLoader, device):
    # set to eval mode
    model.eval()

    intersectionMeter = common.AverageMeter()
    unionMeter = common.AverageMeter()
    targetMeter = common.AverageMeter()
    lossMeter = common.AverageMeter()

    for i, (x, y) in enumerate(dataLoader):
        x = x.to(device)
        y = y.to(device)
        
        out = model(x)
        mainLoss = criterion(out[0], y)
        
        # update val loss
        lossMeter.update(mainLoss.item(), x.shape[0])

        # compute IoU/accuracy
        result = out[0].max(1)[1]
        intersection, union, target = common.intersectionAndUnionGPU(result, y, numClasses, 255)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        ## update meter
        intersectionMeter.update(intersection), unionMeter.update(union), targetMeter.update(target)
    
    # show the log
    IoU = intersectionMeter.sum/(unionMeter.sum + 1e-10)
    accuracy = intersectionMeter.sum/(targetMeter.sum + 1e-10)
    print(f'val loss:',lossMeter.avg)
    for i in range(numClasses):
        print('Class_'+str(i)+' IoU:',IoU[i],' acc:',accuracy[i])

def train():
    #device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = fastscnn.FastSCNN(numClasses, True)
    numParams = sum(torch.numel(p) for p in model.parameters() )
    print(f'Total paramers: {numParams}')
    model = model.to(device)
    weightsInit(model)
    mean,std = getMeanStd()

    #criterion = nn.CrossEntropyLoss(ignore_index=255)
    criterion = diceloss.DiceLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=baseLr, momentum=0.9, weight_decay=0.0001)

    # get dataset
    trainDataLoader, valDataLoader = prepareDataset(dataRoot, trainList, valList, mean, std)

    # prepare something for learning rate
    maxIter = globalEpoch * len(trainDataLoader)

    # start training
    for epoch in range(1, globalEpoch):
        # do train on every epoch
        subTrain(model, optimizer, criterion, trainDataLoader, epoch, maxIter, device)
        # evaluate
        subVal(model, criterion, valDataLoader, device)        
        # save model
        if ( (epoch) % 20) == 0:
            filename = 'save/'+'train_'+str(epoch)+'.pth'
            torch.save({'epoch':epoch, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}, filename)


if __name__ == '__main__':
    train()
