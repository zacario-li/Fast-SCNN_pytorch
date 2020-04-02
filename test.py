'''
fast scnn test script

author: zacario li
date: 2020-03-27
'''


import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
from models.fastscnn import FastSCNN
import os

WEIGHTS_PATH = 'save/train_1999.pth'
MDL_CLS = 4

class FSCNNSegModel():
    def __init__(self, classes, weightpath):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.classes = classes
        # transform
        self.imgTrans = transforms.Compose(
            [
                #transforms.Resize([473, 473]),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]
        )
        self.hw = [540, 640]
        self.model = FastSCNN(self.classes)
        self.model.cuda()
        self.state_dict = torch.load(weightpath)
        self.model.load_state_dict(self.state_dict['state_dict'], strict=False)
        self.model.eval()

    def warmup(self):
        x = torch.rand(1,3,self.hw[0],self.hw[1])
        self.model(x.cuda())
    
    def colorArray(self):
        colStringArray =np.array([[0,0,0],
                                [128,0,0],
                                [0,128,0],
                                [128,128,0],
                                [0,0,128],
                                [128,0,128],
                                [0,128,128],
                                [128,128,128],
                                [64,0,0],
                                [192,0,0],
                                [64,128,0],
                                [192,128,0],
                                [64,0,128],
                                [192,0,128],
                                [64,128,128],
                                [192,128,128],
                                [0,64,0],
                                [128,64,0],
                                [0,192,0],
                                [128,192,0],
                                [0,64,128]],dtype='uint8')
        return colStringArray

    def colorize(self, gray, palette):
        color = Image.fromarray(gray.astype(np.uint8)).convert('P')
        color.putpalette(palette)
        return color

    def imgProcess(self, imgs):
        # imgs 为list
        #尽量传入多张
        pImgs = []
        start = time.time()
        for i in range(len(imgs)):
            img = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.imgTrans(img)
            img = img.unsqueeze(0)
            pImgs.append(img)
        # cat
        firsttensor = pImgs[0]
        for i in range(len(pImgs)-1):
            firsttensor = torch.cat((firsttensor,pImgs[i+1]),0)
        bimgs = firsttensor.cuda()
        print('imgProcess done:', time.time() - start)
        return bimgs

    def detect(self, batchimg):
        start = time.time()
        with torch.no_grad():
            output = self.model(batchimg)
        output = F.softmax(output[0], dim=1)
        output = output.data.cpu().numpy()
        output = output.transpose(0,2,3,1)
        p = np.argmax(output, axis=3)
        print('detect done:',time.time() - start)
        return p

def mergeResult(seg, cpuTensors, idx):
    empty = np.zeros((1080,1920), dtype='uint8')
    empty[:seg.hw[0],:seg.hw[1]] = np.uint8(cpuTensors[0])
    empty[:seg.hw[0],seg.hw[1]:seg.hw[1]*2] = np.uint8(cpuTensors[1])
    empty[:seg.hw[0],seg.hw[1]*2:] = np.uint8(cpuTensors[2])
    empty[seg.hw[0]:,:seg.hw[1]] = np.uint8(cpuTensors[3])
    empty[seg.hw[0]:,seg.hw[1]:seg.hw[1]*2] = np.uint8(cpuTensors[4])
    empty[seg.hw[0]:,seg.hw[1]*2:] = np.uint8(cpuTensors[5])

    color = seg.colorArray()
    gray = seg.colorize(empty, color)
    gray.save('./result/result_'+str(idx)+'.png')

def processVideo(filename):
    seg = FSCNNSegModel(MDL_CLS, WEIGHTS_PATH)
    seg.warmup()
    cap = cv2.VideoCapture(filename)
    idx = 0
    while True:
        print('processing...'+str(idx))
        imglist = []
        ret, frame = cap.read()
        if ret is False:
            break
        # 左上
        imglist.append(frame[:seg.hw[0],:seg.hw[1],:])
        # 中上
        imglist.append(frame[:seg.hw[0],seg.hw[1]:seg.hw[1]*2,:])
        # 右上
        imglist.append(frame[:seg.hw[0],seg.hw[1]*2:,:])
        # 左下
        imglist.append(frame[seg.hw[0]:,:seg.hw[1],:])
        # 中下
        imglist.append(frame[seg.hw[0]:,seg.hw[1]:seg.hw[1]*2,:])
        # 右下
        imglist.append(frame[seg.hw[0]:,seg.hw[1]*2:,:])

        # start detect
        bimgs = seg.imgProcess(imglist)
        result = seg.detect(bimgs)
        mergeResult(seg, result, str(idx).zfill(6))
        idx += 1


if __name__ == '__main__':
    processVideo('test.mp4')