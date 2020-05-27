'''
fast scnn

author: zacario li
date: 2020-03-27
'''
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSCNN(nn.Module):
    def __init__(self, numClasses, aux=False, **kwargs):
        super(FastSCNN, self).__init__()
        # auxiliary, use to accelarate the convergence
        self.aux = aux
       
        # learning to down-sample (ph1)
        self.learningToDownSample = LearningToDownSample(32, 48, 64)
        # global feature extractor (ph2)
        self.globalFeatureExtractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3,3,3])
        # feature fusion (ph3)
        self.featureFusion = FeatureFusion(64,128, 128)
        # classifier (ph4)
        self.classifier = Classifier(128, numClasses)
        # for training only use
        if self.aux is not None:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, numClasses, 1)
            )

    def forward(self, x):
        inputSize = x.shape[2:]
        out = []
        # ph1
        ph1 = self.learningToDownSample(x)
        # ph2
        x = self.globalFeatureExtractor(ph1)
        # ph3
        x = self.featureFusion(ph1,x)
        # ph4
        x = self.classifier(x)
        # resize to input img size
        x = F.interpolate(x, inputSize, mode='bilinear', align_corners=True)
        out.append(x)
        # when training, use auxiliary
        if self.aux:
            auxout = self.auxlayer(ph1)
            auxout = F.interpolate(auxout, inputSize, mode='bilinear', align_corners=True)
            out.append(auxout)
       
        return out


'''
common used module in paper
Red: Conv2D
Gray: DWConv
Blue: DSConv
Green: Bottleneck
Pink: Pyramid Pooling
Yellow: Upsample
'''
class _Conv2D(nn.Module):
    '''
    Red
    '''
    def __init__(self, inChannels, outChannels, kernel, stride=1, padding=0, **kwargs):
        super(_Conv2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(True)
        )
   
    def forward(self, x):
        x = self.conv(x)
        return x

class _DSConv(nn.Module):
    '''
    Blue
    '''
    def __init__(self, inChannels, outChannels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inChannels,inChannels, 3, stride, 1, groups=inChannels, bias=False),
            nn.BatchNorm2d(inChannels),
            nn.ReLU(True),
            nn.Conv2d(inChannels,outChannels,1,bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class _DWConv(nn.Module):
    '''
    Gray
    '''
    def __init__(self, inChannels, outChannels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, 3, stride, 1, groups=inChannels,bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class _Bottleneck(nn.Module):
    '''
    Green:
    Bottleneck
    '''
    def __init__(self, inChannels, outChannels, t=6, stride=2, **kwargs):
        super(_Bottleneck, self).__init__()
        self.shortcut = stride == 1 and inChannels == outChannels
        self.block = nn.Sequential(
            _Conv2D(inChannels, inChannels*t, 1),
            _DWConv(inChannels*t, inChannels*t, stride),
            #the last pointwise conv does not use non-linearity f. described in Table 2. Page 4
            nn.Conv2d(inChannels*t, outChannels, 1,bias=False),
            nn.BatchNorm2d(outChannels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.shortcut:
            out = x + out
        return out


class _PPM(nn.Module):
    '''
    Pink
    '''
    def __init__(self,inChannels, outChannels, **kwargs):
        super(_PPM, self).__init__()
        # described in PSPNet paper(https://arxiv.org/pdf/1612.01105.pdf), 3.2, page 3
        tempChannel = int(inChannels/4)
        self.p1 = _Conv2D(inChannels, tempChannel, 1)
        self.p2 = _Conv2D(inChannels, tempChannel, 1)
        self.p3 = _Conv2D(inChannels, tempChannel, 1)
        self.p4 = _Conv2D(inChannels, tempChannel, 1)
        # why need conv2d here? There isn't any words about it in the paper
        self.cat = _Conv2D(inChannels*2, outChannels, 1)
        
    def featurePooling(self, x, size):
        avgp = nn.AdaptiveAvgPool2d(size)
        x = avgp(x)
        return x
    
    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        size = x.shape[2:]
        f1 = self.upsample(self.p1(self.featurePooling(x,1)),size)
        f2 = self.upsample(self.p2(self.featurePooling(x,2)),size)
        f3 = self.upsample(self.p3(self.featurePooling(x,3)),size)
        f4 = self.upsample(self.p4(self.featurePooling(x,6)),size)
        x = torch.cat([x, f1, f2, f3, f4],dim=1)
        x = self.cat(x)
        return x
# ph1
class LearningToDownSample(nn.Module):
    '''
    ph1 has two dsconv, so wo need input these parameters
    '''
    def __init__(self, dsc1, dsc2, dsc2out, **kwargs):
        super(LearningToDownSample, self).__init__()
        # described in paper, Table 1, page 4
        self.conv = _Conv2D(3,dsc1, 3, 2)
        self.dsc1 = _DSConv(dsc1,dsc2,2)
        self.dsc2 = _DSConv(dsc2,dsc2out,2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsc1(x)
        x = self.dsc2(x)
        return x

# ph2
class GlobalFeatureExtractor(nn.Module):
    '''
    ph2
    '''
    def __init__(self, inChannels=64, btChannels=[64,96,128], 
                 outChannels=128, t=6, numBt=[3,3,3], **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        # described in paper, Figure 1, page 2, we have 3 different shape bottlenecks
        self.bt1 = self._make_btlayer(_Bottleneck, inChannels, btChannels[0],numBt[0],t,2)
        self.bt2 = self._make_btlayer(_Bottleneck, btChannels[0], btChannels[1],numBt[1],t,2)
        self.bt3 = self._make_btlayer(_Bottleneck, btChannels[1], btChannels[2],numBt[2],t,1)
        self.ppm = _PPM(btChannels[2],outChannels)

    def _make_btlayer(self, bt, inChannels, outChannels, numBlock, t=6, stride=1):
        layers = []
        layers.append(bt(inChannels, outChannels, t, stride))
        for i in range(1, numBlock):
            layers.append(bt(outChannels, outChannels, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bt1(x)
        x = self.bt2(x)
        x = self.bt3(x)
        x = self.ppm(x)
        return x


# ph3
class FeatureFusion(nn.Module):
    def __init__(self, ph1InChannel, ph2InChannel, outChannels, scale=4, **kwargs):
        super(FeatureFusion, self).__init__()
        self.scale = scale
        self.dwconv = _DWConv(ph2InChannel,outChannels,1)
        self.upBranch = nn.Sequential(nn.Conv2d(outChannels, outChannels, 1),
                                      nn.BatchNorm2d(outChannels))
        self.downBranch = nn.Sequential(nn.Conv2d(ph1InChannel, outChannels, 1),
                                        nn.BatchNorm2d(outChannels))
        self.activation = nn.ReLU(True)
    

    def forward(self, ph1Feature, ph2Feature):
        xUp = F.interpolate(ph2Feature, size=ph1Feature.shape[2:], mode='bilinear', align_corners=True)
        xUp = self.dwconv(xUp)
        xUp = self.upBranch(xUp)
        
        xDown = self.downBranch(ph1Feature)
        
        out = xUp + xDown
        out = self.activation(out)
        return out
    
# ph4
class Classifier(nn.Module):
    '''
    without upsample and softmax
    '''
    def __init__(self, inChannels, numClasses, stride=1):
        super(Classifier, self).__init__()
        # described in 3.2.4 Classifier, page 5
        self.dsconv1 = _DSConv(inChannels, inChannels, stride)
        self.dsconv2 = _DSConv(inChannels, inChannels, stride)
        self.conv = nn.Conv2d(inChannels, numClasses, 1)
        
    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x
    
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ntimes = 100
    model = FastSCNN(4)
    model.cuda()
    model.eval()
    with torch.no_grad():
        x = torch.randn(1,3,320,320)
        x = x.cuda()
        # warmup
        out = model(x)
        start = time.time()
        for i in range(ntimes):
            model(x)
        print('fps is :', 1.0/((time.time() - start)/ntimes))