import torch
from torch.autograd import Variable, Function
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import os
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, Function
import time
import numpy as np
import matplotlib.pyplot as plt
#from MLP_sample import MLP_sample

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,padding=2)
def conv5x5(in_planes, out_planes,stride=1):
    "5x5 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,padding=2)

def conv3x3(in_planes, out_planes,stride=3):
    "7x7 convolution without padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=9, stride=stride,padding=0)

class MLP_sample(nn.Module):
  def __init__(self):
    super(MLP_sample, self).__init__()
    self.fc1 = nn.Linear(3072,512)
    #self.bn1 = nn.BatchNorm1d(512,track_running_stats=True)
    self.fc3 = nn.Linear(512,256)
    #self.bn3 = nn.BatchNorm1d(256,track_running_stats=True)
    self.fc5 = nn.Linear(256,128)
    #self.bn5 = nn.BatchNorm1d(128,track_running_stats=True)    
    self.fc7 = nn.Linear(128,64)
    #self.bn7 = nn.BatchNorm1d(64,track_running_stats=True)
    self.fc8 = nn.Linear(64,10)

  def forward(self,x):
    x = x.view(x.shape[0], -1)
    x = F.relu((self.fc1(x)))
    x = F.relu((self.fc3(x)))
    x = F.relu((self.fc5(x)))
    x = F.relu((self.fc7(x)))
    x = self.fc8(x)
    return F.log_softmax(x)

class MLP_sample_lenet(nn.Module):
  def __init__(self):
    super(MLP_sample_lenet, self).__init__()
    self.fc1 = nn.Linear(784,300)
    #self.bn1 = nn.BatchNorm1d(512,track_running_stats=True)
    self.fc3 = nn.Linear(300,100)
    #self.bn3 = nn.BatchNorm1d(256,track_running_stats=True)
    self.fc8 = nn.Linear(100,10)

  def forward(self,x):
    x = x.view(x.shape[0], -1)
    x = F.relu((self.fc1(x)))
    x = F.relu((self.fc3(x)))
    x = self.fc8(x)
    return F.log_softmax(x)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = conv3x3(3, 64)
    self.pool1 = nn.MaxPool2d(2,2)
    self.pool2 = nn.MaxPool2d(4,4)
    self.conv2 = conv5x5(64, 128)
    self.conv3 = conv5x5(128, 256)
    self.conv4 = conv5x5(256, 400)
    self.conv5 = conv5x5(400, 512)
    self.fc1 = nn.Linear(512 * 1 * 1, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    #print(x.shape)
    x = self.pool1(F.relu(self.conv1(x)))
    #print(x.shape)  
    x = F.relu(self.conv2(x))
    #print(x.shape)
    x = F.relu(self.conv3(x))
    #print(x.shape)
    x = F.relu(self.conv4(x))
    #print(x.shape)
    x = self.pool2(F.relu(self.conv5(x)))
    #print("After second pooling",x.shape)
    x = x.view(-1, 512 * 1 * 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


model = MLP_sample()

#X = np.load('X.npy')
#X = np.load('X_lenet.py')

class skiplinear(Function):
  def __init__(self,mask):
      super(skiplinear, self).__init__()
      self.mask = mask

  def forward(self, input, weight):
      self.save_for_backward(input, weight)
      extendWeights = weight.clone()
      extendWeights.mul_(self.mask.data)
      output = input.mm(extendWeights.t())
      return output

  def backward(self, grad_output):
      input, weight = self.saved_tensors
      grad_input = grad_weight  = None
      extendWeights = weight.clone()
      extendWeights.mul_(self.mask.data)

      if self.needs_input_grad[0]:
          grad_input = grad_output.mm(extendWeights)
      if self.needs_input_grad[1]:
          grad_weight = grad_output.clone().t().mm(input)
          grad_weight.mul_(self.mask.data)

      return grad_input, grad_weight

'''Introducing skip connections'''
class SkipLinear(torch.nn.Module):
  def __init__(self, input_features,output_features,matrix):
    super(SkipLinear, self).__init__()
    self.input_features = input_features
    self.output_features = output_features
    self.matrix = matrix
    self.weight = nn.Parameter(data=torch.Tensor(output_features, input_features), requires_grad=True).float()
    self.weight = nn.Parameter(self.weight.cuda(), requires_grad=True)

    '''**********************************Getting the mask**************************************'''
    self.mask = torch.zeros(output_features, input_features,1,1) 
    
    k = 0
    kk = 0
    global last_conv_weight
    last_conv_weight = 0
    for m in model.modules():
      if isinstance(m,nn.Conv2d):
        if k == 0:
          k += m.weight.shape[1]
        else:
          k += m.weight.shape[1]
          kk = k - m.weight.shape[1]
          last_conv_weight = m.weight.shape[1]
      elif isinstance(m,nn.Linear):
        if(m.weight.shape[1] != output_features):
          k += m.weight.shape[1]
        else:
          break
    kk += last_conv_weight
    for m in model.modules():
      if isinstance(m,nn.Linear):
        if(m.weight.shape[1] != input_features):
          kk += m.weight.shape[1]
        else:
          break

    '''
    ##Randomly building skip connections
    if output_features < input_features:
        for i in range(output_features):
            x = torch.randperm(input_features)
            for j in range(expandSize):
                self.mask[i][x[j]] = 1
    else:
        for i in range(input_features):
            x = torch.randperm(output_features)
            for j in range(expandSize):
                self.mask[x[j]][i] = 1

    '''
    self.mask = matrix[k:k+output_features,kk:kk+input_features].astype(float)
    self.mask = torch.from_numpy(self.mask).cuda().float()

    nn.init.kaiming_normal_(self.weight.data,mode='fan_in')
    self.mask =  nn.Parameter(self.mask.cuda())
    self.mask.requires_grad = False

  def forward(self, input):
      return skiplinear(self.mask)(input, self.weight)

'''
class MulSkip(Function):
    def __init__(self,mask):
        super(MulSkip, self).__init__()
        self.mask = mask

    def forward(self, weight):
        extendWeights = weight.clone()
        extendWeights.mul_(self.mask.data)
        #print(np.count_nonzero(self.mask.cpu().numpy())/np.count_nonzero(weight.shape[0]*weight.shape[1]))
        return extendWeights

    def backward(self, grad_output):
        grad_weight = grad_output.clone()
        grad_weight.mul_(self.mask.data)
        return grad_weight

class execute2DConvolution(torch.nn.Module):
    def __init__(self, mask, inStride=1, inPadding=0, inDilation=1, inGroups=1):
        super(execute2DConvolution, self).__init__()
        self.cStride = inStride
        self.cPad = inPadding
        self.cDil = inDilation
        self.cGrp = inGroups
        self.mask = mask

    def forward(self, dataIn, weightIn):
        fpWeights = MulSkip(self.mask)(weightIn)
        return torch.nn.functional.conv2d(dataIn, fpWeights, bias=None,
                                          stride=self.cStride, padding=self.cPad,dilation=self.cDil, groups=self.cGrp)
'''
class MulSkip(Function):
    def __init__(self,mask):
        super(MulSkip, self).__init__()
        self.mask = mask

    def forward(self, weight):
        extendWeights = weight.clone()
        #print(extendWeights[23])
        extendWeights.mul_(self.mask.data)
        #print(extendWeights[23])
        return extendWeights

    def backward(self, grad_output):
        grad_weight = grad_output.clone()
        #print(extendWeights[23])
        grad_weight.mul_(self.mask.data)
        #print(extendWeights[23])
        return grad_weight

class execute2DConvolution(torch.nn.Module):
    def __init__(self, mask, inStride=1, inPadding=0, inDilation=1, inGroups=1):
        super(execute2DConvolution, self).__init__()
        self.cStride = inStride
        self.cPad = inPadding
        self.cDil = inDilation
        self.cGrp = inGroups
        self.mask = mask

    def forward(self, dataIn, weightIn):
        fpWeights = MulSkip(self.mask)(weightIn)
        return torch.nn.functional.conv2d(dataIn, fpWeights, bias=None,
                                          stride=self.cStride, padding=self.cPad,
                                          dilation=self.cDil, groups=self.cGrp)

class SkipConv2d(torch.nn.Module):
    def __init__(self, inWCin, inWCout, matrix,kernel_size, stride=1, padding=0, inDil=1, groups=1, mode='random'):
        super(SkipConv2d, self).__init__()
        # Initialize all parameters that the convolution function needs to know
        self.kernel_size = kernel_size
        self.in_channels = inWCin
        self.out_channels = inWCout
        self.conStride = stride
        self.conPad = padding
        self.outPad = 0
        self.conDil = inDil
        self.conTrans = False
        self.conGroups = groups
        #self.matrix = matrix
      
        n = kernel_size * kernel_size * inWCout
        # initialize the weights and the bias as well as the
        self.weight = torch.nn.Parameter(data=torch.Tensor(inWCout, inWCin, kernel_size, kernel_size), requires_grad=True).float()
        nn.init.kaiming_normal_(self.weight.data,mode='fan_out')

        '''
        self.mask = torch.zeros(inWCout, (inWCin),1,1)        
        if inWCin > inWCout:
            for i in range(inWCout):
                x = torch.randperm(inWCin)
                for j in range(expandSize):
                    self.mask[i][x[j]][0][0] = 1
        else:
            for i in range(inWCin):
                x = torch.randperm(inWCout)
                for j in range(expandSize):
                    self.mask[x[j]][i][0][0] = 1

        self.mask = self.mask.repeat(1, 1, kernel_size, kernel_size)
        self.mask =  nn.Parameter(self.mask.cuda().float())
        self.mask.requires_grad = False
        '''
        
        k=0
        kk=0
        for m in model.modules():
          if isinstance(m,nn.Conv2d):
            if(m.weight.shape[1] != inWCout):
              k += m.weight.shape[1]
            else:
              break
        for m in model.modules():
          if isinstance(m,nn.Conv2d):
            if(m.weight.shape[1] != inWCin):
              kk += m.weight.shape[1]
            else:
              break
        
        self.mask = torch.from_numpy(matrix[k:k+inWCout,kk:kk+inWCin]).float()
        self.mask = torch.reshape(self.mask,(self.mask.shape[0],self.mask.shape[1],1,1))
        self.mask = self.mask.repeat(1,1,kernel_size,kernel_size)
        self.mask =  nn.Parameter(self.mask.cuda())
        self.mask.requires_grad = False
                

    def forward(self, dataInput):
        return execute2DConvolution(self.mask, self.conStride, self.conPad,self.conDil, self.conGroups)(dataInput, self.weight)
