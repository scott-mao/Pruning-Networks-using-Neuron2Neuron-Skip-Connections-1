from __future__ import print_function, division
import torch
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
from models.layers.skipconnectionlayers import NewLinear,SkipLinear
from models.layers.expandergraphlayer import ExpanderLinear,ExpanderConv2d

class MLPexpander_lenet(nn.Module):
  def __init__(self,sparsity):
    super(MLPexpander_lenet, self).__init__()
    self.fc1 = ExpanderLinear(784,300,expandSize=np.floor(0.01*sparsity*784).astype(int))
    self.fc3 = ExpanderLinear(300,100,expandSize=np.floor(0.01*sparsity*300).astype(int))
    self.fc8 = ExpanderLinear(100,10,expandSize=np.floor(0.02*sparsity*100).astype(int))

  def forward(self,x):
    x = x.view(x.shape[0], -1)
    x = F.relu((self.fc1(x)))
    x = F.relu((self.fc3(x)))
    x = self.fc8(x)
    return F.log_softmax(x,dim=1)

class MLP_lenet(nn.Module):
  def __init__(self):
    super(MLP_lenet, self).__init__()
    self.fc1 = nn.Linear(784,300)
    self.fc3 = nn.Linear(300,100)
    self.fc8 = nn.Linear(100,10)

  def forward(self,x):
    x = x.view(x.shape[0], -1)
    x = F.relu((self.fc1(x)))
    x = F.relu((self.fc3(x)))
    x = self.fc8(x)
    return F.log_softmax(x,dim=1)


class MLP_new_lenet(nn.Module):
  def __init__(self,X):
    super(MLP_new_lenet, self).__init__()
    self.fc1 = NewLinear(784,300,matrix = X)
    self.fc2 = NewLinear(300,100,matrix = X)
    self.fc3 = NewLinear(100,10,matrix = X)
  
    self.skip1 = SkipLinear(784,100,matrix = X)
    self.skip2 = SkipLinear(784,10,matrix = X)
    self.skip3 = SkipLinear(300,10,matrix = X)


  def forward(self,x):
    x = x.view(x.shape[0], -1)
    a_784 = F.relu((self.fc1(x)))
    a_300 = F.relu((self.fc2(a_784))+self.skip1(x))  
    x_100 = self.fc3(a_300)+F.relu(self.skip3(a_784))+F.relu(self.skip2(x))
    return F.log_softmax(x_100,dim=1)

