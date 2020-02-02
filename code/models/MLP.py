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
from models.layers.skipconnectionlayers import SkipLinear
from models.layers.expandergraphlayer import ExpanderLinear,ExpanderConv2d


#MLPexpander Model
class MLPexpander(nn.Module):
  def __init__(self,sparsity):
    super(MLPexpander, self).__init__()
    self.fc1 = ExpanderLinear(3072,512,expandSize=np.floor(0.005*sparsity*3072).astype(int))
    #self.bn1 = nn.BatchNorm1d(512,track_running_stats=True)
    self.fc3 = ExpanderLinear(512,256,expandSize=np.floor(0.005*sparsity*512).astype(int))
    #self.bn3 = nn.BatchNorm1d(256,track_running_stats=True)
    self.fc5 = ExpanderLinear(256,128,expandSize=np.floor(0.005*sparsity*256).astype(int))
    #self.bn5 = nn.BatchNorm1d(128,track_running_stats=True)    
    self.fc7 = ExpanderLinear(128,64,expandSize=np.floor(0.005*sparsity*128).astype(int))
    #self.bn7 = nn.BatchNorm1d(64,track_running_stats=True)
    self.fc8 = ExpanderLinear(64,10,expandSize=np.floor(0.01*sparsity*64).astype(int))

  def forward(self,x):
    x = x.view(x.shape[0], -1)
    x = F.relu((self.fc1(x)))
    x = F.relu((self.fc3(x)))
    x = F.relu((self.fc5(x)))
    x = F.relu(self.fc7(x))
    x = self.fc8(x)
    return F.log_softmax(x,dim=1)

#MLP model

class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
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
    return F.log_softmax(x,dim=1)


class MLP_new(nn.Module):
  def __init__(self,X):
    super(MLP_new, self).__init__()
    self.fc1 = SkipLinear(3072,512,matrix = X)
    self.fc2 = SkipLinear(512,256,matrix = X)
    self.fc3 = SkipLinear(256,128,matrix = X)
    self.fc4 = SkipLinear(128,64,matrix = X)
    self.fc5 = SkipLinear(64,10,matrix = X)

    self.skip1 = SkipLinear(3072,256,matrix = X)
    self.skip2 = SkipLinear(3072,128,matrix = X)
    self.skip3 = SkipLinear(3072,64,matrix = X)
    self.skip4 = SkipLinear(3072,10,matrix = X)
    self.skip5 = SkipLinear(512,128,matrix = X)
    self.skip6 = SkipLinear(512,64,matrix = X)
    self.skip7 = SkipLinear(512,10,matrix = X)
    self.skip8 = SkipLinear(256,64,matrix = X)
    self.skip9 = SkipLinear(256,10,matrix = X)
    self.skip10 = SkipLinear(128,10,matrix = X)

  def forward(self,x):
    x = x.view(x.shape[0], -1)
    a_3072 = F.relu((self.fc1(x)))
    a_512 = F.relu((self.fc2(a_3072))+self.skip1(x))   ##Input to 512x256
    a_256 = F.relu((self.fc3(a_512))+F.relu(self.skip5(a_3072))+F.relu(self.skip2(x)))
    a_128 = F.relu((self.fc4(a_256))+F.relu(self.skip6(a_3072))+F.relu(self.skip8(a_512))+F.relu(self.skip3(x)))
    x_64 = self.fc5(a_128)+F.relu(self.skip7(a_3072))+F.relu(self.skip9(a_512))+F.relu(self.skip10(a_256)+F.relu(self.skip4(x)))  ##Input to 10x1
    return F.log_softmax(x_64,dim=1)




