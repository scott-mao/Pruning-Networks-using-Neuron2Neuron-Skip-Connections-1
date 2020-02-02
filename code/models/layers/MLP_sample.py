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

#MLP model

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

