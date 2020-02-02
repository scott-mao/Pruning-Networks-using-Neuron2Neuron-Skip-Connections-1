import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.expandergraphlayer import ExpanderConv2d,ExpanderLinear
from models.layers.skipconnectionlayers import SkipLinear, SkipConv2d
from adj_matrix import adj_matrix
import numpy as np

X = np.load("X_4_10.npy")
#print(X.shape)


def expconv5x5(in_planes, out_planes, sparsity, stride=1):
    "5x5 convolution with padding"
    return ExpanderConv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2,expandSize=int(out_planes*sparsity/100))
def expconv1x1(in_planes, out_planes, sparsity, stride=1):
    "1x1 convolution with padding"
    return ExpanderConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, expandSize=int(out_planes*sparsity/100))

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,padding=2)
def conv5x5(in_planes, out_planes,stride=1):
    "5x5 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,padding=2)

def conv3x3(in_planes, out_planes,stride=3):
    "7x7 convolution without padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=9, stride=stride,padding=0)

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

class Netexpander(nn.Module):
  def __init__(self,sparsity,stride=1):
    super(Netexpander, self).__init__()
    self.conv1 = conv3x3(3, 64)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.pool2 = nn.MaxPool2d(4, 4)
    self.conv2 = expconv5x5(64, 128,sparsity,stride)
    self.conv3 = expconv5x5(128, 256, sparsity,stride)
    self.conv4 = expconv5x5(256, 400,sparsity,stride)
    self.conv5 = expconv5x5(400, 512, sparsity,stride)
    self.fc1 = ExpanderLinear(512 * 1 * 1, 120,expandSize= np.floor(512*sparsity/100).astype(int))
    self.fc2 = ExpanderLinear(120, 84,expandSize= np.floor(120*sparsity/100).astype(int))
    self.fc3 = ExpanderLinear(84, 10,expandSize= np.floor(84*sparsity/100).astype(int))

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
    #print(x.shape)
    x = x.view(-1, 512 * 1 * 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x



#X = adj_matrix(Netexpander(sparsity=20),10)



def Skipconv5x5(in_planes, out_planes, stride=1):
    "5x5 convolution with padding"
    return SkipConv2d(in_planes, out_planes, X, kernel_size=5, stride=stride,padding=2)
def Skipconv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return SkipConv2d(in_planes, out_planes, X, kernel_size=1, stride=stride,
                     padding=2)


class Net_new(nn.Module):
  def __init__(self,sparsity):
    super(Net_new, self).__init__()
    self.conv1 = conv3x3(3, 64)
    self.pool1 = nn.MaxPool2d(2,2)
    self.pool2 = nn.MaxPool2d(4, 4)
    self.conv2 = conv5x5(64, 128)
    self.conv3 = conv5x5(128, 256)
    self.conv4 = conv5x5(256, 400)
    self.conv5 = conv5x5(400, 512)

    self.skip1 = expconv5x5(64, 256,sparsity, stride=1)

    self.skip2 = expconv5x5(64, 400,sparsity,stride=1)
    self.skip3 = expconv5x5(64, 512,sparsity,stride=1)

    self.skip4 = expconv5x5(128, 400,sparsity,stride=1)
    self.skip5 = expconv5x5(128, 512,sparsity,stride=1)

    self.skip6 = expconv5x5(256, 512,sparsity,stride=1)

    self.fc1 = nn.Linear(512 * 1 * 1, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

    self.skip7 = ExpanderLinear(512 * 1 * 1, 84,int(512*sparsity/100))
    self.skip8 = ExpanderLinear(512 * 1 * 1, 10,int(512*sparsity/100))
    self.skip9 = ExpanderLinear(120, 10,int(120*sparsity/100))

  def forward(self, x, sparsity):
    a_64 = self.pool1(F.relu(self.conv1(x)))  
    a_128 = F.relu(self.conv2(a_64))      
    a_256 = F.relu(self.conv3(a_128) + self.skip1(a_64))
    a_400 = F.relu(self.conv4(a_256) + self.skip4(a_128))# + self.skip2(a_64) + self.skip4(a_128))
    test = F.relu(self.conv5(a_400))
    a_512 = self.pool2(F.relu(test + self.skip6(a_256)))# + self.skip3(a_64) + self.skip5(a_128) + self.skip6(a_256))
    a_512_ = a_512.view(-1, 512 * 1 * 1)
    a_120 = F.relu(self.fc1(a_512_))
    a_84 = F.relu(self.fc2(a_120) + self.skip7(a_512_))
    x_10 = self.fc3(a_84) + self.skip9(a_120)# + self.skip8(a_512)
    ##Getting the activation differences
    activation_diff1 = F.relu(self.conv3(a_128)) - self.skip1(a_64)
    activation_diff2 = F.relu(self.conv4(a_256)) - self.skip4(a_128)
    activation_diff3 = F.relu(self.conv5(a_400)) - self.skip6(a_256)
    activation_diff4 = F.relu(self.fc2(a_120)) - self.skip7(a_512_)
    activation_diff5 = F.relu(self.fc3(a_84)) - self.skip9(a_120)

    torch.save(activation_diff1,f'AD1-{sparsity}')
    torch.save(activation_diff2,f'AD2-{sparsity}')
    torch.save(activation_diff3,f'AD3-{sparsity}')
    torch.save(activation_diff4,f'AD4-{sparsity}')
    torch.save(activation_diff5,f'AD5-{sparsity}')
    torch.save(a_256,f'A1-{sparsity}') 
    torch.save(a_400,f'A2-{sparsity}')
    torch.save(a_512,f'A3-{sparsity}')
    torch.save(a_84,f'A4-{sparsity}')
    torch.save(x_10,f'A5-{sparsity}')

    return F.log_softmax(x_10,dim=1)


