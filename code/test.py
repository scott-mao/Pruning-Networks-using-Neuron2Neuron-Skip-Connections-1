import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, Function
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from models.layers.expandergraphlayer import ExpanderLinear,ExpanderConv2d
from models.conv import Net,Netexpander,Net_new
#from adj_matrix import adj_matrix
from train import validation_conv
#from main_conv import train,train1
import time
#from Random import RandomLinear,RandomConv2d
#from models.layers.skipconnectionlayers import SkipLinear,SkipConv2d
import os
#from LRA import low_rank
#from EVD_functions import skip_matrix,EVD
from models.ResNet import resnet18,resnet34,resnet50,resnet101,resnet152
from models.ResNetexpander import resnet34expander,resnet50expander
from models.ResNetSkip import resnet34skip,resnet50skip
from models.vggnew import vgg11,vgg13,vgg16,vgg19
from models.vggexpander import vggexpander11,vggexpander13,vggexpander16,vggexpander19
from models.vggskip import vggskip16
from models.AlexNet import AlexNet
from models.AlexNetSkip import alexnetskip
from models.AlexNetexpander import alexnetexpander

X = np.load('X_4_10.npy')

def load_checkpoint(model,path,start_epoch):
    if os.path.exists(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        i = checkpoint['batch']
        loss = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']+1))
    else:
        print("No loading checkpoint found")

    return model, start_epoch

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=5,shuffle=True, num_workers=2)
valset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(valset, batch_size=10,shuffle=True, num_workers=2)

alexnet = AlexNet().cuda()
alexnetexpander = alexnetexpander(sparsity = 10).cuda()
alexnetskip = alexnetskip(sparsity = 10).cuda()

resnet34 = resnet34().cuda()
resnet34expander = resnet34expander(sparsity = 5).cuda()
resnet34skip = resnet34skip(sparsity=5).cuda()

resnet50 = resnet50().cuda()
resnet50expander = resnet50expander(sparsity = 10).cuda()
resnet50skip = resnet50skip(sparsity=10).cuda()

vgg16 = vgg16().cuda()
vggexpander16 = vggexpander16(sparsity = 10).cuda()
vggskip16 = vggskip16(sparsity = 2).cuda()
#model_10 = Netexpander(sparsity=45).cuda()
model = Net().cuda()
modelx = Netexpander(sparsity=10).cuda()
#Wprune = adj_matrix(modelx,10)
#W_binary = adj_matrix(model,10)

#Wprune,eprune,vprune = EVD(modelx)
#raw = low_rank(Wprune,eprune,vprune)

#skip_binary_4, X = skip_matrix(raw,5,10,Netexpander,model)
#print(100*np.count_nonzero(X)/np.count_nonzero(W_binary))


modelskip = Net_new(sparsity = 10).cuda()


#W_binary = adj_matrix(model,10)
#Wexpander = adj_matrix(modelx,10)
#Wskip = adj_matrix(modelskip,10)


#print(100*np.count_nonzero(Wexpander)/np.count_nonzero(W_binary))
#print(100*np.count_nonzero(Wprune)/np.count_nonzero(W_binary))


epochs = 20

def train(model,start_epoch,path):
  maximum = -1000
  for epoch in range(start_epoch,epochs):
          since1 = initial = time.time()
          criterion = nn.CrossEntropyLoss()
          optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
          scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.5)
          #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
          running_loss = 0.0
          #maximum = -1000
          for i, data in enumerate(train_loader, 0):
                  #normal = time.time()
                  inputs, labels = data
                  inputs,labels = inputs.cuda(),labels.cuda()
                  # zero the parameter gradients
                  optimizer.zero_grad()
                  # forward + backward + optimize

                  with torch.autograd.set_detect_anomaly(True):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                  # print statistics
                  running_loss += loss.item()
                  if i % 1000 == 999:    # print every 1000 mini-batches
                          print('Epoch {}/{} Batch {}/{} loss: {}'.format(epoch + 1, epochs,np.int_((i + 1)/1000),np.int_(len(train_loader)/1000), running_loss /1000))
                          running_loss = 0.0
                          print("Time per batch: ", time.time()-since1)
                          since1 = time.time()
          print("\n")
          print("Time per epoch: ", time.time()-initial)
          accuracy = validation_conv(model)
          print("\n")
          torch.save({'model_state_dict':model.state_dict(),'epoch':epoch+1,'optimizer_state_dict':optimizer.state_dict(),'batch':i,'losslogger':loss},path)
          np.save('logs/maximum.npy',maximum)
          if(maximum<accuracy):
                  maximum = accuracy
          np.save('logs/maximum.npy',maximum)
          #print("Accuracy: ", accuracy)
  maximum = np.load('logs/maximum.npy')
  print("Maximum accuracy: ",maximum)

if os.path.exists('./skip'):
  os.remove('./skip')
'''
print("***********************Normal Model*************************")
if not os.path.exists('Normal_new.pth.tar'):
  train(model,0,'Normal_new.pth.tar')
else:
  model,start_epoch = load_checkpoint(model,'Normal_new.pth.tar',0)
  train(model,start_epoch,'Normal_new.pth.tar')

print("***********************Expander Model*************************")

if not os.path.exists('Expander.pth.tar'):
  train(modelx,0,'Expander.pth.tar')
else:
  modelx,start_epoch = load_checkpoint(modelx,'Expander.pth.tar',0)
  train(modelx,start_epoch,'Expander.pth.tar')

print("***********************Skip Model*************************")

if not os.path.exists('Skip.pth.tar'):
  train(modelskip,0,'Skip.pth.tar')
else:
  modelskip,start_epoch = load_checkpoint(modelskip,'Skip.pth.tar',0)
  train(modelskip,start_epoch,'Skip.pth.tar')

print("***********************AlexNet*************************")
if not os.path.exists('AlexNet.pth.tar'):
  train(alexnet,0,'AlexNet.pth.tar')
else:
  alexnet,start_epoch = load_checkpoint(alexnet,'AlexNet.pth.tar',0)
  train(alexnet,start_epoch,'AlexNet.pth.tar')

print("***********************AlexNet Skip*************************")
if not os.path.exists('AlexNetSkip.pth.tar'):
  train(alexnetskip,0,'AlexNetSkip.pth.tar')
else:
  alexnetskip,start_epoch = load_checkpoint(alexnetskip,'AlexNetSkip.pth.tar',0)
  train(alexnetskip,start_epoch,'AlexNetSkip.pth.tar')

print("***********************AlexNet Expander*************************")
if not os.path.exists('AlexNetexpander.pth.tar'):
  train(alexnetexpander,0,'AlexNetexpander.pth.tar')
else:
  alexnetexpander,start_epoch = load_checkpoint(alexnetexpander,'AlexNetexpander.pth.tar',0)
  train(alexnetexpander,start_epoch,'AlexNetexpander.pth.tar')

print("***********************ResNet 34 Expander*************************")
if not os.path.exists('ResNet34expander.pth.tar'):
  train(resnet34expander,0,'ResNet34expander.pth.tar')
else:
  resnet34expander,start_epoch = load_checkpoint(resnet34expander,'ResNet34expander.pth.tar',0)
  train(resnet34expander,start_epoch,'ResNet34expander.pth.tar')

print("***********************ResNet 34 Skip*************************")
if not os.path.exists('ResNet34skip.pth.tar'):
  train(resnet34skip,0,'ResNet34skip.pth.tar')
else:
  resnet34skip,start_epoch = load_checkpoint(resnet34skip,'ResNet34skip.pth.tar',0)
  train(resnet34skip,start_epoch,'ResNet34skip.pth.tar')

print("***********************ResNet 34*************************")
if not os.path.exists('ResNet34.pth.tar'):
  train(resnet34,0,'ResNet34.pth.tar')
else:
  resnet34,start_epoch = load_checkpoint(resnet34,'ResNet34.pth.tar',0)
  train(resnet34,start_epoch,'ResNet34.pth.tar')

print("***********************ResNet 50 Skip*************************")
if not os.path.exists('ResNet50skip.pth.tar'):
  train(resnet50skip,0,'ResNet50skip.pth.tar')
else:
  resnet50skip,start_epoch = load_checkpoint(resnet50skip,'ResNet50skip.pth.tar',0)
  train(resnet50skip,start_epoch,'ResNet50skip.pth.tar')

print("***********************ResNet 50*************************")
if not os.path.exists('ResNet50.pth.tar'):
  train(resnet50,0,'ResNet50.pth.tar')
else:
  resnet50,start_epoch = load_checkpoint(resnet50,'ResNet50.pth.tar',0)
  train(resnet50,start_epoch,'ResNet50.pth.tar')

print("***********************VGGskip 16*************************")
if not os.path.exists('VGGskip16.pth.tar'):
  train(vggskip16,0,'VGGskip16.pth.tar')
else:
  vggskip16,start_epoch = load_checkpoint(vggskip16,'VGGskip16.pth.tar',0)
  train(vggskip16,start_epoch,'VGGskip16.pth.tar')

print("***********************VGG 16*************************")
if not os.path.exists('VGG16.pth.tar'):
  train(vgg16,0,'VGG16.pth.tar')
else:
  vgg16,start_epoch = load_checkpoint(vgg16,'VGG16.pth.tar',0)
  train(vgg16,start_epoch,'VGG16.pth.tar')
'''
print("***********************VGGexpander 16*************************")
if not os.path.exists('VGGexpander16.pth.tar'):
  train(vggexpander16,0,'VGGexpander16.pth.tar')
else:
  vggexpander16,start_epoch = load_checkpoint(vggexpander16,'VGGexpander16.pth.tar',0)
  train(vggexpander16,start_epoch,'VGGexpander16.pth.tar')
'''
print("***********************VGG 19*************************")
if not os.path.exists('VGG19.pth.tar'):
  train(vgg19,0,'VGG19.pth.tar')
else:
  vgg19,start_epoch = load_checkpoint(vgg19,'VGG19.pth.tar',0)
  train(vgg19,start_epoch,'VGG19.pth.tar')
'''

