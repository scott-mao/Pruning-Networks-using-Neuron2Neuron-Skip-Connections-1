import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable, Function
import torch.nn.functional as F
import torch.optim as optim
from adj_matrix import adj_matrix
from models.MLP import MLP,MLP_new,MLPexpander
from models.MLP_lenet import MLP_lenet,MLPexpander_lenet,MLP_new_lenet
from models.layers.skipconnectionlayers import SkipLinear,NewLinear
from datasets.load_data import LoadCIFAR10
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torchvision
from torchvision import datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
from collections import Counter
import copy
from train_lenet import train,train1,train_conv,validation,validation1,validation_conv
import logging
import matplotlib.pyplot as plt
import argparse
from models.LeNet import LeNet5


##Getting the models

X_1 = np.load('logs/X_1_lenet.npy')
X_2 = np.load('logs/X_2_lenet.npy')
X_3 = np.load('logs/X_3_lenet.npy')
X_100 = np.load('logs/X_100_lenet.npy')
model_conv = LeNet5()

counter = 0
expander_mean = normal_mean = lenet1_mean  = lenet2_mean = lenet3_mean = lenet100_mean = conv_mean = 0
maximum_lenet1 = maximum_lenet2 = maximum_lenet3 = maximum_normal = maximum_lenet100 = maximum_conv = maximum_expander = -1000
minimum_lenet1 = minimum_lenet2 = minimum_lenet3 = minimum_normal = minimum_lenet100 = minimum_conv= minimum_expander = 1000

while(counter<3):
  model_lenet = MLP_lenet()
  lenet1 = MLP_new_lenet(X_1).cuda().double()
  lenet2 = MLP_new_lenet(X_2).cuda().double()
  lenet3 = MLP_new_lenet(X_3).cuda().double()
  lenet100 = MLP_new_lenet(X_100).cuda().double()
  model_x = MLPexpander_lenet(sparsity = 4).cuda().double()
  #model_conv = LeNet5()
  print(f"************************* Pass {counter+1} ******************************")
  
  print("*******************Training LeNet-5******************")
  conv_training_loss,val_conv = train_conv(model_conv)

  conv_accuracy = validation_conv(model_conv)
  conv_mean += conv_accuracy

  if(conv_accuracy > maximum_conv):
    maximum_conv = conv_accuracy
  if(conv_accuracy < minimum_conv):
    minimum_conv = conv_accuracy
  
  '''
  print("*******************Training the Expander Model - 4% ******************")
  expander_training_loss,val_expander = train1(model_x)

  expander_accuracy = validation1(model_x)
  expander_mean += expander_accuracy

  if(expander_accuracy > maximum_expander):
    maximum_expander = expander_accuracy
  if(expander_accuracy < minimum_expander):
    minimum_expander = expander_accuracy
  '''
  print("*******************Training the Normal model******************")
  normal_training_loss,val_normal = train(model_lenet)

  normal_accuracy = validation(model_lenet)
  normal_mean += normal_accuracy

  if(normal_accuracy > maximum_normal):
    maximum_normal = normal_accuracy
  if(normal_accuracy < minimum_normal):
    minimum_normal = normal_accuracy

  print("*******************Training the Lenet Model - 1% (0.5+0.5) ******************")
  lenet100_training_loss,val_lenet100 = train1(lenet100)

  lenet100_accuracy = validation1(lenet100)
  lenet100_mean += lenet100_accuracy

  if(lenet100_accuracy > maximum_lenet100):
    maximum_lenet100 = lenet100_accuracy
  if(lenet100_accuracy < minimum_lenet100):
    minimum_lenet100 = lenet100_accuracy

  print("*******************Training the Lenet Model - 2% (1+1) ******************")
  lenet1_training_loss,val_lenet1 = train1(lenet1) 

  lenet1_accuracy = validation1(lenet1)
  lenet1_mean += lenet1_accuracy

  if(lenet1_accuracy > maximum_lenet1):
    maximum_lenet1 = lenet1_accuracy
  if(lenet1_accuracy < minimum_lenet1):
    minimum_lenet1 = lenet1_accuracy

  print("*******************Training the Lenet Model - 3% (1.5+1.5) ******************")
  lenet3_training_loss,val_lenet3 = train1(lenet3)

  lenet3_accuracy = validation1(lenet3)
  lenet3_mean += lenet3_accuracy

  if(lenet3_accuracy > maximum_lenet3):
    maximum_lenet3 = lenet3_accuracy
  if(lenet3_accuracy < minimum_lenet3):
    minimum_lenet3 = lenet3_accuracy

  print("*******************Training the Lenet Model - 4% (2+2) ******************")
  lenet2_training_loss,val_lenet2 = train1(lenet2)

  lenet2_accuracy = validation1(lenet2)
  lenet2_mean += lenet2_accuracy

  if(lenet2_accuracy > maximum_lenet2):
    maximum_lenet2 = lenet2_accuracy
  if(lenet2_accuracy < minimum_lenet2):
    minimum_lenet2 = lenet2_accuracy

  counter += 1

print("************************** END OF ALL PASSES ***************************")

counter = 3
normal_mean /= counter
lenet1_mean /= counter
lenet2_mean /= counter
lenet3_mean /= counter
lenet100_mean /= counter
conv_mean /= counter
expander_mean /= counter

print("LeNet-5 Final Accuracy:  ",normal_conv, "+-", max(abs(conv_mean-minimum_conv),abs(conv_mean-maximum_conv)))
print("MLP_Lenet 1% Final Accuracy:  ",lenet100_mean, "+-",max(abs(lenet100_mean-minimum_lenet100),abs(lenet100_mean-maximum_lenet100)))
print("MLP_Lenet 2% Final Accuracy:  ",lenet1_mean, "+-",max(abs(lenet1_mean-minimum_lenet1),abs(lenet1_mean-maximum_lenet1)))
print("MLP_Lenet 3% Final Accuracy:  ",lenet3_mean, "+-",max(abs(lenet3_mean-minimum_lenet3),abs(lenet3_mean-maximum_lenet3)))
print("MLP_Lenet 4% Final Accuracy:  ",lenet2_mean, "+-",max(abs(lenet2_mean-minimum_lenet2),abs(lenet2_mean-maximum_lenet2)))
#print("Expander 4% Final Accuracy:  ",expander_mean, "+-", max(abs(expander_mean-minimum_expander),abs(expander_mean-maximum_expander)))
#print("Expander 2% Final Accuracy:  ",expander_mean, "+-", max(abs(expander_mean-minimum_expander),abs(expander_mean-maximum_expander)))
print("Normal Final Accuracy:  ",normal_mean, "+-", max(abs(normal_mean-minimum_normal),abs(normal_mean-maximum_normal)))

